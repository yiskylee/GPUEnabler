/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.ibm.gpuenabler

import java.nio.ByteOrder
import java.io.{PrintWriter, StringWriter}

import jcuda.driver.JCudaDriver._
import jcuda.driver.{CUdeviceptr, CUresult, CUstream}
import jcuda.runtime.{JCuda, cudaStream_t}
import jcuda.{CudaException, Pointer}
import org.apache.spark.storage.BlockId
import org.apache.spark.storage.RDDBlockId
import breeze.linalg.{DenseVector => BrDenseVector}
import org.apache.spark.gpuenabler.CUDAUtils
import org.apache.spark.mllib.linalg.DenseVector

import scala.collection.mutable.ArrayBuffer
import scala.language.existentials
import scala.reflect.ClassTag
import scala.collection.mutable.HashMap

// scalastyle:off no.finalize
private[gpuenabler] case class KernelParameterDesc(cpuArr: Array[_ >: Byte with Short with Int
                                                      with Float with Long with Double <: AnyVal],
                                                    cpuPtr: Pointer,
                                                    devPtr: CUdeviceptr,
                                                    gpuPtr: Pointer,
                                                    sz: Int)

private[gpuenabler] class HybridIterator[T: ClassTag](inputArr: Array[T],
                                                      __columnsOrder: Seq[DataSchema],
                                                      _blockId: Option[BlockId],
                                                      cuStream: CUstream = null,
                                                      numentries: Int = 0,
                                                      outputArraySizes: Seq[Int] = null
                                                     ) extends Iterator[T] with CUDAUtils._Logging {

  private var _arr: Array[T] = inputArr

  def arr: Array[T] = if (_arr == null) {
    // Validate the CPU pointers before deserializing
    copyGpuToCpu
    _arr = CPUIterTimer.time(getResultList, "getResultList")
    _arr
  } else {
    _arr
  }
  def getName: String = __columnsOrder(0).name

  private var columnsOrder = __columnsOrder

  private val _outputArraySizes = if (outputArraySizes != null) {
    outputArraySizes
  } else {
    val tempbuf = new ArrayBuffer[Int]()
    // if outputArraySizes is not provided by user program; create one
    // based on the number of columns and initialize it to '1' to denote
    // the object has only one element in it.
    columnsOrder.foreach(_ => tempbuf += 1)
    tempbuf
  }

  private val _cuStream = if (cuStream != null) {
    cuStream
  } else {
    val stream = new cudaStream_t
    JCuda.cudaStreamCreateWithFlags(stream, JCuda.cudaStreamNonBlocking)
    new CUstream(stream)
  }

  var _numElements: Int = if (inputArr != null) inputArr.length else 0
  var idx: Int = -1

  val blockId: Option[BlockId] = _blockId match {
    case Some(x) => _blockId
    case None =>
      val r = scala.util.Random
      Some(RDDBlockId(r.nextInt(99999), r.nextInt(99999)))
  }

  def rddId: Int = _blockId.getOrElse(RDDBlockId(0, 0)).asRDDId.get.rddId

  def cachedGPUPointers: HashMap[String, KernelParameterDesc] =
    GPUSparkEnv.get.gpuMemoryManager.getCachedGPUPointers

  private def gpuCache: Boolean = GPUSparkEnv.get.gpuMemoryManager.cachedGPURDDs.contains(rddId)

  def numElements: Int = _numElements

  def hasNext: Boolean = {
    idx < arr.length - 1
  }

  def next: T = {
    idx = idx + 1
    arr(idx)
  }

  def getCuStream: CUstream = _cuStream

  // Function to free the allocated GPU memory if the RDD is not cached.
  def freeGPUMemory(): Unit = {
    if (!gpuCache) {
      // Make sure the CPU ptrs are populated before GPU memory is freed up.
      copyGpuToCpu
      if (_listKernParmDesc == null) return
      _listKernParmDesc = _listKernParmDesc.map(kpd => {
        if (kpd.devPtr != null) {
          GPUSparkEnv.get.cudaManager.freeGPUMemory(kpd.devPtr)
        }
        KernelParameterDesc(kpd.cpuArr, kpd.cpuPtr, null, null, kpd.sz)
      })

      for (name <- cachedGPUPointers.keys if name.startsWith(blockId.get.toString))
        logInfo(s"$name is removed from cachedGPUPointers")

      cachedGPUPointers.retain(
        (name, kernelParameterDesc) => !name.startsWith(blockId.get.toString))

    }
  }

  // TODO: Discuss the need for finalize; how to handle streams;
  //  override def finalize(): Unit = {
  //    JCuda.cudaStreamDestroy(stream)
  //    super.finalize
  //  }

  // This function is used to copy the CPU memory to GPU for
  // an existing Hybrid Iterator
  def copyCpuToGpu(): Unit = {
    if (_listKernParmDesc == null) return
    _listKernParmDesc = _listKernParmDesc.map(kpd => {
      if (kpd.devPtr == null) {
        val devPtr = GPUSparkEnv.get.cudaManager.allocateGPUMemory(kpd.sz)
        // XILI
        cuMemcpyHtoDAsync(devPtr, kpd.cpuPtr, kpd.sz, _cuStream)
        // XILI
        val gPtr = Pointer.to(devPtr)
        KernelParameterDesc(kpd.cpuArr, kpd.cpuPtr, devPtr, gPtr, kpd.sz)
      } else {
        kpd
      }
    })
  }

  // This function is used to copy the GPU memory to CPU for
  // an existing Hybrid Iterator
  def copyGpuToCpu: Unit = {
    // Ensure main memory is allocated to hold the GPU data
    if (_listKernParmDesc == null) return
    _listKernParmDesc = (_listKernParmDesc, columnsOrder).
      zipped.map((kpd, col) => {
      if (kpd.cpuArr == null && kpd.cpuPtr == null && kpd.devPtr != null) {
        val (cpuArr, cpuPtr: Pointer) = col.dataType match {
          case c if c == "Int" => {
            val y = new Array[Int](kpd.sz / INT_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == SHORT_COLUMN => {
            val y = new Array[Short](kpd.sz / SHORT_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == BYTE_COLUMN => {
            val y = new Array[Byte](kpd.sz / BYTE_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == LONG_COLUMN => {
            val y = new Array[Long](kpd.sz / LONG_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == FLOAT_COLUMN => {
            val y = new Array[Float](kpd.sz / FLOAT_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == "Double" => {
            val y = new Array[Double](kpd.sz / DOUBLE_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == INT_ARRAY_COLUMN => {
            val y = new Array[Int](kpd.sz / INT_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == LONG_ARRAY_COLUMN => {
            val y = new Array[Long](kpd.sz / LONG_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == FLOAT_ARRAY_COLUMN => {
            val y = new Array[Float](kpd.sz / FLOAT_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == "Array[Double]" => {
            val y = new Array[Double](kpd.sz / DOUBLE_COLUMN.bytes)
            (y, Pointer.to(y))
          }
          case c if c == "BrDenseVector[Double]" => {
            val y = new Array[Double](kpd.sz / DOUBLE_COLUMN.bytes)
            (y, Pointer.to(y))
          }
        }
        // XILI
        cuMemcpyDtoHAsync(cpuPtr, kpd.devPtr, kpd.sz, _cuStream)
        // XILI
        KernelParameterDesc(cpuArr, cpuPtr, kpd.devPtr, kpd.gpuPtr, kpd.sz)
      } else {
        kpd
      }
    })
    cuCtxSynchronize()
  }

  // Allocate Memory from Off-heap Pinned Memory and returns
  // the pointer & buffer address pointing to it
  private def allocPinnedHeap(size: Long) = {
    val ptr: Pointer = new Pointer()
    try {
      val result: Int = JCuda.cudaHostAlloc(ptr, size, JCuda.cudaHostAllocPortable)
      if (result != CUresult.CUDA_SUCCESS) {
        throw new CudaException(JCuda.cudaGetErrorString(result))
      }
    }
    catch {
      case ex: Exception => {
        throw new OutOfMemoryError("Could not alloc pinned memory: " + ex.getMessage)
      }
    }
    (ptr, ptr.getByteBuffer(0, size).order(ByteOrder.LITTLE_ENDIAN))
  }

  def genKernParamDesc(input: Array[_], col: DataSchema): KernelParameterDesc = {
    val (hPtr: Pointer, colDataSize: Int) = {
      var bufferOffset = 0
      col match {
        case DataSchema(_, "Array[Double]", arraySize) =>
          val size: Int = inputArr.length * arraySize * DOUBLE_COLUMN.bytes
          val (ptr, buffer) = allocPinnedHeap(size)
          input.asInstanceOf[Array[Array[Double]]].foreach( x => {
            buffer.position(bufferOffset)
            buffer.asDoubleBuffer().put(x, 0, arraySize)
            bufferOffset += arraySize * DOUBLE_COLUMN.bytes
          })
          (ptr, size)
        case DataSchema(_, "DenseVector[Double]", arraySize) =>
          val size: Int = input.length * arraySize * DOUBLE_COLUMN.bytes
          val (ptr, buffer) = allocPinnedHeap(size)
          for (i <- 0 until input.length) {
            val vec = input(i).asInstanceOf[DenseVector].values
            for (j <- 0 until vec.size) {
              buffer.asDoubleBuffer().put(j * input.length + i, vec(j))
            }
          }
          (ptr, size)
        case DataSchema(_, "BrDenseVector[Double]", arraySize) =>
          val size: Int = input.length * arraySize * DOUBLE_COLUMN.bytes
          val (ptr, buffer) = allocPinnedHeap(size)
          for (i <- 0 until input.length) {
            val vec = input(i).asInstanceOf[BrDenseVector[Double]].data
            for (j <- 0 until vec.length) {
              buffer.asDoubleBuffer().put(j * input.length + i, vec(j))
            }
          }
          (ptr, size)
        case DataSchema(_, "Double", 1) =>
          val size: Int = input.length * DOUBLE_COLUMN.bytes
          val (ptr, buffer) = allocPinnedHeap(size)
          for (i <- 0 until input.length) {
            val scalar = input(i).asInstanceOf[Double]
            buffer.asDoubleBuffer().put(i, scalar)
          }
          (ptr, size)
        case DataSchema(_, "Int", 1) =>
          val size: Int = input.length * INT_COLUMN.bytes
          val (ptr, buffer) = allocPinnedHeap(size)
          for (i <- 0 until input.length) {
            val scalar = input(i).asInstanceOf[Int]
            buffer.asIntBuffer().put(i, scalar)
          }
          (ptr, size)
      }
    }
    val devPtr = GPUSparkEnv.get.cudaManager.allocateGPUMemory(colDataSize)
    cuMemcpyHtoDAsync(devPtr, hPtr, colDataSize, _cuStream)
    val gPtr = Pointer.to(devPtr)
    new KernelParameterDesc(null, hPtr, devPtr, gPtr, colDataSize)
  }

  def createInputParmDesc: Seq[KernelParameterDesc] = {
    val tupleSize = columnsOrder.length
    val input: Array[_] =
      if (tupleSize == 1)
        inputArr.asInstanceOf[Array[_]]
      else if (tupleSize == 2)
        inputArr.asInstanceOf[Array[Tuple2[_, _]]]
      else if (tupleSize == 3)
        inputArr.asInstanceOf[Array[Tuple3[_, _, _]]]
      else
        null

    val kernParamDescSeq =
      if (tupleSize == 1) {
        val input = inputArr.asInstanceOf[Array[_]]
        val col = columnsOrder(0)
        val kernelParamDesc = cachedGPUPointers.getOrElseUpdate(blockId.get + col.name, {
          genKernParamDesc(input, col)
        })
        Seq(kernelParamDesc)
      } else if (tupleSize == 2) {
        val input = inputArr.asInstanceOf[Array[Tuple2[_,_]]]
        val (col1, col2) = (columnsOrder(0), columnsOrder(1))
        val (input1, input2) = (input.map(_._1), input.map(_._2))
        val kernelParameterDesc1 = cachedGPUPointers.getOrElseUpdate(blockId.get + col1.name, {
          genKernParamDesc(input1, col1)
        })
        val kernelParameterDesc2 = cachedGPUPointers.getOrElseUpdate(blockId.get + col2.name, {
          genKernParamDesc(input2, col2)
        })
        Seq(kernelParameterDesc1, kernelParameterDesc2)
      } else if (tupleSize == 3) {
        val input = inputArr.asInstanceOf[Array[Tuple3[_,_,_]]]
        val (col1, col2, col3) = (columnsOrder(0), columnsOrder(1), columnsOrder(2))
        val (input1, input2, input3) = (input.map(_._1), input.map(_._2), input.map(_._3))
        val kernelParameterDesc1 = cachedGPUPointers.getOrElseUpdate(blockId.get + col1.name, {
          genKernParamDesc(input1, col1)
        })
        val kernelParameterDesc2 = cachedGPUPointers.getOrElseUpdate(blockId.get + col2.name, {
          genKernParamDesc(input2, col2)
        })
        val kernelParameterDesc3 = cachedGPUPointers.getOrElseUpdate(blockId.get + col3.name, {
          genKernParamDesc(input3, col3)
        })
        Seq(kernelParameterDesc1, kernelParameterDesc2, kernelParameterDesc3)
      } else {
        null
      }
    kernParamDescSeq
  }

  def createOutputParmDesc: Seq[KernelParameterDesc] = {
    // initEmptyArrays - mostly used by output argument list
    // set the number of entries to numentries as its initialized to '0'
    _numElements = numentries
    val colOrderSizes = columnsOrder zip _outputArraySizes
    val seqKernParamDesc = colOrderSizes.map { col =>
      cachedGPUPointers.getOrElseUpdate(blockId.get + col._1.name, {
        val colDataSize: Int = col._1 match {
          case DataSchema(name, "Int", length) =>
            numentries * INT_COLUMN.bytes
          case DataSchema(name, "Double", length) =>
            numentries * DOUBLE_COLUMN.bytes
          case DataSchema(name, "Array[Double]", length) =>
            col._1.length * numentries * DOUBLE_COLUMN.bytes
          case DataSchema(name, "BrDenseVector[Double]", length) =>
            col._1.length * numentries * DOUBLE_COLUMN.bytes
        }
        val devPtr = GPUSparkEnv.get.cudaManager.allocateGPUMemory(colDataSize)
        cuMemsetD32Async(devPtr, 0, colDataSize / 4, cuStream)
        val gPtr = Pointer.to(devPtr)
        // Allocate only GPU memory; main memory will be allocated during deserialization
        new KernelParameterDesc(null, null, devPtr, gPtr, colDataSize)
      })
    }
    seqKernParamDesc
  }

  def listKernParmDesc: Seq[KernelParameterDesc] = _listKernParmDesc

  private var _listKernParmDesc: Seq[KernelParameterDesc] =
    if (inputArr != null && inputArr.length > 0) {
      createInputParmDesc
    } else if (numentries != 0) {
      createOutputParmDesc
    } else {
      null
    }

  def deserializeColumnValue(columnType: String, cpuArr: Array[_ >: Byte with Short with Int
    with Float with Long with Double <: AnyVal], index: Int, outsize: Int = 0): Any = {
    columnType match {
      case "Int" => cpuArr(index).asInstanceOf[Int]
      case "Double" => cpuArr(index).asInstanceOf[Double]
      case "Array[Int]" =>
        val array = new Array[Int](outsize)
        var runIndex = index
        for (i <- 0 until outsize) {
          array(i) = cpuArr(runIndex).asInstanceOf[Int]
          runIndex += 1
        }
        array
      case "Array[Double]" =>
        val array = new Array[Double](outsize)
        var runIndex = index
        for (i <- 0 until outsize) {
          array(i) = cpuArr(runIndex).asInstanceOf[Double]
          runIndex += 1
        }
        array
      case "BrDenseVector[Double]" =>
        val vector = BrDenseVector.zeros[Double](outsize)
        var runIndex = index
        for (i <- 0 until outsize) {
          vector(i) = cpuArr(runIndex).asInstanceOf[Double]
          runIndex += numElements
        }
        vector
    }
  }

  def getResultList: Array[T] = {
    val resultsArray = new Array[T](numElements)
    if (columnsOrder.length == 1) {
//      for (index <- 0 until numElements) {
//        resultsArray(index) =
//          deserializeColumnValue(columnsOrder(0).dataType, listKernParmDesc(0).cpuArr,
//            index * columnsOrder(0).length, columnsOrder(0).length).asInstanceOf[T]
//      }
      for (index <- 0 until numElements) {
        resultsArray(index) =
          deserializeColumnValue(columnsOrder(0).dataType, listKernParmDesc(0).cpuArr,
            index, columnsOrder(0).length).asInstanceOf[T]
      }
    } else {
      val tupleSize = columnsOrder.length
      if (tupleSize == 2) {
        if (columnsOrder(0).dataType == "Int" && columnsOrder(1).dataType == "Double")
          for (index <- 0 until numElements) {
            resultsArray(index) = (listKernParmDesc(0).cpuArr(index).asInstanceOf[Int],
              listKernParmDesc(1).cpuArr(index).asInstanceOf[Double]).asInstanceOf[T]
          }
      }
    }
    resultsArray
  }
}