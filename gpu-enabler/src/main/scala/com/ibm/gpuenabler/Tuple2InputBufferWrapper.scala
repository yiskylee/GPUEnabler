package com.ibm.gpuenabler
import java.nio.ByteOrder

import jcuda.driver.CUresult
import org.apache.spark.mllib.linalg.DenseVector
import jcuda.{CudaException, Pointer}
import jcuda.runtime.JCuda

import scala.collection.mutable.ListBuffer
import scala.reflect.ClassTag

class Tuple2InputBufferWrapper[K: ClassTag, V: ClassTag] (inputArray: Array[Tuple2[K, V]])
  extends InputBufferWrapper[Tuple2[K, V]] {

  private val _numTuples = inputArray.length

  val buffer1: InputBufferWrapper[K] = CUDABufferUtils.createInputBufferFor[K](inputArray.map(_._1))
  val buffer2: InputBufferWrapper[V] = CUDABufferUtils.createInputBufferFor[V](inputArray.map(_._2))
  val sizes: ListBuffer[Int] = ListBuffer(buffer1.sizes.head, buffer2.sizes.head)

  override def allocGPUMem(): Unit = {
    buffer1.allocGPUMem()
    buffer2.allocGPUMem()
    val ptr1 = buffer1.gpuPtrs match {
      case Some(ptrs) => ptrs.head
    }
    val ptr2 = buffer2.gpuPtrs match {
      case Some(ptrs) => ptrs.head
    }
    gpuPtrs = Some(ListBuffer(ptr1, ptr2))
  }

  override def copyToGPUMem(memType: String, transpose: Boolean): Unit = {
    buffer1.copyToGPUMem(memType, transpose)
    buffer2.copyToGPUMem(memType, transpose)
  }

}