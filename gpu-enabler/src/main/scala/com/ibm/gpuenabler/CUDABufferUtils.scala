package com.ibm.gpuenabler

import jcuda.driver.CUresult
import jcuda.{CudaException, Pointer}
import jcuda.runtime.JCuda
import org.apache.spark.mllib.linalg.DenseVector

import scala.reflect.ClassTag

object CUDABufferUtils {
  def createInputBufferFor[T: ClassTag](inputArray: Array[T]): InputBufferWrapper[T] = {
    val sampleInput = inputArray(0)
    val numElem = inputArray.length
    sampleInput match {
      case _: DenseVector =>
        new DenseVectorInputBufferWrapper(inputArray.asInstanceOf[Array[DenseVector]]).
          asInstanceOf[InputBufferWrapper[T]]
      case _: Tuple2[_, _] =>
        new Tuple2InputBufferWrapper(inputArray.asInstanceOf[Array[Tuple2[_, _]]]).
          asInstanceOf[InputBufferWrapper[T]]
      case _ : Int =>
        new PrimitiveInputBufferWrapper[Int](inputArray.asInstanceOf[Array[Int]], 4).
          asInstanceOf[InputBufferWrapper[T]]
      case _ : Float =>
        new PrimitiveInputBufferWrapper[Float](inputArray.asInstanceOf[Array[Float]], 4).
          asInstanceOf[InputBufferWrapper[T]]
      case _ : Double =>
        new PrimitiveInputBufferWrapper[Double](inputArray.asInstanceOf[Array[Double]], 8).
          asInstanceOf[InputBufferWrapper[T]]
    }
  }

  def createOutputBufferFor[T: ClassTag](sampleOutput: T, numElem: Int)
  : OutputBufferWrapper[T] = {
    sampleOutput match {
      case _: DenseVector =>
        new DenseVectorOutputBufferWrapper(numElem,
          sampleOutput.asInstanceOf[DenseVector].size).asInstanceOf[OutputBufferWrapper[T]]
      case _: Tuple2[_, _] =>
        new Tuple2OutputBufferWrapper(sampleOutput.asInstanceOf[(_, _)], numElem).
          asInstanceOf[OutputBufferWrapper[T]]
      case _: Int =>
        new PrimitiveOutputBufferWrapper[Int](sampleOutput.asInstanceOf[Int], numElem, 4).
          asInstanceOf[OutputBufferWrapper[T]]
      case _: Float =>
        new PrimitiveOutputBufferWrapper[Float](sampleOutput.asInstanceOf[Float], numElem, 4).
          asInstanceOf[OutputBufferWrapper[T]]
      case _: Double =>
        new PrimitiveOutputBufferWrapper[Double](sampleOutput.asInstanceOf[Double], numElem, 8).
          asInstanceOf[OutputBufferWrapper[T]]
    }
//    val className: String = sampleOutput.getClass.getName
//    if (className.equals("org.apache.spark.mllib.linalg.DenseVector")) {
  }

  // Allocate GPU Memory with given size and returns the GPU Pointer
  def allocGPUMem(size: Int): Pointer = {
    val ptr = new Pointer()
    try {
      val result: Int = JCuda.cudaMalloc(ptr, size)
      if (result != CUresult.CUDA_SUCCESS) {
        throw new CudaException(JCuda.cudaGetErrorString(result))
      }
    } catch {
      case ex: Exception =>
        System.err.println(s"Could not alloc pinned memory: ${ex.getMessage}")
        System.exit(1)
    }
    ptr
  }

  def allocCPUPinnedMem(size: Int): Pointer = {
    val ptr = new Pointer()
    try {
      val result: Int = JCuda.cudaHostAlloc(ptr, size, JCuda.cudaHostAllocPortable)
      if (result != CUresult.CUDA_SUCCESS) {
        throw new CudaException(JCuda.cudaGetErrorString(result))
      }
    } catch {
      case ex: Exception =>
        System.err.println(s"Could not alloc pinned memory: ${ex.getMessage}")
        System.exit(1)
    }
    ptr
  }

}
