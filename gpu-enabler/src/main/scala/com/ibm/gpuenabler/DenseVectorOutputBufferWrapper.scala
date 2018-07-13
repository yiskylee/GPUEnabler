package com.ibm.gpuenabler

import jcuda.{CudaException, Pointer}
import jcuda.driver.CUresult
import jcuda.runtime.JCuda
import org.apache.spark.mllib.linalg.DenseVector

class DenseVectorOutputBufferWrapper(numVecs: Int, vecSize: Int)
  extends OutputBufferWrapper[DenseVector] {
  private var gpuPtr: Option[Pointer] = None
  private val size: Int = numVecs * vecSize * 8
//  override def getGPUPointer: Pointer = {
//    gpuPtr.getOrElse(allocPinnedMemory)
//  }
//  override def allocPinnedMemory: Pointer = {
//    val ptr: Pointer = new Pointer()
//    try {
//      val result: Int = JCuda.cudaHostAlloc(ptr, size, JCuda.cudaHostAllocPortable)
//      if (result != CUresult.CUDA_SUCCESS) {
//        throw new CudaException(JCuda.cudaGetErrorString(result))
//      }
//    }
//    catch {
//      case ex: Exception =>
//        throw new OutOfMemoryError("Could not alloc pinned memory: " + ex.getMessage)
//    }
//    ptr
//  }
}