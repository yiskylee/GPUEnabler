package com.ibm.gpuenabler

import java.nio.ByteOrder

import jcuda.driver.CUresult
import org.apache.spark.mllib.linalg.DenseVector
import jcuda.{CudaException, Pointer}
import jcuda.runtime.JCuda

class DenseVectorInputBufferWrapper(cpuArray: Array[DenseVector])
  extends InputBufferWrapper[DenseVector] {
  private var gpuPtr: Option[Pointer] = None
  private val size: Int = cpuArray.length * cpuArray(0).size * 8
  override def getGPUPointer: Pointer = {
    gpuPtr.getOrElse(allocPinnedMemory)
  }
  override def allocPinnedMemory: Pointer = {
    val ptr: Pointer = new Pointer()
    try {
      val result: Int = JCuda.cudaHostAlloc(ptr, size, JCuda.cudaHostAllocPortable)
      if (result != CUresult.CUDA_SUCCESS) {
        throw new CudaException(JCuda.cudaGetErrorString(result))
      }
    }
    catch {
      case ex: Exception =>
        throw new OutOfMemoryError("Could not alloc pinned memory: " + ex.getMessage)
    }
    ptr
  }
}
