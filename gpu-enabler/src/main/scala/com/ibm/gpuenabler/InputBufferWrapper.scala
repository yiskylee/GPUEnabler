package com.ibm.gpuenabler

import jcuda.driver.CUresult
import jcuda.{CudaException, Pointer}
import jcuda.runtime.JCuda

abstract class InputBufferWrapper[T] {

  val size: Int
  protected var gpuPtr: Option[Pointer] = None

  // Return the GPU Pointer of this buffer
  def getGPUPointer: Option[Pointer] = gpuPtr

  // Return the size of GPU allocation
  def getSize: Int = size

  // Allocate GPU Memory for this buffer and return its pointer on the GPU
  def allocGPUMem(): Unit = {
    val ptr = new Pointer()
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
    gpuPtr = Some(ptr)
  }

  // Copy data from CPU to GPU
  def copyToGPUMem(memType: String, transpose: Boolean): Unit

}