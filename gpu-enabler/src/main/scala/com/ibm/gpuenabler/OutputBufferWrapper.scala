package com.ibm.gpuenabler

import jcuda.{CudaException, Pointer}
import jcuda.driver.CUresult
import jcuda.runtime.{JCuda, cudaStream_t}

trait OutputBufferWrapper[T] {

  var idx: Int = 0
  protected var gpuPtr: Option[Pointer] = None
  protected var cpuPtr: Option[Pointer] = None
  protected var outputArray: Option[Array[T]] = None
  protected var size: Option[Int] = None

  def next: T = outputArray(idx)

  def hasNext: Boolean = idx < outputArray.get.length

  def getKernelParams: Seq[Pointer] = List(gpuPtr.get)

  def allocGPUMem(): Unit = {
    gpuPtr = Some(CUDABufferUtils.allocGPUMem(size.get))
  }

  def getSize: Int = size.get

  // Copy data from GPU to CPU
  def gpuToCpu(stream: cudaStream_t): Unit

  def freeGPUMem(): Unit = {
    JCuda.cudaFree(gpuPtr.get)
  }

  def freeCPUMem(): Unit = {
    JCuda.cudaFreeHost(cpuPtr.get)
  }

}