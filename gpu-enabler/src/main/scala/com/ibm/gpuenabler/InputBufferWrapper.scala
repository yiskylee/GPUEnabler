package com.ibm.gpuenabler

import jcuda.driver.{CUresult, CUstream}
import jcuda.{CudaException, Pointer}
import jcuda.runtime.{JCuda, cudaMemcpyKind, cudaStream_t}

trait InputBufferWrapper[T] {

  protected var gpuPtr: Option[Pointer] = None
  protected var cpuPtr: Option[Pointer] = None
  protected var size: Option[Int] = None
  protected val stream: cudaStream_t = {
    val stream = new cudaStream_t
    JCuda.cudaStreamCreateWithFlags(stream, JCuda.cudaStreamNonBlocking)
    stream
  }

  protected val cuStream: CUstream = new CUstream(stream)

  def getStream: cudaStream_t = stream

  def getCuStream: CUstream = cuStream

  def getKernelParams: Seq[Pointer] = List(gpuPtr.get)

  def allocCPUPinnedMem(): Unit = {
    cpuPtr = Some(CUDABufferUtils.allocCPUPinnedMem(size.get))
  }

  def allocGPUMem(): Unit = {
    gpuPtr = Some(CUDABufferUtils.allocGPUMem(size.get))
  }

  def getSize: Int = size.get

  // Copy data from CPU to GPU
  def cpuToGpu(transpose: Boolean): Unit
}