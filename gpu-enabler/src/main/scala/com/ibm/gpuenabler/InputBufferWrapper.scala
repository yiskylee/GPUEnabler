package com.ibm.gpuenabler

import jcuda.driver.{CUdeviceptr, CUresult, CUstream}
import jcuda.{CudaException, Pointer}
import jcuda.runtime.{JCuda, cudaMemcpyKind, cudaStream_t}
import org.apache.spark.gpuenabler.CUDAUtils

trait InputBufferWrapper[T] extends CUDAUtils._Logging {
  protected var gpuPtr: Option[Pointer] = None
  // devPtr must be used because only jcuda driver API allows own kernels
  protected var devPtr: Option[CUdeviceptr] = None
  protected var cpuPtr: Option[Pointer] = None
  protected var byteSize: Option[Int] = None
  protected var numElems: Option[Int] = None
  // TODO: Find a better way to use cuStreamDestroy(stream) to delete the stream
  protected val stream: cudaStream_t = {
    val stream = new cudaStream_t
    JCuda.cudaStreamCreateWithFlags(stream, JCuda.cudaStreamNonBlocking)
    stream
  }
  protected var transpose: Boolean = false

  protected val cuStream: CUstream = new CUstream(stream)

  def getStream: cudaStream_t = stream

  def getCuStream: CUstream = cuStream

  def getKernelParams: Seq[Pointer] = Seq(gpuPtr.get)

  def allocCPUPinnedMem(): Unit = {
    cpuPtr = CUDABufferUtils.allocCPUPinnedMem(byteSize.get)
  }

  def allocGPUMem(): Unit = {
    devPtr = CUDABufferUtils.allocGPUMem(byteSize.get)
    gpuPtr = Some(Pointer.to(devPtr.get))
  }

  def setTranspose(trans: Boolean): Unit = {
    transpose = trans
  }

  def getSize: Int = byteSize.get

  def getNumElems: Int = numElems.get

  def getGpuPtr: Pointer = gpuPtr.get

  // Copy data from CPU to GPU
  def cpuToGpu(): Unit

  // TODO: When to free input buffer's CPU and GPU memory? (If InputBuffer is not cached, we
  // TODO: should free the buffer after the kernel is done)
}