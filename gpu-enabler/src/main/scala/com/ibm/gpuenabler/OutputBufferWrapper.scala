package com.ibm.gpuenabler

import jcuda.{CudaException, Pointer}
import jcuda.driver.{CUdeviceptr, CUresult, CUstream}
import jcuda.runtime.{JCuda, cudaStream_t}
import org.apache.spark.gpuenabler.CUDAUtils

trait OutputBufferWrapper[T] extends Iterator[T] with CUDAUtils._Logging {

  protected var idx: Int = 0
  protected var gpuPtr: Option[Pointer] = None
  protected var devPtr: Option[CUdeviceptr] = None
  protected var cpuPtr: Option[Pointer] = None
  protected var outputArray: Option[Array[T]] = None
  protected var byteSize: Option[Int] = None
  protected var numElems: Option[Int] = None
  protected var transpose: Boolean = false

  def next: T = {
    val nextVal = outputArray.get(idx)
    idx += 1
    nextVal
  }

  def hasNext: Boolean = idx < outputArray.get.length

  def getKernelParams: Seq[Pointer] = Seq(gpuPtr.get)

  def allocGPUMem(): Unit = {
    devPtr = CUDABufferUtils.allocGPUMem(byteSize.get)
    gpuPtr = Some(Pointer.to(devPtr.get))
  }

  def setTranspose(trans: Boolean): Unit = {
    transpose = trans
  }

  def getByteSize: Int = byteSize.get

  def getGpuPtr: Pointer = gpuPtr.get

  def getOutputArray: Array[T] = outputArray.get

  // Copy data from GPU to CPU
  def gpuToCpu(stream: CUstream): Unit

  def freeGPUMem(): Unit = {
    JCuda.cudaFree(gpuPtr.get)
  }

  def freeCPUMem(): Unit = {
    JCuda.cudaFreeHost(cpuPtr.get)
  }
}