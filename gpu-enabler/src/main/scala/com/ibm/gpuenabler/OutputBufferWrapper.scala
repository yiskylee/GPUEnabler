package com.ibm.gpuenabler

import jcuda.{CudaException, Pointer}
import jcuda.driver.{CUdeviceptr, CUresult, CUstream, JCudaDriver}
import jcuda.runtime.{JCuda, cudaStream_t}
import org.apache.spark.gpuenabler.CUDAUtils

trait OutputBufferWrapper[T] extends Iterator[T] with CUDAUtils._Logging {

  protected var idx: Int = -1
  protected var gpuPtr: Option[Pointer] = None
  protected var devPtr: Option[CUdeviceptr] = None
  protected var cpuPtr: Option[Pointer] = None
  protected var outputArray: Option[Array[T]] = None
  protected var byteSize: Option[Int] = None
  protected var numElems: Option[Int] = None
  protected var transpose: Boolean = false
  protected var cuStream: Option[CUstream] = None

  def next: T = {
    idx += 1
    outputArray.get(idx)
  }

  def hasNext: Boolean = {
    idx < outputArray.getOrElse{
      gpuToCpu()
      outputArray.get
    }.length - 1
  }

  // reset() is called when an output buffer is found in cache
  // It sets outputArray buffer to None, and idx to -1, so when next() is called on the buffer
  // we know that the data on outputArray is still the result from last kernel computation,
  // when the output buffer was added to the cache
  def reset(): Unit = {
    outputArray = None
    idx = -1
//    cuStream match {
//      case None =>
//      case Some(stream) => JCudaDriver.cuStreamDestroy(stream)
//    }
    cuStream = None
  }

  // Pass the stream from the corresponding input buffer to this function,
  // so the CtoG => execution => GtoC is executed with the same stream
  def setStream(stream: CUstream): Unit = {
    cuStream = Some(stream)
  }

  def getKernelParams: Seq[Pointer] = Seq(gpuPtr.get)

  def allocGPUMem(): Unit = {
    devPtr = CUDABufferUtils.allocGPUMem(byteSize.get)
    gpuPtr = Some(Pointer.to(devPtr.get))
  }

  def freeGPUMem(): Unit = {
    JCuda.cudaFree(gpuPtr.get)
  }

  def setTranspose(trans: Boolean): Unit = {
    transpose = trans
  }

  def freeCPUMem(): Unit = {
    JCuda.cudaFree(cpuPtr.get)
    //    JCudaDriver.cuMemFreeHost(cpuPtr.get)
  }

  def getByteSize: Int = byteSize.get

  def getGpuPtr: Pointer = gpuPtr.get

  def getOutputArray: Array[T] = outputArray.get

  def allocCPUMem(): Unit

  // Copy data from GPU to CPU
  def gpuToCpu(): Unit




}