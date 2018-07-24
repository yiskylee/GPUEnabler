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
  protected var size: Option[Int] = None
  protected var numElems: Option[Int] = None
  protected val stream: cudaStream_t = {
    val stream = new cudaStream_t
    JCuda.cudaStreamCreateWithFlags(stream, JCuda.cudaStreamNonBlocking)
    stream
  }

  protected val cuStream: CUstream = new CUstream(stream)

  def getStream: cudaStream_t = stream

  def getCuStream: CUstream = cuStream

  def getKernelParams: Seq[Pointer] = Seq(gpuPtr.get)

  def allocCPUPinnedMem(): Unit = {
    cpuPtr = CUDABufferUtils.allocCPUPinnedMem(size.get)
    System.err.println(s"Input Buffer Alloc CPU Pinned Mem: ${size.get}")
  }

  def allocGPUMem(): Unit = {
    devPtr = CUDABufferUtils.allocGPUMem(size.get)
    gpuPtr = Some(Pointer.to(devPtr.get))
    System.err.println(s"Input Buffer Alloc GPU Pinned Mem: ${size.get}")
  }

  def getSize: Int = size.get

  def getNumElems: Int = numElems.get

  def getGpuPtr: Pointer = gpuPtr.get

  // Copy data from CPU to GPU
  def cpuToGpu(transpose: Boolean): Unit
}