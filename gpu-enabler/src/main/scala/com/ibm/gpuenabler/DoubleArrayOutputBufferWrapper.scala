package com.ibm.gpuenabler
import jcuda.Pointer
import jcuda.driver.{CUstream, JCudaDriver}
import jcuda.driver.JCudaDriver.cuCtxSynchronize
import jcuda.runtime.{JCuda, cudaMemcpyKind, cudaStream_t}

class DoubleArrayOutputBufferWrapper(sample: Array[Double], numArrays: Int)
  extends OutputBufferWrapper[Array[Double]] {
  private val _arraySize = sample.length
  numElems = Some(numArrays * _arraySize)
  size = Some(numElems.get * 8)
  var rawArray = new Array[Double](numElems.get)
  cpuPtr = Some(Pointer.to(rawArray))

  override def gpuToCpu(stream: CUstream, transpose: Boolean): Unit = {
    JCudaDriver.cuMemcpyDtoHAsync(cpuPtr.get, devPtr.get, size.get, stream)
    outputArray =
      if(transpose)
        Some(rawArray.grouped(numArrays).toArray.transpose)
      else
        Some(rawArray.grouped(_arraySize).toArray)
  }
}