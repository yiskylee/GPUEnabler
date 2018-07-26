package com.ibm.gpuenabler
import jcuda.Pointer
import jcuda.driver.{CUstream, JCudaDriver}
import jcuda.driver.JCudaDriver.cuCtxSynchronize
import jcuda.runtime.{JCuda, cudaMemcpyKind, cudaStream_t}
import org.apache.spark.gpuenabler.CUDAUtils

class DoubleArrayOutputBufferWrapper(sample: Array[Double], numArrays: Int)
  extends OutputBufferWrapper[Array[Double]] with CUDAUtils._Logging {
  private val _arraySize = sample.length
  numElems = Some(numArrays)
  byteSize = Some(numArrays * _arraySize * 8)
  var rawArray: Option[Array[Double]] = None

  override def allocCPUMem(): Unit = {
    // Only allocate new buffer when there does not exist one
    rawArray = Some(new Array[Double](numArrays * _arraySize))
    cpuPtr = Some(Pointer.to(rawArray.get))
  }

  override def gpuToCpu(): Unit = {
//    logInfo("output rawArray before gpuToCpu: ")
    logInfo(rawArray.get.mkString(", "))
    JCudaDriver.cuMemcpyDtoHAsync(cpuPtr.get, devPtr.get, byteSize.get, cuStream.get)
//    logInfo("output rawArray after gpuToCpu: ")
//    logInfo(rawArray.get.mkString(", "))
    outputArray =
      if(transpose)
        Some(rawArray.get.grouped(numArrays).toArray.transpose)
      else
        Some(rawArray.get.grouped(_arraySize).toArray)
//    logInfo("outputArray transformed from rawArray")
//    for (array <- outputArray.get)
//      logInfo(array.mkString(", "))
  }
}
