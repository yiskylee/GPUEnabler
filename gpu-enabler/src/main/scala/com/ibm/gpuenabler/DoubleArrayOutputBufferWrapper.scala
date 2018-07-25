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
  var rawArray = new Array[Double](numArrays * _arraySize)
  cpuPtr = Some(Pointer.to(rawArray))

  override def gpuToCpu(stream: CUstream): Unit = {
    logInfo("rawArray before gpuToCpu: ")
    println(rawArray.mkString(", "))

    JCudaDriver.cuMemcpyDtoHAsync(cpuPtr.get, devPtr.get, byteSize.get, stream)
    logInfo("rawArray after gpuToCpu: ")
    println(rawArray.mkString(", "))
    outputArray =
      if(transpose)
        Some(rawArray.grouped(numArrays).toArray.transpose)
      else
        Some(rawArray.grouped(_arraySize).toArray)
    // Reset initial index for Iterator
    idx = 0
    logInfo("outputArray transformed from rawArray")
    for (array <- outputArray.get)
      println(array.mkString(", "))
  }
}