package com.ibm.gpuenabler

import java.nio.ByteOrder

import jcuda.{CudaException, Pointer}
import jcuda.driver.CUresult
import jcuda.runtime.{JCuda, cudaMemcpyKind, cudaStream_t}
import org.apache.spark.mllib.linalg.{DenseVector, Vectors}


class DenseVectorOutputBufferWrapper(numVectors: Int, vecSize: Int)
  extends OutputBufferWrapper[DenseVector] {

  size = Some(numVectors * vecSize * 8)
  var rawArray: Array[Double] = new Array[Double](size.get)
  cpuPtr = Some(Pointer.to(rawArray))

  override def gpuToCpu(stream: cudaStream_t): Unit = {
    JCuda.cudaMemcpyAsync(cpuPtr.get, gpuPtr.get, size.get,
      cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    val output = new Array[DenseVector](numVectors)
    var index = 0
    val rawArrayIter = rawArray.grouped(vecSize)
    for (a <- rawArrayIter) {
      output(index) = new DenseVector(a)
      index += 1
    }
    outputArray = Some(output)
  }
}