package com.ibm.gpuenabler

import java.nio.ByteOrder

import jcuda.{CudaException, Pointer}
import jcuda.driver.{CUresult, CUstream, JCudaDriver}
import jcuda.runtime.{JCuda, cudaMemcpyKind, cudaStream_t}
import org.apache.spark.mllib.linalg.{DenseVector, Vectors}


class DenseVectorOutputBufferWrapper(numVectors: Int, vecSize: Int)
  extends OutputBufferWrapper[DenseVector] {
  numElems = Some(numVectors * vecSize)
  size = Some(numElems.get * 8)
  var rawArray = new Array[Double](numElems.get)
  cpuPtr = Some(Pointer.to(rawArray))

  override def gpuToCpu(stream: CUstream): Unit = {
    JCudaDriver.cuMemcpyDtoHAsync(cpuPtr.get, devPtr.get, size.get, stream)
    val arrayOfArrays =
      if (transpose) rawArray.grouped(numVectors).toArray.transpose
      else rawArray.grouped(vecSize).toArray

    val output = new Array[DenseVector](numVectors)
    for (i <- arrayOfArrays.indices)
        output(i) = new DenseVector(arrayOfArrays(i))
    outputArray = Some(output)
  }
}