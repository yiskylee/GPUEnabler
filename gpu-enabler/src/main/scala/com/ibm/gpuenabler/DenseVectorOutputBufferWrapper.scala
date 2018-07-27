package com.ibm.gpuenabler

import java.nio.ByteOrder

import jcuda.{CudaException, Pointer}
import jcuda.driver.{CUresult, CUstream, JCudaDriver}
import jcuda.runtime.{JCuda, cudaMemcpyKind, cudaStream_t}
import org.apache.spark.mllib.linalg.{DenseVector, Vectors}


class DenseVectorOutputBufferWrapper(
  numVectors: Int,
  vecSize: Int,
  val cache: Boolean,
  val transpose: Boolean)
    extends OutputBufferWrapper[DenseVector] {

  numElems = Some(numVectors * vecSize)
  byteSize = Some(numElems.get * 8)
  var rawArray: Option[Array[Double]] = None

  override def allocCPUMem(): Unit = {
    rawArray = Some(new Array[Double](numElems.get))
    cpuPtr = Some(Pointer.to(rawArray.get))
  }

  override def gpuToCpu(): Unit = {
    JCudaDriver.cuMemcpyDtoHAsync(
      cpuPtr.get, devPtr.get, byteSize.get, cuStream.get)
    val arrayOfArrays =
      if (transpose) rawArray.get.grouped(numVectors).toArray.transpose
      else rawArray.get.grouped(vecSize).toArray

    val output = new Array[DenseVector](numVectors)
    for (i <- arrayOfArrays.indices)
        output(i) = new DenseVector(arrayOfArrays(i))
    outputArray = Some(output)
  }
}