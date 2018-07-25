package com.ibm.gpuenabler

import java.nio.ByteOrder

import jcuda.driver.{CUresult, JCudaDriver}
import org.apache.spark.mllib.linalg.DenseVector
import jcuda.{CudaException, Pointer}
import jcuda.runtime.JCuda
import jcuda.runtime.cudaMemcpyKind

// numVecs denotes how many dense vectors are enclosed in this buffer
// vecSize denotes the length of each dense vector
class DenseVectorInputBufferWrapper(inputArray: Array[DenseVector])
  extends InputBufferWrapper[DenseVector] {

  private val _numVectors = inputArray.length
  private val _vecSize = inputArray(0).size
  numElems = Some(_numVectors)
  size = Some(_numVectors * _vecSize * 8)

  override def cpuToGpu(): Unit = {
    val buffer = cpuPtr.get.getByteBuffer(0, size.get).order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer()
    if (transpose) {
      for (i <- 0 until _numVectors) {
        val vec_i = inputArray(i).values
        for (j <- 0 until _vecSize)
          buffer.put(j * _numVectors + i, vec_i(j))
      }
    } else {
      for (i <- 0 until _numVectors) {
        val vec_i = inputArray(i)
        for (j <- 0 until _vecSize)
          buffer.put(i * _numVectors + j, vec_i(j))
      }
    }
    JCudaDriver.cuMemcpyHtoDAsync(devPtr.get, cpuPtr.get, size.get, cuStream)
  }
}