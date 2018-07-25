package com.ibm.gpuenabler

import java.nio.ByteOrder

import jcuda.driver.JCudaDriver
import jcuda.runtime.{JCuda, cudaMemcpyKind}

class DoubleArrayInputBufferWrapper(inputArray: Array[Array[Double]])
  extends InputBufferWrapper[Array[Double]] {

  private val _numArrays = inputArray.length
  private val _arraySize = inputArray(0).length
  numElems = Some(_numArrays)
  size = Some(_numArrays * _arraySize * 8)

  override def cpuToGpu(): Unit = {
    val buffer = cpuPtr.get.getByteBuffer(0, size.get).order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer()
    if (transpose) {
      for (i <- 0 until _numArrays) {
        val array_i = inputArray(i)
        for (j <- 0 until _arraySize)
          buffer.put(j * _numArrays + i, array_i(j))
      }
    } else {
      var offset: Int = 0
      for (i <- 0 until _numArrays) {
        val array_i = inputArray(i)
        buffer.put(array_i, offset, _arraySize)
        offset += _arraySize * 8
      }
    }
    JCudaDriver.cuMemcpyHtoDAsync(devPtr.get, cpuPtr.get, size.get, cuStream)
  }
}