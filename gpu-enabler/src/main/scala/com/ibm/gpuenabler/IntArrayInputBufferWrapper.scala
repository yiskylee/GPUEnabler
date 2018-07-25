package com.ibm.gpuenabler

import java.nio.{Buffer, ByteOrder}

import jcuda.runtime.{JCuda, cudaMemcpyKind}

class IntArrayInputBufferWrapper(inputArray: Array[Array[Int]])
  extends InputBufferWrapper[Array[Int]] {

  private val _numArrays = inputArray.length
  private val _inputSample = inputArray(0).asInstanceOf[Array[_]]
  private val _arraySize = _inputSample.length
  numElems = Some(_numArrays * _arraySize)
  byteSize = Some(numElems.get * 4)


  override def cpuToGpu(): Unit = {
    val buffer = cpuPtr.get.getByteBuffer(0, byteSize.get).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer()
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
        offset += _arraySize * 4
      }
    }
    JCuda.cudaMemcpyAsync(gpuPtr.get, cpuPtr.get, byteSize.get,
      cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
  }
}