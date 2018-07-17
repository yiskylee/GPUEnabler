package com.ibm.gpuenabler

import java.nio.ByteOrder

import jcuda.driver.CUresult
import org.apache.spark.mllib.linalg.DenseVector
import jcuda.{CudaException, Pointer}
import jcuda.runtime.JCuda

// numVecs denotes how many dense vectors are enclosed in this buffer
// vecSize denotes the length of each dense vector
class DenseVectorInputBufferWrapper(inputArray: Array[DenseVector])
  extends InputBufferWrapper[DenseVector] {

  private val _numVecs = inputArray.length
  private val _vecSize = inputArray(0).size
  val size: Int = _numVecs * _vecSize * 8

  def copyToGPUMem(memType: String, transpose: Boolean): Unit = {
    gpuPtr match {
      case Some(ptr) =>
        memType match {
          case "global" =>
            val buffer = ptr.getByteBuffer(0, size).order(ByteOrder.LITTLE_ENDIAN)
            if (transpose) {
              for (i <- 0 until _numVecs) {
                val vec_i = inputArray(i).asInstanceOf[DenseVector].values
                for (j <- 0 until _vecSize)
                  buffer.asDoubleBuffer().put(j * _numVecs + i, vec_i(j))
              }
            } else {
              for (i <- 0 until _numVecs) {
                val vec_i = inputArray(i)
                for (j <- 0 until _vecSize)
                  buffer.asDoubleBuffer().put(i * _numVecs + j, vec_i(j))
              }
            }
        }
      case None =>
        println("GPU pointer is not allocated")
        System.exit(1)
    }
  }
}
