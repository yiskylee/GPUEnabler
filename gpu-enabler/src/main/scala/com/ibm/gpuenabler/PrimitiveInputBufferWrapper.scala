package com.ibm.gpuenabler
import java.nio.ByteOrder

import jcuda.runtime.{JCuda, cudaMemcpyKind}

import scala.reflect.ClassTag

class PrimitiveInputBufferWrapper[T: ClassTag](inputArray: Array[T], elemSize: Int)
  extends InputBufferWrapper[T] {
  private val _numElems = inputArray.length
  override var size: Option[Int] = Some(_numElems * elemSize)

  override def cpuToGpu(transpose: Boolean): Unit = {

    val buffer = cpuPtr.get.getByteBuffer(0, size.get).order(ByteOrder.LITTLE_ENDIAN)
    inputArray(0) match {
      case _: Int => buffer.asIntBuffer().put(inputArray.asInstanceOf[Array[Int]])
      case _: Float => buffer.asFloatBuffer().put(inputArray.asInstanceOf[Array[Float]])
      case _: Double => buffer.asDoubleBuffer().put(inputArray.asInstanceOf[Array[Double]])
    }
    JCuda.cudaMemcpyAsync(gpuPtr.get, cpuPtr.get, size.get,
      cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
  }
}