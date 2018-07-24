package com.ibm.gpuenabler

import java.nio.ByteOrder
import jcuda.runtime.{JCuda, cudaMemcpyKind}
import scala.reflect.ClassTag

class PrimitiveInputBufferWrapper[T: ClassTag]
  extends InputBufferWrapper[T] {
  private var _inputArray: Option[Array[T]] = None
  private var _elemSize: Option[Int] = None
  override def cpuToGpu(transpose: Boolean): Unit = {
    val buffer = cpuPtr.get.getByteBuffer(0, size.get).order(ByteOrder.LITTLE_ENDIAN)
    _inputArray.get(0) match {
      case _: Int =>
        buffer.asIntBuffer().put(_inputArray.asInstanceOf[Array[Int]])
        _elemSize = Some(4)
      case _: Float =>
        buffer.asFloatBuffer().put(_inputArray.asInstanceOf[Array[Float]])
        _elemSize = Some(4)
      case _: Double =>
        buffer.asDoubleBuffer().put(_inputArray.asInstanceOf[Array[Double]])
        _elemSize = Some(8)
    }
    JCuda.cudaMemcpyAsync(gpuPtr.get, cpuPtr.get, size.get,
      cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
  }
}
