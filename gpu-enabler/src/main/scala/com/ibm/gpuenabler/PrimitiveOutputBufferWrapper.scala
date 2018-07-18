package com.ibm.gpuenabler
import jcuda.Pointer
import jcuda.runtime.{JCuda, cudaMemcpyKind, cudaStream_t}

import scala.reflect.ClassTag

class PrimitiveOutputBufferWrapper[T: ClassTag](sample: T, numElems: Int, elemSize: Int)
  extends OutputBufferWrapper[T] {
  override var size: Option[Int] = Some(numElems * elemSize)
  override def gpuToCpu(stream: cudaStream_t): Unit = {
    sample match {
      case _: Int =>
        var rawArray = new Array[Int](size.get)
        cpuPtr = Some(Pointer.to(rawArray))
        JCuda.cudaMemcpyAsync(cpuPtr.get, gpuPtr.get, size.get,
          cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        outputArray = Some(rawArray.asInstanceOf[Array[T]])
      case _: Float =>
        var rawArray = new Array[Float](size.get)
        cpuPtr = Some(Pointer.to(rawArray))
        JCuda.cudaMemcpyAsync(cpuPtr.get, gpuPtr.get, size.get,
          cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        outputArray = Some(rawArray.asInstanceOf[Array[T]])
      case _: Double =>
        var rawArray = new Array[Double](size.get)
        cpuPtr = Some(Pointer.to(rawArray))
        JCuda.cudaMemcpyAsync(cpuPtr.get, gpuPtr.get, size.get,
          cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        outputArray = Some(rawArray.asInstanceOf[Array[T]])
      case _ =>
        System.err("Does not support this primitive type")
        System.exit(1)
    }
  }
}