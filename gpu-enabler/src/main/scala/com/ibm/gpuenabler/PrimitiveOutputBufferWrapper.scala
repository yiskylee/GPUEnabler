package com.ibm.gpuenabler
import jcuda.Pointer
import jcuda.driver.{CUstream, JCudaDriver}
import jcuda.runtime.{JCuda, cudaMemcpyKind, cudaStream_t}

import scala.reflect.ClassTag

class PrimitiveOutputBufferWrapper[T: ClassTag](sample: T, numElems: Int, elemSize: Int)
  extends OutputBufferWrapper[T] {
  byteSize = Some(numElems * elemSize)
  override def gpuToCpu(stream: CUstream): Unit = {
    sample match {
      case _: Int =>
        var rawArray = new Array[Int](byteSize.get)
        cpuPtr = Some(Pointer.to(rawArray))
        JCudaDriver.cuMemcpyDtoHAsync(cpuPtr.get, devPtr.get, byteSize.get, stream)
        outputArray = Some(rawArray.asInstanceOf[Array[T]])
      case _: Float =>
        var rawArray = new Array[Float](byteSize.get)
        cpuPtr = Some(Pointer.to(rawArray))
        JCudaDriver.cuMemcpyDtoHAsync(cpuPtr.get, devPtr.get, byteSize.get, stream)
        outputArray = Some(rawArray.asInstanceOf[Array[T]])
      case _: Double =>
        var rawArray = new Array[Double](byteSize.get)
        cpuPtr = Some(Pointer.to(rawArray))
        JCudaDriver.cuMemcpyDtoHAsync(cpuPtr.get, devPtr.get, byteSize.get, stream)
        outputArray = Some(rawArray.asInstanceOf[Array[T]])
      case _ =>
        System.err.println("Does not support this primitive type")
        System.exit(1)
    }
    // Reset initial index for Iterator
    idx = 0
  }
}