package com.ibm.gpuenabler
import jcuda.Pointer
import jcuda.driver.{CUstream, JCudaDriver}
import jcuda.runtime.{JCuda, cudaMemcpyKind, cudaStream_t}

import scala.reflect.ClassTag

class PrimitiveOutputBufferWrapper[T: ClassTag](
  sample: T,
  numScalars: Int,
  elemSize: Int,
  val cache: Boolean,
  val transpose: Boolean)
    extends OutputBufferWrapper[T] {

  numElems = Some(numScalars)
  byteSize = Some(numScalars * elemSize)
  var rawArray: Option[Array[_]] = None

  override def allocCPUMem(): Unit = {
    sample match {
      case _: Int =>
        rawArray = Some(new Array[Int](numScalars))
        cpuPtr = Some(Pointer.to(rawArray.get.asInstanceOf[Array[Int]]))
      case _: Float =>
        rawArray = Some(new Array[Float](numScalars))
        cpuPtr = Some(Pointer.to(rawArray.get.asInstanceOf[Array[Float]]))
      case _: Double =>
        rawArray = Some(new Array[Double](numScalars))
        cpuPtr = Some(Pointer.to(rawArray.get.asInstanceOf[Array[Double]]))
      case _ =>
        System.err.println("Does not support this primitive type")
        System.exit(1)
    }
  }

  override def gpuToCpu(): Unit = {
    JCudaDriver.cuMemcpyDtoHAsync(cpuPtr.get, devPtr.get, byteSize.get, cuStream.get)
    outputArray = Some(rawArray.get.asInstanceOf[Array[T]])
  }
}