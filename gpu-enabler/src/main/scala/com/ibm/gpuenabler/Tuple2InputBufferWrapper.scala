package com.ibm.gpuenabler

import scala.reflect.ClassTag

import jcuda.Pointer

class Tuple2InputBufferWrapper[K: ClassTag, V: ClassTag] (inputArray: Array[Tuple2[K, V]])
  extends InputBufferWrapper[Tuple2[K, V]] {

  val buffer1: InputBufferWrapper[K] = CUDABufferUtils.createInputBufferFor[K](inputArray.map(_._1))
  val buffer2: InputBufferWrapper[V] = CUDABufferUtils.createInputBufferFor[V](inputArray.map(_._2))

  override def getKernelParams: Seq[Pointer] = {
    List(buffer1.gpuPtr.get, buffer2.gpuPtr.get)
  }

  override def allocCPUPinnedMem(): Unit = {
    buffer1.allocCPUPinnedMem()
    buffer2.allocCPUPinnedMem()
  }

  override def allocGPUMem(): Unit = {
    buffer1.allocGPUMem()
    buffer2.allocGPUMem()
  }

  override def cpuToGpu(transpose: Boolean): Unit = {
    buffer1.cpuToGpu(transpose)
    buffer2.cpuToGpu(transpose)
  }
}