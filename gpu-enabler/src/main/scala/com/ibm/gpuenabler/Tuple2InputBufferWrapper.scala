package com.ibm.gpuenabler

import scala.reflect.ClassTag

import jcuda.Pointer

class Tuple2InputBufferWrapper[K: ClassTag, V: ClassTag](
  inputArray: Array[Tuple2[K, V]],
  val cache: Boolean,
  val transpose: Boolean)
    extends InputBufferWrapper[Tuple2[K, V]] {

  val buffer1: InputBufferWrapper[K] =
    CUDABufferUtils.createInputBufferFor[K](
      inputArray.map(_._1), cache, transpose)
  val buffer2: InputBufferWrapper[V] =
    CUDABufferUtils.createInputBufferFor[V](
      inputArray.map(_._2), cache, transpose)

  override def getKernelParams: Seq[Pointer] = {
    Seq(buffer1.getGpuPtr, buffer2.getGpuPtr)
  }

  override def allocCPUPinnedMem(): Unit = {
    buffer1.allocCPUPinnedMem()
    buffer2.allocCPUPinnedMem()
  }

  override def allocGPUMem(): Unit = {
    buffer1.allocGPUMem()
    buffer2.allocGPUMem()
  }

  override def cpuToGpu(): Unit = {
    buffer1.cpuToGpu()
    buffer2.cpuToGpu()
  }
}