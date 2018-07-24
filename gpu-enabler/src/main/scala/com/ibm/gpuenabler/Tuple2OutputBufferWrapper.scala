package com.ibm.gpuenabler

import jcuda.{CudaException, Pointer}
import jcuda.driver.{CUresult, CUstream}
import jcuda.runtime.{JCuda, cudaStream_t}

import scala.reflect.ClassTag

class Tuple2OutputBufferWrapper[K: ClassTag, V: ClassTag](sample: Tuple2[K, V], numTuples: Int)
  extends OutputBufferWrapper[Tuple2[K, V]] {

  val buffer1: OutputBufferWrapper[K] = CUDABufferUtils.createOutputBufferFor[K](sample._1, numTuples)
  val buffer2: OutputBufferWrapper[V] = CUDABufferUtils.createOutputBufferFor[V](sample._2, numTuples)

  override def getKernelParams: Seq[Pointer] = {
    Seq(buffer1.getGpuPtr, buffer2.getGpuPtr)
  }

  override def allocGPUMem(): Unit = {
    buffer1.allocGPUMem()
    buffer2.allocGPUMem()
  }

  override def gpuToCpu(stream: CUstream, transpose: Boolean): Unit = {
    buffer1.gpuToCpu(stream, transpose)
    buffer2.gpuToCpu(stream, transpose)
    val zipped = buffer1.getOutputArray zip buffer2.getOutputArray
    outputArray = Some(zipped)
  }
}