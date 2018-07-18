package com.ibm.gpuenabler

import jcuda.{CudaException, Pointer}
import jcuda.driver.CUresult
import jcuda.runtime.{JCuda, cudaStream_t}

import scala.reflect.ClassTag

class Tuple2OutputBufferWrapper[K: ClassTag, V: ClassTag](sample: Tuple2[K, V], numTuples: Int)
  extends OutputBufferWrapper[Tuple2[K, V]] {

  val buffer1: OutputBufferWrapper[K] = CUDABufferUtils.createOutputBufferFor[K](sample._1, numTuples)
  val buffer2: OutputBufferWrapper[V] = CUDABufferUtils.createOutputBufferFor[V](sample._2, numTuples)

  override var outputArray: Option[Array[Tuple2[K, V]]] = None

  override def getKernelParams: Seq[Pointer] = {
    List(buffer1.gpuPtr.get, buffer2.gpuPtr.get)
  }

  override def allocGPUMem(): Unit = {
    buffer1.allocGPUMem()
    buffer2.allocGPUMem()
  }

  override def gpuToCpu(stream: cudaStream_t): Unit = {
    buffer1.gpuToCpu(stream)
    buffer2.gpuToCpu(stream)
    val zipped = buffer1.outputArray.get zip buffer2.outputArray.get
    outputArray = Some(zipped)
  }
}