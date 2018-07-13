package com.ibm.gpuenabler
import jcuda.driver.CUresult
import org.apache.spark.mllib.linalg.DenseVector
import jcuda.{CudaException, Pointer}
import jcuda.runtime.JCuda

import scala.reflect.ClassTag

class Tuple2InputBufferWrapper[K: ClassTag, V: ClassTag] (sample: Tuple2[K, V], numTuples: Int)
  extends InputBufferWrapper[Tuple2[K, V]] {

  val buffer1: InputBufferWrapper[K] = CUDABufferUtils.createInputBufferFor[K](sample._1, numTuples)
  val buffer2: InputBufferWrapper[V] = CUDABufferUtils.createInputBufferFor[V](sample._2, numTuples)

}
