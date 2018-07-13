package com.ibm.gpuenabler

import org.apache.spark.mllib.linalg.DenseVector

import scala.reflect.ClassTag

object CUDABufferUtils {
  def createInputBuffer[T: ClassTag](cpuArray: Array[T]): InputBufferWrapper[T] = {
    cpuArray(0) match {
      case DenseVector =>
        new DenseVectorInputBufferWrapper(cpuArray.asInstanceOf[Array[DenseVector]]).
          asInstanceOf[InputBufferWrapper[T]]
    }
  }

  def createOutputBufferFor[T: ClassTag](sampleOutput: T, numElem: Int)
  : OutputBufferWrapper[T] = {
    sampleOutput match {
      case DenseVector =>
        new DenseVectorOutputBufferWrapper(numElem,
          sampleOutput.asInstanceOf[DenseVector].size).asInstanceOf[OutputBufferWrapper[T]]
      case Tuple2[_, _] =>
        new Tuple2OutputBufferWrapper(numElem).asInstanceOf[OutputBufferWrapper[T]]
    }
//    val className: String = sampleOutput.getClass.getName
//    if (className.equals("org.apache.spark.mllib.linalg.DenseVector")) {
  }
}
