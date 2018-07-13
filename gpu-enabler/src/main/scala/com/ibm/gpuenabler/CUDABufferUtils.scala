package com.ibm.gpuenabler

import org.apache.spark.mllib.linalg.DenseVector

import scala.reflect.ClassTag

object CUDABufferUtils {
  def createInputBufferFor[T: ClassTag](sampleInput: T, numElem: Int): InputBufferWrapper[T] = {
    sampleInput match {
      case _: DenseVector =>
        new DenseVectorInputBufferWrapper(numElem, sampleInput.asInstanceOf[DenseVector].size).
          asInstanceOf[InputBufferWrapper[T]]
      case _: Tuple2[_, _] =>
        new Tuple2InputBufferWrapper(sampleInput.asInstanceOf[(_, _)], numElem).
          asInstanceOf[InputBufferWrapper[T]]

    }
  }

  def createOutputBufferFor[T: ClassTag](sampleOutput: T, numElem: Int)
  : OutputBufferWrapper[T] = {
    sampleOutput match {
      case _: DenseVector =>
        new DenseVectorOutputBufferWrapper(numElem,
          sampleOutput.asInstanceOf[DenseVector].size).asInstanceOf[OutputBufferWrapper[T]]
      case _: Tuple2[_, _] =>
        new Tuple2OutputBufferWrapper(sampleOutput.asInstanceOf[(_, _)], numElem).
          asInstanceOf[OutputBufferWrapper[T]]
    }
//    val className: String = sampleOutput.getClass.getName
//    if (className.equals("org.apache.spark.mllib.linalg.DenseVector")) {
  }
}
