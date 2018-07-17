package com.ibm.gpuenabler

import org.apache.spark.mllib.linalg.DenseVector

import scala.reflect.ClassTag

object CUDABufferUtils {
  def createInputBufferFor[T: ClassTag](inputArray: Array[T]): InputBufferWrapper[T] = {
    val sampleInput = inputArray(0)
    val numElem = inputArray.length
    sampleInput match {
      case _: DenseVector =>
        new DenseVectorInputBufferWrapper(inputArray.asInstanceOf[Array[DenseVector]]).
          asInstanceOf[InputBufferWrapper[T]]
      case _: Tuple2[_, _] =>
        new Tuple2InputBufferWrapper(inputArray.asInstanceOf[Array[Tuple2[_, _]]])
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
