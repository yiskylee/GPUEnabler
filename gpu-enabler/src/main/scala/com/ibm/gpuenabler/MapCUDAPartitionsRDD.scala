package com.ibm.gpuenabler

import org.apache.spark.{Partition, TaskContext}
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

class MapCUDAPartitionsRDD[U: ClassTag, T: ClassTag](val prev: RDD[T], val f: T => U)
  extends RDD[U](prev) {

  override val partitioner = firstParent[T].partitioner

  override def getPartitions: Array[Partition] = firstParent[T].partitions

  override def compute(split: Partition, context: TaskContext) : Iterator[U] = {
    val parentRDDArray = firstParent[T].iterator(split, context).toArray
    val sampleOutput = f(parentRDDArray(0))
    val numElem = parentRDDArray.length

    inputBuffer = CUDABufferUtils.createInputBuffer(parentRDDArray)
    outputBuffer = CUDABufferUtils.createOutputBufferFor[U](sampleOutput, numElem)

    new Iterator[U] {
      def next() : U = {
        f(parentRDDArray.iterator.next)
      }

      def hasNext() : Boolean = {
        parentRDDArray.iterator.hasNext
      }
    }
  }
  var inputBuffer: InputBufferWrapper[T] = _
  var outputBuffer: OutputBufferWrapper[U] = _
}

