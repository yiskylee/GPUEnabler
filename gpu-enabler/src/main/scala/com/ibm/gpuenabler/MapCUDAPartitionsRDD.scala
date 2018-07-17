package com.ibm.gpuenabler

import jcuda.Pointer
import org.apache.spark.{Partition, TaskContext}
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ListBuffer
import scala.reflect.ClassTag




class MapCUDAPartitionsRDD[U: ClassTag, T: ClassTag](val prev: RDD[T],
                                                     val f: T => U,
                                                     kernel: CUDAFunction2,
                                                     constArgs: Seq[AnyVal],
                                                     freeArgs: Seq[Array[_]],
                                                     sizeDepArgs: Seq[Int => Int])
  extends RDD[U](prev) {

  override val partitioner = firstParent[T].partitioner

  override def getPartitions: Array[Partition] = firstParent[T].partitions

  override def compute(split: Partition, context: TaskContext) : Iterator[U] = {
    val parentRDDArray = firstParent[T].iterator(split, context).toArray
    val sampleInput = parentRDDArray(0)
    val sampleOutput = f(sampleInput)
    val numElem = parentRDDArray.length
    var kernelParams = new ListBuffer[Pointer]()

    kernel.params.foreach {
      case param: InputParam =>
        if (InputBufferCache.contains(param.name)) {
          inputBuffer = Some(InputBufferCache.get(param.name).asInstanceOf[InputBufferWrapper[T]])
        } else {
          val buffer = CUDABufferUtils.createInputBufferFor(parentRDDArray)
          buffer.allocGPUMem()
          buffer.copyToGPUMem(param.memType, param.transpose)
          inputBuffer = Some(buffer.asInstanceOf[InputBufferWrapper[T]])
        }
      case param: OutputParam =>
        outputBuffer = OutputBufferCache.getOrElseUpdate(param.name,
          CUDABufferUtils.createOutputBufferFor(sampleOutput, numElem)).
          asInstanceOf[OutputBufferWrapper[U]]
    }

    // Add size, input, output to the kernel parameters
    kernelParams += Pointer.to(Array.fill(1)(numElem))
    inputBuffer match {
      case Some(buffer) =>
        buffer.getGPUPointers match {
          case Some(pointers) =>
            kernelParams ++ pointers
        }
    }




    kernelParams += outputBuffer.getGPUPointer

    // Add free input to the kernel paramters
    val freeParams = kernel.params.collect { case free: FreeParam => free }
    for ((freeParam, freeArg) <- freeParams.zip(freeArgs)) {
      kernelParams += InputBufferCache.getOrElseUpdate(freeParam.name,
        CUDABufferUtils.createInputBufferFor(freeArg(0), freeArg.length)).getGPUPointer
    }

    // Add constant values to the kernel parameters
    val constParams = kernel.params.collect { case c: ConstParam => c }
    for ((constParam, constArg) <- constParams.zip(constArgs)) {
      kernelParams += Pointer.to(Array.fill(1)(constArg.asInstanceOf[Int]))
    }

    // Add arguments whose values depend on the input size to the kernel parameters
    val sizeDepParams = kernel.params.collect { case s: SizeDepParam => s }
    for ((sizeDepParam, sizeDepArg) <- sizeDepParams.zip(sizeDepArgs)) {
      kernelParams += Pointer.to(Array.fill(1)(sizeDepArg(numElem)))
    }

//    kernel.compute[U, T](inputBuffer, outputBuffer)

    new Iterator[U] {
      def next() : U = {
        f(parentRDDArray.iterator.next)
      }

      def hasNext() : Boolean = {
        parentRDDArray.iterator.hasNext
      }
    }
  }
  var inputBuffer: Option[InputBufferWrapper[T]] = None
  var outputBuffer: Option[OutputBufferWrapper[U]] = None
}