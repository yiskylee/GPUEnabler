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
        val key: String = param.name + '_' + split.index
        if (InputBufferCache.contains(key)) {
          inputBuffer = Some(InputBufferCache.get(key).get
            .asInstanceOf[InputBufferWrapper[T]])
        } else {
          val buffer = CUDABufferUtils.createInputBufferFor(parentRDDArray)
          buffer.allocCPUPinnedMem()
          buffer.allocGPUMem()
          buffer.cpuToGpu(param.transpose)
          InputBufferCache.update(key, buffer)
          inputBuffer = Some(buffer)
        }
      case param: OutputParam =>
        val key: String = param.name + '_' + split.index
        val buffer = CUDABufferUtils.createOutputBufferFor(sampleOutput, numElem)
        if (OutputBufferCache.contains(key)) {
          val existingBuffer = OutputBufferCache.get(key).get
          if (existingBuffer.getSize == buffer.getSize) {
            outputBuffer = Some(existingBuffer.asInstanceOf[OutputBufferWrapper[U]])
          } else {
            // The output buffer used in the last iteration has a different size as the incoming one
            // Might be caused by different partitions
            System.err(s"Existing output buffer's size is ${existingBuffer.getSize} " +
              s"while the requested size is ${buffer.getSize}")
            existingBuffer.freeGPUMem()
            existingBuffer.freeCPUMem()
            buffer.allocGPUMem()
            OutputBufferCache.update(key, buffer)
            outputBuffer = Some(buffer)
          }
        } else {
          buffer.allocGPUMem()
          OutputBufferCache.update(key, buffer)
          outputBuffer = Some(buffer)
        }
    }

    // Add size, input, output to the kernel parameters
    kernelParams += Pointer.to(Array.fill(1)(numElem))
    kernelParams ++= inputBuffer.get.getKernelParams
    kernelParams ++= outputBuffer.get.getKernelParams

//    // Add free input to the kernel paramters
//    val freeParams = kernel.params.collect { case free: FreeParam => free }
//    for ((freeParam, freeArg) <- freeParams.zip(freeArgs)) {
//      kernelParams += InputBufferCache.getOrElseUpdate(freeParam.name,
//        CUDABufferUtils.createInputBufferFor(freeArg(0), freeArg.length)).getGPUPointer
//    }
//
//    // Add constant values to the kernel parameters
//    val constParams = kernel.params.collect { case c: ConstParam => c }
//    for ((constParam, constArg) <- constParams.zip(constArgs)) {
//      kernelParams += Pointer.to(Array.fill(1)(constArg.asInstanceOf[Int]))
//    }
//
//    // Add arguments whose values depend on the input size to the kernel parameters
//    val sizeDepParams = kernel.params.collect { case s: SizeDepParam => s }
//    for ((sizeDepParam, sizeDepArg) <- sizeDepParams.zip(sizeDepArgs)) {
//      kernelParams += Pointer.to(Array.fill(1)(sizeDepArg(numElem)))
//    }

    kernel.compute[U, T](inputBuffer.get, outputBuffer.get, kernelParams,
                         constArgs, freeArgs, sizeDepArgs)

    outputBuffer.get.gpuToCpu(inputBuffer.get.getStream)

    new Iterator[U] {
      def next : U = outputBuffer.get.next
      def hasNext : Boolean = outputBuffer.get.hasNext
    }
  }
  var inputBuffer: Option[InputBufferWrapper[T]] = None
  var outputBuffer: Option[OutputBufferWrapper[U]] = None
}