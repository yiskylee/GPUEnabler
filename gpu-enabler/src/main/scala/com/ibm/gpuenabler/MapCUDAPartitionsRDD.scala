package com.ibm.gpuenabler

import jcuda.Pointer
import org.apache.spark.gpuenabler.CUDAUtils
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
  extends RDD[U](prev) with CUDAUtils._Logging {

  override val partitioner = firstParent[T].partitioner

  override def getPartitions: Array[Partition] = firstParent[T].partitions

  override def compute(split: Partition, context: TaskContext) : Iterator[U] = {
    val parentRDDArray = firstParent[T].iterator(split, context).toArray
    val sampleInput = parentRDDArray(0)
    val sampleOutput = f(sampleInput)
    val numElems = parentRDDArray.length

    kernel.params.foreach {
      case param: InputParam =>
        val key: String = param.name + '_' + split.index
        if (InputBufferCache.contains(key)) {
          logInfo(s"Input buffer $key found in the cache")
          inputBuffer = Some(InputBufferCache.get(key).get
            .asInstanceOf[InputBufferWrapper[T]])
        } else {
          logInfo(s"Input buffer $key not found in the cache, create a new one: ")
          val buffer = CUDABufferUtils.createInputBufferFor(parentRDDArray)
          buffer.setTranspose(param.transpose)
          buffer.allocCPUPinnedMem()
          buffer.allocGPUMem()
          buffer.cpuToGpu()
          InputBufferCache.update(key, buffer)
          inputBuffer = Some(buffer)
        }
      case param: OutputParam =>
        val key: String = param.name + '_' + split.index
        val buffer = CUDABufferUtils.createOutputBufferFor(sampleOutput, numElems)
        if (OutputBufferCache.contains(key)) {
          logInfo(s"Output buffer $key found in the cache")
          val existingBuffer = OutputBufferCache.get(key).get
          if (existingBuffer.getByteSize == buffer.getByteSize) {
            // When the buffer is found in the cache, we need to reset the array backing it
            // essentially invalidate the stale copy from last time when the buffer is
            // inserted to the cache, we still keep the GPU Pointer and GPU allocation though
            existingBuffer.reset()
            existingBuffer.setStream(inputBuffer.get.getCuStream)
            outputBuffer = Some(existingBuffer.asInstanceOf[OutputBufferWrapper[U]])
          } else {
            // The output buffer used in the last iteration has a different size as the incoming one
            // Might be caused by different partitions
            System.err.println(s"Existing output buffer's size is ${existingBuffer.getByteSize}" +
              s"while the requested size is ${buffer.getByteSize}, " +
              s"free the old buffer and create a new one")
            existingBuffer.freeGPUMem()
            existingBuffer.freeCPUMem()
            // TODO: At the moment, I am manually doing allocGPUMem, allocCPUMem for all new
            // TODO: buffers, there might be a way to make them all lazily evaluated
            // TODO: e.g., when I call compute on the input and output buffer, they realize that
            // TODO: data is not ready, then they would populate the buffers by itself.
            buffer.setTranspose(param.transpose)
            buffer.setStream(inputBuffer.get.getCuStream)
            buffer.allocGPUMem()
            buffer.allocCPUMem()
            OutputBufferCache.update(key, buffer)
            outputBuffer = Some(buffer)
          }
        } else {
          logInfo(s"Output buffer $key not found in the cache, create a new one")
          buffer.setTranspose(param.transpose)
          buffer.setStream(inputBuffer.get.getCuStream)
          buffer.allocGPUMem()
          buffer.allocCPUMem()
          OutputBufferCache.update(key, buffer)
          outputBuffer = Some(buffer)
        }
//        outputBuffer.get.setStream(inputBuffer.get.getCuStream)
    }

    // Add size, input, output to the kernel parameters
    var kernelParams = Seq(Pointer.to(Array.fill(1)(numElems)))
    kernelParams ++= inputBuffer.get.getKernelParams
    kernelParams ++= outputBuffer.get.getKernelParams

//    // Add free input to the kernel paramters
//    val freeParams = kernel.params.collect { case free: FreeParam => free }
//    for ((freeParam, freeArg) <- freeParams.zip(freeArgs)) {
//      kernelParams += InputBufferCache.getOrElseUpdate(freeParam.name,
//        CUDABufferUtils.createInputBufferFor(freeArg(0), freeArg.length)).getGPUPointer
//    }

    constArgs.foreach {
      case arg: Int => kernelParams :+= Pointer.to(Array.fill(1)(arg.asInstanceOf[Int]))
      case arg: Float => kernelParams :+= Pointer.to(Array.fill(1)(arg.asInstanceOf[Float]))
      case arg: Double => kernelParams :+= Pointer.to(Array.fill(1)(arg.asInstanceOf[Double]))
    }

//    // Add arguments whose values depend on the input size to the kernel parameters
//    val sizeDepParams = kernel.params.collect { case s: SizeDepParam => s }
//    for ((sizeDepParam, sizeDepArg) <- sizeDepParams.zip(sizeDepArgs)) {
//      kernelParams += Pointer.to(Array.fill(1)(sizeDepArg(numElem)))
//    }
    kernel.compute[U, T](inputBuffer.get, outputBuffer.get, kernelParams,
                         constArgs, freeArgs, sizeDepArgs)
    outputBuffer.get
  }
  var inputBuffer: Option[InputBufferWrapper[T]] = None
  var outputBuffer: Option[OutputBufferWrapper[U]] = None
}