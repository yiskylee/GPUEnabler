package com.ibm.gpuenabler

import jcuda.{CudaException, Pointer}
import jcuda.driver.CUresult
import jcuda.runtime.JCuda

import scala.reflect.ClassTag

class Tuple2OutputBufferWrapper[K: ClassTag, V: ClassTag](sample: Tuple2[K, V], numTuples: Int)
  extends OutputBufferWrapper[Tuple2[K, V]] {

//  val outputBuffer0: OutputBufferWrapper[K] = CUDABufferUtils.createOutputBufferFor[K]()

  private var gpuPtr: Option[Pointer] = None
//  private val size: Int = numVectors * elemPerVector * 8
//  override def getGPUPointer: Pointer = {
//    gpuPtr.getOrElse(allocPinnedMemory)
//  }
//  override def allocPinnedMemory: Pointer = {
//    val ptr: Pointer = new Pointer()
//    try {
//      val result: Int = JCuda.cudaHostAlloc(ptr, size, JCuda.cudaHostAllocPortable)
//      if (result != CUresult.CUDA_SUCCESS) {
//        throw new CudaException(JCuda.cudaGetErrorString(result))
//      }
//    }
//    catch {
//      case ex: Exception =>
//        throw new OutOfMemoryError("Could not alloc pinned memory: " + ex.getMessage)
//    }
//    ptr
//  }
}