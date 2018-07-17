package com.ibm.gpuenabler

import jcuda.Pointer

trait OutputBufferWrapper[T] {
  // Return the GPU Pointer of this buffer
  def getGPUPointer: Pointer
//
//  // Allocate GPU Memory for this buffer and return its pointer on the GPU
//  def allocPinnedMemory: Pointer

//  def next(): T
//
//  def hasNext(): Boolean

}