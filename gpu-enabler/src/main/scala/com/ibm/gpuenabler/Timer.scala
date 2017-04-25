package com.ibm.gpuenabler

import jcuda.driver.{CUevent, CUstream, JCudaDriver}

import scala.collection.mutable


object GPUTimers {
  private var timers = new mutable.ListBuffer[GPUTimer]()
  def time[R](block: =>R, stream: CUstream, eventType: String): R = {
    val timer = new GPUTimer(stream, eventType)
    timer.start
    val result = block
    timer.stop
    addTimer(timer)
    result
  }

  def getTimers = timers
  def addTimer(timer: GPUTimer): Unit = {
    timers += timer
  }
  def clear: Unit = {
    for (timer <- timers) {
      timer.shutDown
      timers -= timer
    }
  }
  def sum: Float = {
    var totalTime: Float = 0
    timers.foreach (t => totalTime = totalTime + t.getTime)
    totalTime
  }
  def printStats: Unit = {
    for (timer <- timers) {
      val time = timer.getTime / 1e3
      println(timer.stream + "->" + timer.eventType +
        ": " + f"$time%.5f seconds")
    }
    val totalTime = sum / 1e3
    println("Total GPU Time: " + f"$totalTime%.5f seconds")
  }
}

object CPUTimer {
  def time[R](block: => R, name: String): R = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    val t = (t1 - t0) / 1e9
    println(name + ": " + f"$t%.5f seconds")
    result
  }
}


class GPUTimer(val stream: CUstream, val eventType: String) {
  private var eventStart = new CUevent
  private var eventStop = new CUevent
  private var synced: Boolean = false
  private var elapsedTime: Array[Float] = new Array[Float](1)

  def start: Unit = {
    JCudaDriver.cuEventCreate(eventStart, 0)
    JCudaDriver.cuEventCreate(eventStop, 0)
    JCudaDriver.cuEventRecord(eventStart, stream)
  }

  def stop: Unit = {
    JCudaDriver.cuEventRecord(eventStop, stream)
  }

  def sync: Unit = {
    JCudaDriver.cuEventSynchronize(eventStop)
    synced = true
  }

  def getTime: Float = {
    if (synced)
      elapsedTime(0)
    else {
      sync
      JCudaDriver.cuEventElapsedTime(elapsedTime, eventStart, eventStop)
      elapsedTime(0)
    }
  }

  def shutDown: Unit = {
    JCudaDriver.cuEventDestroy(eventStart)
    JCudaDriver.cuEventDestroy(eventStop)
  }
}
