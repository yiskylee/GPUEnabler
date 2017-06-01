package com.ibm.gpuenabler

import jcuda.driver.{CUevent, CUstream, JCudaDriver}
import scala.collection.mutable.HashMap
import scala.collection.mutable.ListBuffer

object CPUIterTimer {
  // Each element in the list is a timer corresponding to one iteration
  val timerList = new ListBuffer[HashMap[String, ListBuffer[(Double, Double)]]]
  var iterNum = -1

  def startNewIter(): Unit = {
    // This marks the end of an iteration
    iterNum += 1
    timerList += new HashMap[String, ListBuffer[(Double, Double)]]
  }


  def time[R](block: => R, name: String): R = {
    // Use nanoTime() for accurate measurement of duration
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    // use currentTimeMills to get the correct time stamp
    val end = System.currentTimeMillis.toDouble
    val elapsedTime = (t1 - t0) / 1e6
    val start = end - elapsedTime

    val curMap = timerList(iterNum)
    if (curMap.contains(name)) {
      // If the event has already been recorded in this iteration
      curMap(name) += ((start, end))
    } else {
      // If this is the first time the event happens in this iteration
      // We first create a list and then add the time
      curMap(name) = new ListBuffer[(Double, Double)]
      curMap(name) += ((start, end))
    }
    result
  }

}


//class Timer {
//  protected var timesAccum = new HashMap[String, Double]
//  protected var timesPerIter = new HashMap[String, ListBuffer[Double]]
//  protected var timesStartEnd = new HashMap[String, ListBuffer[(Double, Double)]]
//
//  def printIterTime(): Unit = {
//    for ((name, timeList) <- timesPerIter) {
//      print("Timer: " + name)
//      for (time <- timeList) {
//        print(", " + time)
//      }
//      println
//    }
//  }
//
//  def printStartEndTime(): Unit = {
//    for ((name, timeList) <- timesStartEnd) {
//      print("StartEndTimer: " + name)
//      for (startEnd <- timeList) {
//        print(", " + startEnd._1 + ", " + startEnd._2)
//      }
//      println
//    }
//  }
//}
//
//object CPUTimer extends Timer {
//  def printTime[R](block: => R, name: String): R = {
//    val t0 = System.nanoTime()
//    val result = block
//    val t1 = System.nanoTime()
//    val t = (t1 - t0) / 1e6
//    println("Timer: " + name + ": " + f"$t%.5f ms")
//    result
//  }
//
//  // Gather accumulated time and port them to a table of time per iteration
//  // Then reset the accumulated time table for the next iteration
//  def restart(): Unit = {
//    for ((name, time) <- timesAccum) {
//      if (!timesPerIter.contains(name)) {
//        timesPerIter(name) = new ListBuffer[Double]
//      }
//      timesPerIter(name) += time
//      timesAccum(name) = 0.0
//    }
//  }
//
//
//  // Function to record starting and ending time of every event timed by
//  // accumuTimer
//  def recordStartAndEnd(startNano: Long, endNano: Long, name: String): Unit = {
//    val end = System.currentTimeMillis.toDouble
//    val elapsedTime = (endNano - startNano) / 1e6
//    val start = end - elapsedTime
//    if (!timesStartEnd.contains(name)) {
//      timesStartEnd(name) = new ListBuffer[(Double, Double)]
//    }
//    timesStartEnd(name) += ((start, end))
//  }
//
//
//  def accumuTime[R](block: => R, name: String): R = {
////    if (name == "putIteratorAsValues") {
////      println("Receive putIteratorAsValues, timer so far: ")
////      for ((name, _) <- timesStartEnd) {
////        print(name + ", ")
////      }
////    }
//    val t0 = System.nanoTime()
//    val result = block
//    val t1 = System.nanoTime()
//    recordStartAndEnd(t0, t1, name)
//    val t = (t1 - t0) / 1e6
//    if (timesAccum.contains(name)) {
//      timesAccum(name) += t
//    } else {
//      timesAccum(name) = t
//    }
//    result
//  }
//}
//
//object GPUTimer extends Timer {
//
//  class GPUTimerCls(val stream: CUstream, val eventType: String) {
//    private var eventStart = new CUevent
//    private var eventStop = new CUevent
//    private var synced: Boolean = false
//    private var elapsedTime: Array[Float] = new Array[Float](1)
//
//    def start(): Unit = {
//      JCudaDriver.cuEventCreate(eventStart, 0)
//      JCudaDriver.cuEventCreate(eventStop, 0)
//      JCudaDriver.cuEventRecord(eventStart, stream)
//    }
//
//    def stop(): Unit = {
//      JCudaDriver.cuEventRecord(eventStop, stream)
//    }
//
//    def sync(): Unit = {
//      JCudaDriver.cuEventSynchronize(eventStop)
//      synced = true
//    }
//
//    def getTime: Float = {
//      if (synced)
//        elapsedTime(0)
//      else {
//        sync()
//        JCudaDriver.cuEventElapsedTime(elapsedTime, eventStart, eventStop)
//        elapsedTime(0)
//      }
//    }
//
//    def shutDown(): Unit = {
//      JCudaDriver.cuEventDestroy(eventStart)
//      JCudaDriver.cuEventDestroy(eventStop)
//    }
//  }
//
//  private var timers = new ListBuffer[GPUTimerCls]()
//
//  def time[R](block: =>R, stream: CUstream, eventType: String): R = {
//    val timer = new GPUTimerCls(stream, eventType)
//    timer.start()
//    val result = block
//    timer.stop()
//    timers += timer
//    result
//  }
//  def getTimers:ListBuffer[GPUTimerCls] = timers
//
//  def restart(): Unit = {
//    // First we sync all the cuda streams and update the accumulated timer
//    // for just this one iteration
//    for (timer <- timers) {
//      val name = timer.eventType
//      val t = timer.getTime
//      if (timesAccum.contains(name)) {
//        timesAccum(name) += t
//      } else {
//        timesAccum(name) = t
//      }
//    }
//    // Then we use the accumulated time to update the iteration timer
//    for ((name, time) <- timesAccum) {
//      if (!timesPerIter.contains(name))
//        timesPerIter(name) = new ListBuffer[Double]
//      timesPerIter(name) += time
//      timesAccum(name) = 0.0
//    }
//    // Then we destroy the cuda event and remove all the times
//    for (timer <- timers) {
//      timer.shutDown()
//      timers -= timer
//    }
//  }
//
//  def sum: Float = {
//    var totalTime: Float = 0
//    timers.foreach (t => totalTime = totalTime + t.getTime)
//    totalTime
//  }
//
//  def printStatsDetail(): Unit = {
//    for (timer <- timers) {
//      val time = timer.getTime
//      println(timer.stream + "->" + timer.eventType +
//        ": " + f"$time%.5f ms")
//    }
//    printStats()
//  }
//
//  def printStats(): Unit = {
//    val totalTime = sum
//    println("Total GPU Time: " + f"$totalTime%.5f ms")
//  }
//}
//
//// scalastyle:on
