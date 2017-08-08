package com.ibm.gpuenabler

import jcuda.driver.{CUevent, CUstream, JCudaDriver}

import scala.collection.concurrent.TrieMap
import scala.collection.mutable.{HashMap, ListBuffer, SortedSet}
import scala.util.Random

object CPUIterTimer {
  // Each element in the list is a timer corresponding to one iteration,
  // each timer contains the name and all (start, end) pairs
  val timerList = new ListBuffer[TrieMap[String, ListBuffer[(Double, Double)]]]
  // Each element in the list is a timer corresponding to one iteration
  // each timer contains the name and elapsed time
  val accumuTimerList = new ListBuffer[TrieMap[String, Double]]
  val timerNames = SortedSet[String]()
  var iterNum = -1
  var running = false

  // Under which can we merge two durations
  var mergeCondition = 0.05

  def start(): Unit = {
    running = true
  }

  def stop(): Unit = {
    running = false
  }

  def startNewIter(): Unit = {
    // This marks the end of an iteration
    iterNum += 1
    timerList += new TrieMap[String, ListBuffer[(Double, Double)]]
    accumuTimerList += new TrieMap[String, Double]
  }

  def time[R](block: => R, name: String): R = {
    if (running && iterNum > -1) {
      val threadID = Thread.currentThread().getId()
      val newName = name + threadID
      timerNames += newName

      // Only record time if the CPUIterTimer is started by the user
      // Use nanoTime() for accurate measurement of duration
      val t0 = System.nanoTime()
      val result = block
      val t1 = System.nanoTime()
      // use currentTimeMills to get the correct time stamp
      val end = System.currentTimeMillis.toDouble
      val elapsedTime = (t1 - t0) / 1e6
      val start = end - elapsedTime

      val curMap = timerList(iterNum)
      if (curMap.contains(newName)) {
        // If the event has already been recorded in this very iteration
        curMap(newName) += ((start, end))
      } else {
        // If this is the first time the event happens in this iteration
        // We first create a list and then add the time
          curMap(newName) = new ListBuffer[(Double, Double)]
          curMap(newName) += ((start, end))
      }
      result
    } else {
      // When the timer is not running, just run the code blck
      block
    }
  }

  def accumuTime[R](block: => R, name: String): R = {
    val threadID = Thread.currentThread().getId()
    val newName = name + threadID
    if (running && iterNum > -1) {
      val t0 = System.nanoTime()
      val result = block
      val t1 = System.nanoTime()
      // use currentTimeMills to get the correct time stamp
      val end = System.currentTimeMillis.toDouble
      val elapsedTime = (t1 - t0) / 1e6
      val start = end - elapsedTime

      val curMap = timerList(iterNum)
      if (curMap.contains(newName)) {
        // If the event has already been recorded in this very iteration
        // just accumulate the end time
        val newEnd = curMap(newName)(0)._2 + elapsedTime
        val newStartEnd = curMap(newName)(0).copy(_2=newEnd)
        // Replace the old start end with the new start end
        curMap(newName)(0) = newStartEnd
      } else {
        curMap(newName) = new ListBuffer[(Double, Double)]
        curMap(newName) += ((start, end))
      }
      result
    } else {
      block
    }
  }


  def mergeStartEndList(startEndList: ListBuffer[(Double, Double)]) = {
    val mergedList = new ListBuffer[(Double, Double)]
    // Initialize the start and end time of the last event
    // if n+1 th event's start time is no more than 1 ms later
    // than the last event's end time, we consider
    // the two events are very close, thus just merge them and
    // only record the start end for one event
    var lastStart = 0.0
    var lastEnd = 0.0
    for ((startEnd, i) <- startEndList.zipWithIndex) {

      var start = startEnd._1
      val end = startEnd._2

      if (i == 0) {
        lastStart = start
        lastEnd = end
      } else {
        if ((start - lastEnd) / (end - start) <= mergeCondition) {
          // If the current duration can be merged to the last one
          start = lastStart
        } else {
          // If the current duration cannot be merged
          // then we need to add previous duration to the list
          mergedList += ((lastStart, lastEnd))
        }
      }
      // No matter if we can merge or not, always update lastStart and
      // lastEnd with the current start and end
      lastStart = start
      lastEnd = end
    }
    // Make sure to include the last duration
    mergedList += ((lastStart, lastEnd))
  }

  def takeSample[T](a: ListBuffer[T], n: Int, seed: Long) = {
    val rnd = new Random(seed)
    ListBuffer.fill(n)(a(rnd.nextInt(a.length)))
  }

  def printIterTime(): Unit = {
    if (!timerList(0).isEmpty) {
      print("StartEndTimer: ")
      for ((name, index) <- timerNames.zipWithIndex) {
        if (index == timerNames.size - 1)
          print(name + "\n")
        else
          print(name + ",")
      }
      for (timer <- timerList) {
        print("StartEndTimer: ")
        for ((name, i) <- timerNames.zipWithIndex) {
          val startEndList =
            if (timer.contains(name)) timer(name)
            else {
              val dummyStartEndList = new ListBuffer[(Double, Double)]
              dummyStartEndList += ((0, 0))
              dummyStartEndList
            }
          for ((startEnd, j) <- startEndList.zipWithIndex) {
            val start = startEnd._1
            val end = startEnd._2
            if (j == startEndList.length - 1 && i != timerNames.size - 1)
              print(start + "|" + end + ",")
            else if (j == startEndList.length - 1 && i == timerNames.size - 1) {
              print(start + "|" + end + "\n")
            } else {
              print(start + "|" + end + "|")
            }
          }
        }
      }
    } else {
      println("There is no timer")
    }
  }

  def printIterAccumuTime(): Unit = {
    if (!accumuTimerList(0).isEmpty) {
      print("AccumuTimer: ")
      val sortedNames = accumuTimerList(0).toSeq.sortBy(_._1).map(_._1)
      for ((name, index) <- sortedNames.zipWithIndex) {
        if (index == sortedNames.length - 1)
          print(name + "\n")
        else
          print(name + ",")
      }
      for (timer <- accumuTimerList) {
        print("AccumuTimer: ")
        for ((name, i) <- sortedNames.zipWithIndex) {
          if (i != sortedNames.length - 1)
            print(timer(name) + ",")
          else
            print(timer(name) + "\n")
        }
      }
    } else {
      println("There is no accumuTimer")
    }
  }
}


class Timer {
  protected var timesAccum = new HashMap[String, Double]
  protected var timesPerIter = new HashMap[String, ListBuffer[Double]]
  protected var timesStartEnd = new HashMap[String, ListBuffer[(Double, Double)]]

  def printIterTime(): Unit = {
    for ((name, timeList) <- timesPerIter) {
      print("Timer: " + name)
      for (time <- timeList) {
        print(", " + time)
      }
      println
    }
  }

  def printStartEndTime(): Unit = {
    for ((name, timeList) <- timesStartEnd) {
      print("StartEndTimer: " + name)
      for (startEnd <- timeList) {
        print(", " + startEnd._1 + ", " + startEnd._2)
      }
      println
    }
  }
}
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
object GPUTimer extends Timer {

  class GPUTimerCls(val stream: CUstream, val eventType: String) {
    private var eventStart = new CUevent
    private var eventStop = new CUevent
    private var synced: Boolean = false
    private var elapsedTime: Array[Float] = new Array[Float](1)

    def start(): Unit = {
      JCudaDriver.cuEventCreate(eventStart, 0)
      JCudaDriver.cuEventCreate(eventStop, 0)
      JCudaDriver.cuEventRecord(eventStart, stream)
    }

    def stop(): Unit = {
      JCudaDriver.cuEventRecord(eventStop, stream)
    }

    def sync(): Unit = {
      JCudaDriver.cuEventSynchronize(eventStop)
      synced = true
    }

    def getTime: Float = {
      if (synced)
        elapsedTime(0)
      else {
        sync()
        JCudaDriver.cuEventElapsedTime(elapsedTime, eventStart, eventStop)
        elapsedTime(0)
      }
    }

    def shutDown(): Unit = {
      JCudaDriver.cuEventDestroy(eventStart)
      JCudaDriver.cuEventDestroy(eventStop)
    }
  }

  private var timers = new ListBuffer[GPUTimerCls]()

  def time[R](block: =>R, stream: CUstream, eventType: String): R = {
    val timer = new GPUTimerCls(stream, eventType)
    timer.start()
    val result = block
    timer.stop()
    timers += timer
    result
  }
  def getTimers:ListBuffer[GPUTimerCls] = timers

  def restart(): Unit = {
    // First we sync all the cuda streams and update the accumulated timer
    // for just this one iteration
    for (timer <- timers) {
      val name = timer.eventType
      val t = timer.getTime
      if (timesAccum.contains(name)) {
        timesAccum(name) += t
      } else {
        timesAccum(name) = t
      }
    }
    // Then we use the accumulated time to update the iteration timer
    for ((name, time) <- timesAccum) {
      if (!timesPerIter.contains(name))
        timesPerIter(name) = new ListBuffer[Double]
      timesPerIter(name) += time
      timesAccum(name) = 0.0
    }
    // Then we destroy the cuda event and remove all the times
    for (timer <- timers) {
      timer.shutDown()
      timers -= timer
    }
  }

  def sum: Float = {
    var totalTime: Float = 0
    timers.foreach (t => totalTime = totalTime + t.getTime)
    totalTime
  }

  def printStatsDetail(): Unit = {
    for (timer <- timers) {
      val time = timer.getTime
      println(timer.stream + "->" + timer.eventType +
        ": " + f"$time%.5f ms")
    }
    printStats()
  }

  def printStats(): Unit = {
    val totalTime = sum
    println("Total GPU Time: " + f"$totalTime%.5f ms")
  }
}
//
//// scalastyle:on
