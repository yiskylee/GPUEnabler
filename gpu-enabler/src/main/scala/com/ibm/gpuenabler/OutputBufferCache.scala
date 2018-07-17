package com.ibm.gpuenabler
import scala.collection.concurrent.TrieMap

object OutputBufferCache {
  val cache = new TrieMap[String, OutputBufferWrapper[_]]
  def getOrElseUpdate(key: String, value: OutputBufferWrapper[_]): OutputBufferWrapper[_]
  = cache.getOrElseUpdate(key, value)
  def contains(key: String): Boolean = cache.contains(key)
  def update(key: String, value: OutputBufferWrapper[_]): Unit = {
    cache.update(key, value)
  }
  def get(key: String) = cache.get(key)
}
