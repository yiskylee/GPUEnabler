package com.ibm.gpuenabler
import scala.collection.concurrent.TrieMap

object InputBufferCache {
  val cache = new TrieMap[String, InputBufferWrapper[_]]
  def getOrElseUpdate(key: String, value: InputBufferWrapper[_]): InputBufferWrapper[_]
    = cache.getOrElseUpdate(key, value)
  def contains(key: String): Boolean = cache.contains(key)
  def update(key: String, value: InputBufferWrapper[_]): Unit = {
    cache.update(key, value)
  }
  def get(key: String): Option[InputBufferWrapper[_]] = cache.get(key)
}