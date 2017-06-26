#!/bin/bash

#
# Check for Maven
# 
type mvn >/dev/null 2>&1 || {
  echo "Maven is missing to compile the source files. Kindly install maven to proceed."
  exit -1
}

export MAVEN_OPTS="-Xmx2048m -XX:MaxPermSize=1024m"

CUDA=0
CUDAVER=0
#
# Identify the CUDA runtime version to activate the correct profile
# Install Cuda for your platform from
# https://developer.nvidia.com/cuda-downloads.
# 
type nvcc >/dev/null 2>&1 && CUDA=1

if [[ $CUDA != 0 ]]; then
  CUDAVER=`nvcc --version | tail -1 | awk '{ print \$5 }' | cut -d ',' -f 1`
  if [[ $CUDAVER == "7.5" ]]; then
    echo "Identified CUDA version is 7.5"
    CUDAVER="jcuda75"
  elif [[ $CUDAVER == "7.0" ]]; then
    echo "Identified CUDA version is 7.0a"
    CUDAVER="jcuda70a"
  else
    echo "Not a supported version. Installation will fallback to default"
  fi
fi

MVN_CMD="mvn"

if [[ $CUDAVER != 0 ]]; then
  MVN_ARGS="-P$CUDAVER -Dmaven.compiler.showWarnings=true -Dmaven.compiler.showDeprecation=true"
else
  MVN_ARGS="-Dmaven.compiler.showWarnings=true -Dmaven.compiler.showDeprecation=true"
fi

echo "Executing :: $MVN_CMD $MVN_ARGS -DskipTests $@ clean install "

if [[ $1 == "clean" ]]; then
	$MVN_CMD $MVN_ARGS -DskipTests $@ clean install 2>&1
else
$MVN_CMD $MVN_ARGS -DskipTests $@ install -pl gpu-enabler 2>&1
fi

if [[ $? -eq 0 ]]; then
	echo "Successfully build gpu-enabler, now building Spark assembly"
	cp gpu-enabler/target/gpu-enabler_2.11-1.0.0.jar /home/xiangyu/Dropbox/spark/spark-2.1.0/assembly/target/scala-2.11/jars/
#	cd ${SPARK_HOME}
#	./build/mvn -pl :spark-assembly_2.11 -DskipTests source:jar install
#	cd -
else
	echo "Failed to build gpu-enabler, exiting"
fi
	

# ./utils/embed.sh -d  gpu-enabler_2.11-1.0.0.jar
