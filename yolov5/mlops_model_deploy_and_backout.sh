#!/bin/bash

echo "Deploying new Yolov5s model version."
mkdir -p FP16-INT8/2
pushd FP16-INT8/2
wget https://raw.githubusercontent.com/dlstreamer/pipeline-zoo-models/main/storage/yolov5s-416/FP32/yolov5s.xml
wget https://raw.githubusercontent.com/dlstreamer/pipeline-zoo-models/main/storage/yolov5s-416/FP32/yolov5s.bin
popd


sleep 5


echo "Backing out newly deployed Yolov5s model version"
rm -R FP16-INT8/2
