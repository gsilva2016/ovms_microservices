#!/bin/bash

sudo apt update; sudo apt install -y wget git

echo "Building custom OpenVINO Model Server image with GStreamer video decoding enabled in OpenCV"
git clone https://github.com/G2020sudo/model_server.git
cd model_server
make
cd ..

echo "Downloading sample video coca-cola-4465029.mp4 https://www.pexels.com/video/4465029/download/"
cd yolov5
wget -O coca-cola-4465029.mp4 https://www.pexels.com/video/4465029/download/

echo "Downloading quantized int yolov5s model"
cd FP16-INT8/1
wget https://github.com/dlstreamer/pipeline-zoo-models/raw/main/storage/yolov5s-416_INT8/FP16-INT8/yolov5s.bin
wget https://github.com/dlstreamer/pipeline-zoo-models/raw/main/storage/yolov5s-416_INT8/FP16-INT8/yolov5s.xml


echo "TODO: Download and build yolov5 post processing libraries"
# REF: https://github.com/dlstreamer/pipeline-zoo-models/blob/main/storage/yolov5s-416_INT8/yolov5s.json
# REF: https://github.com/search?q=repo%3Adlstreamer%2Fdlstreamer%20yolov5&type=code

# Future todo: use for mlops demo to push more accurate version
#mkdir -p FP16-INT8/2
#wget https://github.com/dlstreamer/pipeline-zoo-models/raw/main/storage/yolov5m-460_INT8/FP32-INT8/yolov5m.bin
#wget https://github.com/dlstreamer/pipeline-zoo-models/raw/main/storage/yolov5m-460_INT8/FP32-INT8/yolov5m.xml

echo "Building yolov5 OVMS-CAPI POC"
cd ../..
./build-ovms-capi.sh
