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
wget -O people-detection.mp4 https://github.com/gsilva2016/sample-videos/raw/12_fps/Store-Aisle-Detection_12fps.mp4

echo "Downloading int8 quantized models"
cd object_detection/yolov5s/FP16-INT8/1; rm *.xml || true; rm *.bin || true;
wget https://github.com/dlstreamer/pipeline-zoo-models/raw/main/storage/yolov5s-416_INT8/FP16-INT8/yolov5s.bin
wget https://github.com/dlstreamer/pipeline-zoo-models/raw/main/storage/yolov5s-416_INT8/FP16-INT8/yolov5s.xml
cd ../../..; cd person-detection-retail-0013/FP16-INT8/1; rm *.xml || true; rm *.bin || true; 
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/person-detection-retail-0013/FP16-INT8/person-detection-retail-0013.bin
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/person-detection-retail-0013/FP16-INT8/person-detection-retail-0013.xml


echo "Building yolov5 OVMS-CAPI POC"
cd ../../..
./object_detection/build-ovms-capi.sh
