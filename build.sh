#!/bin/bash

sudo apt update; sudo apt install -y wget git

echo "Download OpenVINO Model Server 2023-gpu"
docker pull openvino/model_server:2023.0-gpu

echo "Downloading sample video coca-cola-4465029.mp4 https://www.pexels.com/video/4465029/download/"
cd object_detection
wget -O coca-cola-4465029.mp4 https://www.pexels.com/video/4465029/download/
wget -O people-detection.mp4 https://github.com/gsilva2016/sample-videos/raw/12_fps/Store-Aisle-Detection_12fps.mp4
# 4k@24
wget -O crow-travelers-transport-terminal.mp4 https://www.pexels.com/download/video/3740034/

echo "Downloading FP32 and INT8 quantized models"
# Open source - Yolov5s FP32 version1
cd model_repo/
cd yolov5s/1; rm *.xml || true; rm *.bin || true;
# FP32
#wget https://github.com/dlstreamer/pipeline-zoo-models/raw/main/storage/yolov5s-416/FP32/yolov5s.xml
#wget https://github.com/dlstreamer/pipeline-zoo-models/raw/main/storage/yolov5s-416/FP32/yolov5s.bin
# INT8
wget https://github.com/dlstreamer/pipeline-zoo-models/raw/main/storage/yolov5s-416_INT8/FP16-INT8/yolov5s.bin
wget https://github.com/dlstreamer/pipeline-zoo-models/raw/main/storage/yolov5s-416_INT8/FP16-INT8/yolov5s.xml

# OpenVINO Model Zoo - Retail Person Detection INT8
cd ../..; cd person-detection-retail-0013/1; rm *.xml || true; rm *.bin || true; 
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/person-detection-retail-0013/FP16-INT8/person-detection-retail-0013.bin
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/person-detection-retail-0013/FP16-INT8/person-detection-retail-0013.xml

# OpenVINO Model Zoo - text-detection INT8
cd ../..; cd text-detect-0002/1; rm *.xml || true; rm *.bin || true; 
wget https://github.com/dlstreamer/pipeline-zoo-models/raw/main/storage/horizontal-text-detection-0002/FP16-INT8/horizontal-text-detection-0002.bin
wget https://github.com/dlstreamer/pipeline-zoo-models/raw/main/storage/horizontal-text-detection-0002/FP16-INT8/horizontal-text-detection-0002.xml

# OpenVINO Model Zoo - Face - https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/face-detection-retail-0005
cd ../..; cd face-detect-retail-0005/1; rm *.xml || true; rm *.bin || true;
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/face-detection-retail-0005/FP16-INT8/face-detection-retail-0005.bin
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/face-detection-retail-0005/FP16-INT8/face-detection-retail-0005.xml

# OpenVINO Model Zoo - Face Landmarks - https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/facial-landmarks-35-adas-0002
cd ../..; cd face-landmarks-0002/1; rm *.xml || true; rm *.bin || true;
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/facial-landmarks-35-adas-0002/FP16-INT8/facial-landmarks-35-adas-0002.bin
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/facial-landmarks-35-adas-0002/FP16-INT8/facial-landmarks-35-adas-0002.xml

# OpenVINO Model Zoo - Face Reid - https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/face-reidentification-retail-0095
cd ../..; cd face-reid-retail-0095/1; rm *.xml || true; rm *.bin || true;
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/face-reidentification-retail-0095/FP16-INT8/face-reidentification-retail-0095.bin
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/face-reidentification-retail-0095/FP16-INT8/face-reidentification-retail-0095.xml

echo "Building yolov5 OVMS-CAPI POC"
cd ../../..
./build-ovms-capi.sh
