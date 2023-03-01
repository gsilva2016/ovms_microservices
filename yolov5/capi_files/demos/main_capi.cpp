//*****************************************************************************
// Copyright 2022 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>
#include <vector>

#include <signal.h>
#include <stdio.h>

// OpenCV for media decode
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "ovms.h"  // NOLINT

const char* MODEL_NAME = "yolov5s";
const uint64_t MODEL_VERSION = 1;
const char* INPUT_NAME = "images";
constexpr size_t DIM_COUNT = 4;
constexpr size_t SHAPE[DIM_COUNT] = {1, 3, 416, 416};

// OpenCV video capture
cv::VideoCapture vidcap;
bool first_read;
size_t nextImgId;
size_t readLengthLimit;
cv::Mat img;
bool canRead;

bool openVideo(const std::string& videoFileName)
{
	//auto startTime = std::chrono::steady_clock::now();

        //std::ifstream file(videoFileName.c_str());
        //if (!file.good()) {
	//	printf("Can not find the video file\n");
	//}

	if (!vidcap.open(videoFileName, cv::CAP_ANY)) {
		printf("Unable to open the video file %s\n", videoFileName.c_str());
		//return false;
	}

	return vidcap.isOpened();
}

bool getVideoFrame()
{
	int ret = vidcap.read(img);
        //printf("Return read val: %d\n", ret);
	return ret;
}

// end of OpenCV video capture

namespace {
volatile sig_atomic_t shutdown_request = 0;
}

static void onInterrupt(int status) {
    shutdown_request = 1;
}

static void onTerminate(int status) {
    shutdown_request = 1;
}

static void onIllegal(int status) {
    shutdown_request = 2;
}

static void installSignalHandlers() {
    static struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = onInterrupt;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    static struct sigaction sigTermHandler;
    sigTermHandler.sa_handler = onTerminate;
    sigemptyset(&sigTermHandler.sa_mask);
    sigTermHandler.sa_flags = 0;
    sigaction(SIGTERM, &sigTermHandler, NULL);

    static struct sigaction sigIllHandler;
    sigIllHandler.sa_handler = onIllegal;
    sigemptyset(&sigIllHandler.sa_mask);
    sigIllHandler.sa_flags = 0;
    sigaction(SIGILL, &sigIllHandler, NULL);
}

int main(int argc, char** argv) {

    // Open video
    if (!openVideo("coca-cola-4465029.mp4")) {
	    std::cout << "Nothing to do." << std::endl;
	    return 0;
    }

    installSignalHandlers();

    OVMS_ServerSettings* serverSettings = 0;
    OVMS_ModelsSettings* modelsSettings = 0;
    OVMS_Server* srv;

    OVMS_ServerSettingsNew(&serverSettings);
    OVMS_ModelsSettingsNew(&modelsSettings);
    OVMS_ServerNew(&srv);

    OVMS_ServerSettingsSetGrpcPort(serverSettings, 9178);
    OVMS_ServerSettingsSetRestPort(serverSettings, 11338);

    OVMS_ServerSettingsSetLogLevel(serverSettings, OVMS_LOG_DEBUG);
    OVMS_ModelsSettingsSetConfigPath(modelsSettings, "config_yolov5.json");

    OVMS_Status* res = OVMS_ServerStartFromConfigurationFile(srv, serverSettings, modelsSettings);

    if (res) {
        uint32_t code = 0;
        const char* details = nullptr;

        OVMS_StatusGetCode(res, &code);
        OVMS_StatusGetDetails(res, &details);
        std::cerr << "error during start: code:" << code << "; details:" << details << std::endl;

        OVMS_StatusDelete(res);

        OVMS_ServerDelete(srv);
        OVMS_ModelsSettingsDelete(modelsSettings);
        OVMS_ServerSettingsDelete(serverSettings);
        return 1;
    }

    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "Server ready for inference " << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;

    // prepare request
    OVMS_InferenceRequest* request{nullptr};
    OVMS_InferenceRequestNew(&request, srv, MODEL_NAME, MODEL_VERSION);
    OVMS_InferenceRequestAddInput(request, INPUT_NAME, OVMS_DATATYPE_FP32, SHAPE, DIM_COUNT);

    while (getVideoFrame()) {

    cv::Mat newImage;
    // Reize to expected network size
    cv::Size networkSize(416, 416);
    cv::resize(img, newImage, networkSize, 0,0, cv::INTER_CUBIC);
    //newImage.convertTo(newImage, CV_8UC3);
    //newImage = img.clone();
    //std::cout << "--------------------------" << newImage.channels() << " " << newImage.rows << " " << sizeof(float) * 416 * 416 * 3 << std::endl;

    // normalize
    // TODO

    OVMS_InferenceRequestInputSetData(request, INPUT_NAME, reinterpret_cast<void*>(newImage.data), sizeof(float) * newImage.cols * newImage.rows * newImage.channels(), OVMS_BUFFERTYPE_CPU, 0);

    // run sync request
    OVMS_InferenceResponse* response = nullptr;
    res = OVMS_Inference(srv, request, &response);
    if (res != nullptr) {
        uint32_t code = 0;
        const char* details = 0;
        OVMS_StatusGetCode(res, &code);
        OVMS_StatusGetDetails(res, &details);
        std::cout << "Error occured during inference. Code:" << code
                  << ", details:" << details << std::endl;
    }
    // read output
    uint32_t outputCount = 0;
    OVMS_InferenceResponseGetOutputCount(response, &outputCount);
    const void* voutputData;
    size_t bytesize = 0;
    uint32_t outputId = outputCount - 1;
    OVMS_DataType datatype = (OVMS_DataType)42;
    const uint64_t* shape{nullptr};
    uint32_t dimCount = 0;
    OVMS_BufferType bufferType = (OVMS_BufferType)42;
    uint32_t deviceId = 42;
    const char* outputName{nullptr};
    OVMS_InferenceResponseGetOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId);

    std::stringstream ss;
    ss << "Got response from OVMS via C-API. "
       << "Request for model: " << MODEL_NAME
       << "; version: " << MODEL_VERSION
       << "ms; output name: " << outputName
       << "; response with values:\n";
    for (size_t i = 0; i < shape[1]; ++i) {
        ss << *(reinterpret_cast<const float*>(voutputData) + i) << " ";
    }
    break;
    } // end while get frames

    //std::vector<float> expectedOutput;
    //std::transform(data.begin(), data.end(), std::back_inserter(expectedOutput),
    //    [](const float& s) -> float {
    //        return s + 1;
    //    });

    //if (std::memcmp(voutputData, expectedOutput.data(), expectedOutput.size() * sizeof(float)) != 0) {
    //    std::cout << "Incorrect result of inference" << std::endl;
    //}
    //std::cout << ss.str() << std::endl;


    // comment line below to have app running similarly to OVMS
    shutdown_request = 1;
    while (shutdown_request == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    std::cout << "No more job to be done, will shut down" << std::endl;

    OVMS_ServerDelete(srv);
    OVMS_ModelsSettingsDelete(modelsSettings);
    OVMS_ServerSettingsDelete(serverSettings);

    fprintf(stdout, "main() exit\n");
    return 0;
}
