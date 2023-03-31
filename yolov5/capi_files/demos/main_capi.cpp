//*****************************************************************************
// Copyright 2023 Intel Corporation
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
#include <iomanip>

#include <signal.h>
#include <stdio.h>

// OpenCV for media decode
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

// Utilized for infernece output layer post-processing
#include <cmath>


#include "ovms.h"  // NOLINT

const char* MODEL_NAME = "yolov5s";
const uint64_t MODEL_VERSION = 0;
constexpr size_t DIM_COUNT = 4;
// YOLOV5 - 1x3x416x416 NCHW  
// TODO: Optimize to to convert model to NHWC. For now do the conversion.
const char* INPUT_NAME = "images";
constexpr size_t SHAPE[DIM_COUNT] = {1, 416, 416, 3};

// OpenCV video capture
cv::VideoCapture vidcap;
bool first_read;
size_t nextImgId;
size_t readLengthLimit;
cv::Mat img;
bool canRead;

bool openVideo(const std::string& videoFileName)
{
	if (!vidcap.open(videoFileName, cv::CAP_ANY)) {
		printf("Unable to open the video file %s\n", videoFileName.c_str());
	}

	return vidcap.isOpened();
}

bool getVideoFrame()
{
	int ret = vidcap.read(img);
	if (!ret)
        {
          //retry one time
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
          ret = vidcap.read(img);
        }
	return ret;
}

// end of OpenCV video capture

// Parse Yolov5 
//
const size_t batch_size = 1;
float confidence_threshold = .5;
float boxiou_threshold = .5;
float iou_threshold = 0.4;
const int classes =  80;
const float anchors[] = { 
		10.0, 
                13.0, 
                16.0,
                30.0,
                33.0,
                23.0,
                30.0,
                61.0,
                62.0,
                45.0,
                59.0,
                119.0,
                116.0,
                90.0,
                156.0,
                198.0,
                373.0,
                326.0
};

const int masks[] = {
		6,
                7,
                8,
                3,
                4,
                5,
                0,
                1,
                2
};

typedef struct DetectedResult {
	int frameId;
	float x;
	float y;
	float width;
	float height; 
	float confidence;
	int classId;
	const char *classText;
} DetectedResult;

std::vector<DetectedResult> _detectedResults;

static inline float sigmoid(float x) {
    return 1.f / (1.f + std::exp(-x));
}

static inline float linear(float x) {
    return x;
}

const std::string labels[80] = {
                "person",
                "bicycle",
                "car",
                "motorbike",
                "aeroplane",
                "bus",
                "train",
                "truck",
                "boat",
                "traffic light",
                "fire hydrant",
                "stop sign",
                "parking meter",
                "bench",
                "bird",
                "cat",
                "dog",
                "horse",
                "sheep",
                "cow",
                "elephant",
                "bear",
                "zebra",
                "giraffe",
                "backpack",
                "umbrella",
                "handbag",
                "tie",
                "suitcase",
                "frisbee",
                "skis",
                "snowboard",
                "sports ball",
                "kite",
                "baseball bat",
                "baseball glove",
                "skateboard",
                "surfboard",
                "tennis racket",
                "bottle",
                "wine glass",
                "cup",
                "fork",
                "knife",
                "spoon",
                "bowl",
                "banana",
                "apple",
                "sandwich",
                "orange",
                "broccoli",
                "carrot",
                "hot dog",
                "pizza",
                "donut",
                "cake",
                "chair",
                "sofa",
                "pottedplant",
                "bed",
                "diningtable",
                "toilet",
                "tvmonitor",
                "laptop",
                "mouse",
                "remote",
                "keyboard",
                "cell phone",
                "microwave",
                "oven",
                "toaster",
                "sink",
                "refrigerator",
                "book",
                "clock",
                "vase",
                "scissors",
                "teddy bear",
                "hair drier",
                "toothbrush"
};

const char * getClassLabelText(int classIndex)
{
        if (classIndex > 80)
                return "";

        return labels[classIndex].c_str();
}


int calculateEntryIndex(int totalCells, int lcoords, size_t lclasses, int location, int entry) {
    int n = location / totalCells;
    int loc = location % totalCells;
    return (n * (lcoords + lclasses) + entry) * totalCells + loc;
}

double intersectionOverUnion(const DetectedResult& o1, const DetectedResult& o2) {
    double overlappingWidth = std::fmin(o1.x + o1.width, o2.x + o2.width) - std::fmax(o1.x, o2.x);
    double overlappingHeight = std::fmin(o1.y + o1.height, o2.y + o2.height) - std::fmax(o1.y, o2.y);
    double intersectionArea =
        (overlappingWidth < 0 || overlappingHeight < 0) ? 0 : overlappingHeight * overlappingWidth;
    double unionArea = o1.width * o1.height + o2.width * o2.height - intersectionArea;
    return intersectionArea / unionArea;
}

void postprocess(std::vector<DetectedResult> &detectedResults, std::vector<DetectedResult> &outDetectedResults)	
{
  const bool useAdvancedPostprocessing = true;
    	  
  if (useAdvancedPostprocessing) {
        // Advanced postprocessing
        // Checking IOU threshold conformance
        // For every i-th object we're finding all objects it intersects with, and comparing confidence
        // If i-th object has greater confidence than all others, we include it into result
        for (const auto& obj1 : detectedResults) {
            bool isGoodResult = true;
            for (const auto& obj2 : detectedResults) {
                if (obj1.classId == obj2.classId && obj1.confidence < obj2.confidence &&
                    intersectionOverUnion(obj1, obj2) >= boxiou_threshold) {  // if obj1 is the same as obj2, condition
                                                                             // expression will evaluate to false anyway
                    isGoodResult = false;
                    break;
                }
            }
            if (isGoodResult) {
                outDetectedResults.push_back(obj1);
            }
        }
  } else {
        // Classic postprocessing
        std::sort(detectedResults.begin(), detectedResults.end(), [](const DetectedResult& x, const DetectedResult& y) {
            return x.confidence > y.confidence;
        });
        for (size_t i = 0; i < detectedResults.size(); ++i) {
            if (detectedResults[i].confidence == 0)
                continue;
            for (size_t j = i + 1; j < detectedResults.size(); ++j)
                if (intersectionOverUnion(detectedResults[i], detectedResults[j]) >= boxiou_threshold)
                    detectedResults[j].confidence = 0;
            outDetectedResults.push_back(detectedResults[i]);
        }
  } // end if/else
}

void postprocess(const uint64_t* output_shape, const void* voutputData, const size_t *input_shape, const size_t bytesize, const uint32_t dimCount, std::vector<DetectedResult> &detectedResults)
{
        if (!voutputData || !output_shape) {
                printf("*************DEBUG: Nothing to do.\n");
                return;
        }

	const int regionCoordsCount  = dimCount;
	const int sideH = output_shape[2]; // NCHW
	const int sideW = output_shape[3]; // NCHW
	const int regionNum = 3;
	const int scaleH = input_shape[1]; // NHWC
	const int scaleW = input_shape[2]; // NHWC

	auto entriesNum = sideW * sideH;
	const float* outData = reinterpret_cast<const float*>(voutputData);

	// Yolov5 uses sigmoid similar to v4 or linear
    	auto postprocessRawData = sigmoid;

	// --------------------------- Parsing YOLO Region output -------------------------------------
	for (int i = 0; i < entriesNum; ++i) {
         int row = i / sideW;
         int col = i % sideW;
         for (int n = 0; n < regionNum; ++n) {
            //--- Getting region data
            int obj_index = calculateEntryIndex(entriesNum,  regionCoordsCount, classes + 1 /* + confidence byte */, n * entriesNum + i,regionCoordsCount);
            int box_index = calculateEntryIndex(entriesNum, regionCoordsCount, classes + 1, n * entriesNum + i, 0);
	    float outdata = outData[obj_index];
            float scale = postprocessRawData(outData[obj_index]);


            //--- Preliminary check for confidence threshold conformance
	    int original_im_w = 1920;
	    int original_im_h = 1080;

            if (scale >= confidence_threshold) 
	    {
                //--- Calculating scaled region's coordinates
                float x, y;
                x = static_cast<float>((col + postprocessRawData(outData[box_index + 0 * entriesNum])) / sideW * original_im_w);
                y = static_cast<float>((row + postprocessRawData(outData[box_index + 1 * entriesNum])) / sideH * original_im_h);
                float height = static_cast<float>(std::exp(outData[box_index + 3 * entriesNum]) * anchors[2 * n + 1] * original_im_h / scaleH);
                float width = static_cast<float>(std::exp(outData[box_index + 2 * entriesNum]) * anchors[2 * n] * original_im_w / scaleW);

		//todo : need to update anchors based on masks
                DetectedResult obj;
                obj.x = std::clamp(x - width / 2, 0.f, static_cast<float>(original_im_w));
                obj.y = std::clamp(y - height / 2, 0.f, static_cast<float>(original_im_h));
                obj.width = std::clamp(width, 0.f, static_cast<float>(original_im_w - obj.x));
                obj.height = std::clamp(height, 0.f, static_cast<float>(original_im_h - obj.y));



                for (size_t j = 0; j < classes; ++j) {
                    int class_index = calculateEntryIndex(entriesNum, regionCoordsCount, classes + 1, n * entriesNum + i, regionCoordsCount + 1 + j);
                    float prob = scale * postprocessRawData(outData[class_index]);

                    //--- Checking confidence threshold conformance and adding region to the list
                    if (prob >= confidence_threshold) {
                        obj.confidence = prob;
                        obj.classId = j;
                        obj.classText = getClassLabelText(j);
                        detectedResults.push_back(obj);
			//printf("Adding boxes: %f %f %f %f, %f %f %f %f\n", x, y, width, height, obj.x,obj.y, obj.width,obj.height);
                    }
                }
            } // else
	 } // for
	} // for
    
        return;
}
// end of Parse Yolov5

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

void printInferenceResults(std::vector<DetectedResult> &results)
{
	for (auto & obj : results) {
	  std::cout << "Rect: [ " << obj.x << " , " << obj.y << " " << obj.width << ", " << obj.height << "] Class: " << obj.classText << "(" << obj.classId << ") Conf: " << obj.confidence << std::endl;
	}
}

void displayGUIInferenceResults(cv::Mat presenter, std::vector<DetectedResult> &results)
{
  //for (auto & obj : results) {
  //cv::imshow("Detection Results", presenter);
  //cv::waitKey(1);
}

void saveInferenceResultsAsVideo(cv::Mat &presenter, std::vector<DetectedResult> &results)
{
  for (auto & obj : results) {
    std::cout << "Drawing at " << obj.x << "," << obj.y << " " << obj.x+obj.width << " " << obj.y+obj.height << std::endl;
    cv::rectangle( presenter,
         cv::Point(obj.x,obj.y),
         cv::Point(obj.x+obj.width, obj.y+obj.height ),
         cv::Scalar(255, 0, 0),
         2, cv::LINE_8 );

    cv::rectangle( presenter,
         cv::Point(100, 100),
         cv::Point(500, 500),
         cv::Scalar(0, 255, 0),
         4, cv::LINE_8 );
  }
  cv::imwrite("result.jpg", presenter);
}

int main(int argc, char** argv) {

    std::cout << std::setprecision(2) << std::fixed;

    std::string videoStream = "coca-cola-4465029.mp4";
    int server_grpc_port = 9178;
    int server_http_port = 11338;

    if (argc > 1)
    {
	    videoStream = argv[1];
    }
    if (argc > 3) {
	     server_grpc_port = std::stoi(argv[2]);
	     server_http_port = std::stoi(argv[3]);
    }

    std::cout << "Video Stream: " << videoStream.c_str() << std::endl;

    // Open video
    if (!openVideo(videoStream.c_str())) {
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

    OVMS_ServerSettingsSetGrpcPort(serverSettings, server_grpc_port);
    OVMS_ServerSettingsSetRestPort(serverSettings, server_http_port);

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

    auto initTime = std::chrono::high_resolution_clock::now();
    unsigned long numberOfFrames = 0;
    while (getVideoFrame()) {

    auto startTime = std::chrono::high_resolution_clock::now();

    // prepare request
    OVMS_InferenceRequest* request{nullptr};
    OVMS_InferenceRequestNew(&request, srv, MODEL_NAME, MODEL_VERSION);
    OVMS_InferenceRequestAddInput(request, INPUT_NAME, OVMS_DATATYPE_FP32, SHAPE, DIM_COUNT);

    cv::Mat newImage;
    cv::Mat floatImage;
    cv::Size networkSize(416, 416);

    // Resize to input network size
    cv::resize(img, newImage, networkSize, 0,0, cv::INTER_LINEAR);

    // Convert to needed FP32 input format
    newImage.convertTo(floatImage, CV_32F);

    const int DATA_SIZE = floatImage.step[0] * floatImage.rows;

    OVMS_InferenceRequestInputSetData(request, INPUT_NAME, reinterpret_cast<void*>(floatImage.data), DATA_SIZE , OVMS_BUFFERTYPE_CPU, 0);    
    //std::cout << "Perform inference" << std::endl;


    // run sync request
    OVMS_InferenceResponse* response = nullptr;
    res = OVMS_Inference(srv, request, &response);
    if (res != nullptr) {
	std::cout << "OVMS_Inference failed " << std::endl;
        uint32_t code = 0;
        const char* details = 0;
        OVMS_StatusGetCode(res, &code);
        OVMS_StatusGetDetails(res, &details);
        std::cout << "Error occured during inference. Code:" << code
                  << ", details:" << details << std::endl;
    } else {
	//std::cout << "Inference completed " << std::endl;
    }

    // read output
    uint32_t outputCount = 0;
    OVMS_InferenceResponseGetOutputCount(response, &outputCount);
    //std::cout << "Ouputs returned: " << outputCount << std::endl;

    const void* voutputData3;
    const void* voutputData2;
    const void* voutputData1;
    size_t bytesize1 = 0;
    size_t bytesize2 = 0;
    size_t bytesize3 = 0;
    uint32_t outputId = outputCount - 1;
    OVMS_DataType datatype1 = (OVMS_DataType)42;
    OVMS_DataType datatype2 = (OVMS_DataType)42;
    OVMS_DataType datatype3 = (OVMS_DataType)42;
    const uint64_t* shape1{nullptr};
    const uint64_t* shape2(nullptr);
    const uint64_t* shape3(nullptr);
    uint32_t dimCount1 = 0;
    uint32_t dimCount2 = 0;
    uint32_t dimCount3 = 0;
    OVMS_BufferType bufferType1 = (OVMS_BufferType)42;
    OVMS_BufferType bufferType2 = (OVMS_BufferType)42;
    OVMS_BufferType bufferType3 = (OVMS_BufferType)42;
    uint32_t deviceId1 = 42;
    uint32_t deviceId2 = 42;
    uint32_t deviceId3 = 42;
    const char* outputName1{nullptr};
    const char* outputName2{nullptr};
    const char* outputName3{nullptr};

    OVMS_InferenceResponseGetOutput(response, outputId, &outputName1, &datatype1, &shape1, &dimCount1, &voutputData1, &bytesize1, &bufferType1, &deviceId1);
    //std::cout << "------------> " << outputName1  << " " << dimCount1 << " " << shape1[0] << " " << shape1[1] << " " << shape1[2] << " " << shape1[3] << " " << bytesize1 << " " << outputCount << std::endl;

    //OVMS_InferenceResponseGetOutput(response, outputId - 1, &outputName2, &datatype2, &shape2, &dimCount2, &voutputData2, &bytesize2, &bufferType2, &deviceId2);
    //std::cout << "------------> " << outputName2  << " " << dimCount2 << " " << shape2[0] << " " << shape2[1] << " " << shape2[2] << " " << shape2[3] << " " << bytesize2 << " " << outputCount << std::endl;

    //OVMS_InferenceResponseGetOutput(response, outputId-2, &outputName3, &datatype3, &shape3, &dimCount3, &voutputData3, &bytesize3, &bufferType3, &deviceId3);
    //std::cout << "------------> " << outputName3  << " " << dimCount3 << " " << shape3[0] << " " << shape3[1] << " " << shape3[2] << " " << shape3[3] << " " << bytesize3 << " " << outputCount << std::endl;

    std::vector<DetectedResult> detectedResults;
    std::vector<DetectedResult> detectedResultsFiltered;
    postprocess(shape1, voutputData1, SHAPE, bytesize1, dimCount1, detectedResults);
    postprocess(detectedResults, detectedResultsFiltered);
    printInferenceResults(detectedResultsFiltered);
    //displayGUIInferenceResults(img, detectedResultsFiltered);
    //saveInferenceResultsAsVideo(img, detectedResultsFiltered);

    /*
    std::stringstream ss;
    ss << "Got response from OVMS via C-API. "
       << "Request for model: " << MODEL_NAME
       << "; version: " << MODEL_VERSION
       << "ms; output name: " << outputName
       << "; response with values:\n";
    for (size_t i = 0; i < shape[1]; ++i) {
        ss << *(reinterpret_cast<const float*>(voutputData) + i) << " ";
    }
    */
    //std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    // todo comment below to run against full video or pass in time interval and remove instead
    auto endTime = std::chrono::high_resolution_clock::now();
    std::cout << "Pipeline Latency (ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() << std::endl;

    numberOfFrames++;
    unsigned long elapsedTotalTime = ((std::chrono::duration_cast<std::chrono::seconds>(endTime-initTime)).count());
    if (elapsedTotalTime > 0)
    {
      float fps = (float)numberOfFrames/(float)((std::chrono::duration_cast<std::chrono::milliseconds>(endTime-initTime)).count()/1000); // convert to seconds
      std::cout << "FPS: " << fps << std::endl;
    }
    //break;
    } // end while get frames


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
