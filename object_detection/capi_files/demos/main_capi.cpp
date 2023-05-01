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

// Utilized for OpenCV media decode
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

// Utilized for infernece output layer post-processing
#include <cmath>

#include "ovms.h"  // NOLINT

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


/* OpenCV media decode */
cv::VideoCapture _vidcap;
cv::Mat _img;
int _video_input_width = 0;  // Get from media _img 
int _video_input_height = 0; // Get from media _img

/* Inference results */
std::vector<DetectedResult> _detectedResults;

class ObjectDetectionInterface {
public:
    virtual ~ObjectDetectionInterface() {}

    virtual const char* getModelName() = 0;
    virtual const uint64_t getModelVersion() = 0;
    virtual const char* getModelInputName() = 0;
    virtual const  size_t getModelDimCount() = 0;
    virtual const std::vector<int> getModelInputShape() = 0;
    virtual const std::string getClassLabelText(int classIndex) = 0;
    virtual void postprocess(const uint64_t* output_shape, const void* voutputData, const size_t bytesize, const uint32_t dimCount, std::vector<DetectedResult> &detectedResults) = 0;

    static inline float sigmoid(float x) {
        return 1.f / (1.f + std::exp(-x));
    }

    static inline float linear(float x) {
        return x;
    }

    double intersectionOverUnion(const DetectedResult& o1, const DetectedResult& o2) {
        double overlappingWidth = std::fmin(o1.x + o1.width, o2.x + o2.width) - std::fmax(o1.x, o2.x);
        double overlappingHeight = std::fmin(o1.y + o1.height, o2.y + o2.height) - std::fmax(o1.y, o2.y);
        double intersectionArea = (overlappingWidth < 0 || overlappingHeight < 0) ? 0 : overlappingHeight * overlappingWidth;
        double unionArea = o1.width * o1.height + o2.width * o2.height - intersectionArea;
        return intersectionArea / unionArea;
    }

    void postprocess(std::vector<DetectedResult> &detectedResults, std::vector<DetectedResult> &outDetectedResults)	
    {
        
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
            } //end for
        } // end if
    } // end postprocess filter


protected:
    float confidence_threshold = .9;
    float boxiou_threshold = .4;
    float iou_threshold = 0.4;
    int classes =  80;
    bool useAdvancedPostprocessing = false;

};

class SSD : public ObjectDetectionInterface {
public:

    SSD() {
        confidence_threshold = .9;
        classes = 2;
        std::cout << "Using object detection type person-detection-retail-0013" << std::endl;
    }

    const char* getModelName() {
        return MODEL_NAME;
    }

    const uint64_t getModelVersion() {
        return MODEL_VERSION;
    }

    const char* getModelInputName() {
        return INPUT_NAME;
    }

    const size_t getModelDimCount() {
        return DIM_COUNT;
    }

    const std::vector<int> getModelInputShape() {
        std::vector<int> shape{1, 320, 544, 3};
        return shape;
    }

    const std::string getClassLabelText(int classIndex) {
        return (classIndex == 1 ? "Person" : "Unknown");
    }

    /*
    * Reference: https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/common/cpp/models/src/detection_model_yolo.cpp
    * TODO: Move a shared lib. 
    */
    void postprocess(const uint64_t* output_shape, const void* voutputData, const size_t bytesize, const uint32_t dimCount, std::vector<DetectedResult> &detectedResults)
    {
        if (!voutputData || !output_shape) {
            // nothing to do
            return;
        }
            // detection_out 4 1 1 200 7 5600 1
        const int numberOfDetections = output_shape[2];
        const int objectSize = output_shape[3];
        //const int scaleH = input_shape[1]; // NHWC
        //const int scaleW = input_shape[2]; // NHWC
        const float* outData = reinterpret_cast<const float*>(voutputData);
        int original_im_w = _video_input_width; 
        int original_im_h = _video_input_height;
        std::vector<int> input_shape = getModelInputShape();
        float network_h = (float) input_shape[1];
        float network_w = (float) input_shape[2];
        printf("Network %f %f numDets %d \n", network_h, network_w, numberOfDetections);

        for (int i = 0; i < numberOfDetections; i++)
        {
            float image_id = outData[i * objectSize + 0];
            if (image_id < 0)
                break;

            float confidence = outData[i * objectSize + 2];

            if (confidence > confidence_threshold ) {
                DetectedResult obj;
                        obj.x = std::clamp(outData[i * objectSize + 3] * network_w, 0.f, static_cast<float>(network_w)); // std::clamp(outData[i * objectSize +3], 0.f,network_w);
                        obj.y = std::clamp(outData[i * objectSize + 4] * network_h, 0.f, static_cast<float>(network_h)); //std::clamp(outData[i * objectSize +4], 0.f,network_h);
                        obj.width = std::clamp(outData[i * objectSize + 5] * network_w, 0.f, static_cast<float>(network_w)) - obj.x; // std::clamp(outData[i*objectSize+5],0.f,network_w-obj.x);
                        obj.height = std::clamp(outData[i * objectSize + 6] * network_h, 0.f, static_cast<float>(network_h)) - obj.y; // std::clamp(outData[i*objectSize+6],0.f, network_h-obj.y);
                obj.confidence = confidence;
                            obj.classId = outData[i * objectSize + 1];
                            obj.classText = getClassLabelText(obj.classId).c_str();
                //if (strncmp(obj.classText, "person", sizeof("person") != 0 ))
                //	continue;
                if (obj.classId != 1)
                printf("---------found: %s\n", obj.classText);
                
                            detectedResults.push_back(obj);

            } // end if confidence
        } // end for
    } // End of SSD Person Detection Post-Processing


private:
    /* Model Serving Info for https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/person-detection-retail-0013 */
    // SSD - 1x3x320x544 NCHW     
    const char* MODEL_NAME = "ssd-persondet";
    const uint64_t MODEL_VERSION = 0;
    const size_t DIM_COUNT = 4;
    const char* INPUT_NAME = "data";    
};

class Yolov5 : public ObjectDetectionInterface
{
public:

    Yolov5()
    {
        confidence_threshold = .5;
        classes = 80;

        std::cout << "Using object detection type Yolov5" << std::endl;
    }

    const char* getModelName() {
        return MODEL_NAME;
    }

    const uint64_t getModelVersion() {
        return MODEL_VERSION;
    }

    const char* getModelInputName() {
        return INPUT_NAME;
    }

    const size_t getModelDimCount() {
        return DIM_COUNT;
    }

    const std::vector<int> getModelInputShape() {
        std::vector<int> shape{1, 416, 416, 3};
        return shape;
    }

    const std::string getClassLabelText(int classIndex) {
        if (classIndex > 80)
            return "Unknown";
        return labels[classIndex];
    }

    int calculateEntryIndex(int totalCells, int lcoords, size_t lclasses, int location, int entry) {
        int n = location / totalCells;
        int loc = location % totalCells;
        return (n * (lcoords + lclasses) + entry) * totalCells + loc;
    }

    void postprocess(const uint64_t* output_shape, const void* voutputData, const size_t bytesize, const uint32_t dimCount, std::vector<DetectedResult> &detectedResults)
    {
        if (!voutputData || !output_shape) {
            // nothing to do
            return;
        }

        const int regionCoordsCount  = dimCount;
        const int sideH = output_shape[2]; // NCHW
        const int sideW = output_shape[3]; // NCHW
        const int regionNum = 3;
        std::vector<int> input_shape = getModelInputShape();
        const int scaleH = input_shape[1]; // NHWC
        const int scaleW = input_shape[2]; // NHWC

        auto entriesNum = sideW * sideH;
        const float* outData = reinterpret_cast<const float*>(voutputData);
            int original_im_w = _video_input_width; 
        int original_im_h = _video_input_height;

        auto postprocessRawData = sigmoid; //sigmoid or linear

        for (int i = 0; i < entriesNum; ++i) {
            int row = i / sideW;
            int col = i % sideW;

            for (int n = 0; n < regionNum; ++n) {
            
                int obj_index = calculateEntryIndex(entriesNum,  regionCoordsCount, classes + 1 /* + confidence byte */, n * entriesNum + i,regionCoordsCount);
                int box_index = calculateEntryIndex(entriesNum, regionCoordsCount, classes + 1, n * entriesNum + i, 0);
                float outdata = outData[obj_index];
                float scale = postprocessRawData(outData[obj_index]);

                if (scale >= confidence_threshold) {
                    float x, y,height,width;
                    x = static_cast<float>((col + postprocessRawData(outData[box_index + 0 * entriesNum])) / sideW * original_im_w);
                    y = static_cast<float>((row + postprocessRawData(outData[box_index + 1 * entriesNum])) / sideH * original_im_h);
                    height = static_cast<float>(std::pow(2*postprocessRawData(outData[box_index + 3 * entriesNum]),2) * anchors_13[2 * n + 1] * original_im_h / scaleH  );
                    width = static_cast<float>(std::pow(2*postprocessRawData(outData[box_index + 2 * entriesNum]),2) * anchors_13[2 * n] * original_im_w / scaleW  );

                    DetectedResult obj;
                    obj.x = std::clamp(x - width / 2, 0.f, static_cast<float>(original_im_w));
                    obj.y = std::clamp(y - height / 2, 0.f, static_cast<float>(original_im_h));
                    obj.width = std::clamp(width, 0.f, static_cast<float>(original_im_w - obj.x));
                    obj.height = std::clamp(height, 0.f, static_cast<float>(original_im_h - obj.y));

                    for (size_t j = 0; j < classes; ++j) {
                        int class_index = calculateEntryIndex(entriesNum, regionCoordsCount, classes + 1, n * entriesNum + i, regionCoordsCount + 1 + j);
                        float prob = scale * postprocessRawData(outData[class_index]);

                        if (prob >= confidence_threshold) {
                            obj.confidence = prob;
                            obj.classId = j;
                            obj.classText = getClassLabelText(j).c_str();
                            detectedResults.push_back(obj);
                        }
                    }
                } // end else
            } // end for
        } // end for
    }
// End of Yolov5 Post-Processing

private:
    /* Yolov5s Model Serving Info */
    // YOLOV5 - 1x3x416x416 NCHW  
    const char* MODEL_NAME = "yolov5s";
    const uint64_t MODEL_VERSION = 0;
    const size_t DIM_COUNT = 4;
    const char* INPUT_NAME = "images";

    // Anchors by region/output layer
    const float anchors_52[6] = {
        10.0, 
        13.0, 
        16.0,
        30.0,
        33.0,
        23.0
    };

    const float anchors_26[6] = {
        30.0,
        61.0,
        62.0,
        45.0,
        59.0,
        119.0
    };

    const float anchors_13[6] = {
        116.0,
        90.0,
        156.0,
        198.0,
        373.0,
        326.0
    };

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

};


bool openVideo(const std::string& videoFileName)
{
    // Default to GSTREAMER for media decode
	if (!_vidcap.open(videoFileName, cv::CAP_GSTREAMER)) {
		printf("Unable to open the video stream %s\n", videoFileName.c_str());
	}

	return _vidcap.isOpened();
}

bool getVideoFrame()
{
	int ret = _vidcap.read(_img);
	if (!ret)
    {
        //retry one time
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        ret = _vidcap.read(_img);
    }
    
    if ( ret && _video_input_height == 0 )
    {
        cv::Size s = _img.size();
        _video_input_height = s.height;
        _video_input_width = s.width;
    }
	return ret;
}


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
     for (auto & obj : results) {
        //const float scaler_w = 416.0f/video_input_width;
        //const float scaler_h = 416.0f/video_input_height;
	
	const float x0 = obj.x; //std::min(std::max(0.0f, obj.x / video_input_width), 1.0f) * video_input_width;
        const float y0 = obj.y; //std::min(std::max(0.0f, obj.y / video_input_height), 1.0f) * video_input_height;
        const float x1 = obj.x + obj.width; // std::min(std::max(0.0f, (obj.x + obj.width) / video_input_width), 1.0f) * video_input_width;
        const float y1 = obj.y + obj.height; //std::min(std::max(0.0f, (obj.y + obj.height) / video_input_height), 1.0f) * video_input_height;

	//printf("DrawBox: %f %f %f %f\n", x0, y0, x1, y1);
	if (((y0 + y1) * (x0+x1)) <= 43442 )
	       continue;	
	printf("area: %f\n", ((y0 + y1) * (x0+x1)));

        cv::rectangle( presenter,
         cv::Point( (int)(x0),(int)(y0) ),
         cv::Point( (int)x1, (int)y1 ),
         cv::Scalar(255, 0, 0),
         2, cv::LINE_8 );
  } // end for
  //cv::imwrite("result.jpg", presenter);

  cv::imshow("Detection Results", presenter);
  cv::waitKey(1);
}

void saveInferenceResultsAsVideo(cv::Mat &presenter, std::vector<DetectedResult> &results)
{
    for (auto & obj : results) {

        const float scaler_w = 416.0f/_video_input_width;
        const float scaler_h = 416.0f/_video_input_height;
        //std::cout << " Scalers " << scaler_w << " " << scaler_h << std::endl;
        //std::cout << "xDrawing at " << (int)obj.x*scaler_w << "," << (int)obj.y*scaler_h << " " << (int)(obj.x+obj.width)*scaler_w << " " << (int) (obj.y+obj.height)* scaler_h << std::endl;
    
        cv::rectangle( presenter,
         cv::Point( (int)(obj.x*scaler_w),(int)(obj.y*scaler_h) ),
         cv::Point( (int)((obj.x+obj.width) * scaler_w), (int)((obj.y+obj.height)*scaler_h) ),
         cv::Scalar(255, 0, 0),
         4, cv::LINE_8 );  
  } // end for
  cv::imwrite("result.jpg", presenter);
}

int main(int argc, char** argv) {
    std::cout << std::setprecision(2) << std::fixed;

    std::string videoStream = "people-detection.mp4";
    int server_grpc_port = 9178;
    int server_http_port = 11338;
    std::string detectionType = "yolov5";

    if (argc > 1) {
	    videoStream = argv[1];
    }

    if (argc > 3) {
	     server_grpc_port = std::stoi(argv[2]);
	     server_http_port = std::stoi(argv[3]);
    }

    if (argc > 4) {
        if (detectionType != "yolov5" && detectionType != "people-detection-retail-0013") {
            std::cout << "detectionType option must be yolov5 or people-detection-retail-0013" << std::endl;
            return 1;
        }
	    detectionType = argv[4];
    }
    
    ObjectDetectionInterface* objDet;
    if (detectionType == "yolov5") {
        objDet = new Yolov5();
    }
    else {
        objDet = new SSD();
    }

    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "Opening Video Stream: " << videoStream.c_str() << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;

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

    // Loads SSD & YOLO models in single OVMS C-API server
    OVMS_ModelsSettingsSetConfigPath(modelsSettings, "config_object_detection.json");

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
    const int MODEL_DIM_COUNT = objDet->getModelDimCount();
    const std::vector<int> vmodel_input_shape = objDet->getModelInputShape();
    uint64_t model_input_shape[MODEL_DIM_COUNT] = { 0 };
    std::copy(vmodel_input_shape.begin(), vmodel_input_shape.end(), model_input_shape);
    std::cout << "Model Input: --------->" << model_input_shape << " " << MODEL_DIM_COUNT << std::endl;

    while (getVideoFrame()) {
        auto startTime = std::chrono::high_resolution_clock::now();

        // Final FP32 image for inference
        cv::Mat floatImage;

        // prepare request
        OVMS_InferenceRequest* request{nullptr};
        OVMS_InferenceRequestNew(&request, srv, objDet->getModelName(), objDet->getModelVersion());
        OVMS_InferenceRequestAddInput(
            request, 
            objDet->getModelInputName(), 
            OVMS_DATATYPE_FP32, 
            model_input_shape, 
            MODEL_DIM_COUNT
        );

        // Convert to needed FP32 input format
        _img.convertTo(floatImage, CV_32F);

        const int DATA_SIZE = floatImage.step[0] * floatImage.rows;

        OVMS_InferenceRequestInputSetData(
            request, 
            objDet->getModelInputName(), 
            reinterpret_cast<void*>(floatImage.data), 
            DATA_SIZE , 
            OVMS_BUFFERTYPE_CPU, 
            0
        );

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
        } 

        // read output. Yolov5s has 3 outputs...only processing one.
        // TODO: process remaining
        uint32_t outputCount = 0;
        OVMS_InferenceResponseGetOutputCount(response, &outputCount);
	    //printf("Got %d model outputs\n", outputCount);

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
        objDet->postprocess(shape1, voutputData1, bytesize1, dimCount1, detectedResults);
        objDet->postprocess(detectedResults, detectedResultsFiltered);
        //printInferenceResults(detectedResultsFiltered);
        displayGUIInferenceResults(_img, detectedResultsFiltered);
        //saveInferenceResultsAsVideo(img, detectedResultsFiltered);

        auto endTime = std::chrono::high_resolution_clock::now();
        std::cout << "Pipeline Latency (ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() << std::endl;

        numberOfFrames++;
        unsigned long elapsedTotalTime = ((std::chrono::duration_cast<std::chrono::seconds>(endTime-initTime)).count());
        if (elapsedTotalTime > 0) {
            float fps = (float)numberOfFrames/(float)((std::chrono::duration_cast<std::chrono::milliseconds>(endTime-initTime)).count()/1000); // convert to seconds
            std::cout << "FPS: " << fps << std::endl;
        }

        if (shutdown_request > 0)
            break;
    } // end while get frames
    
    std::cout << "Goodbye..." << std::endl;

    if ( objDet != NULL) {
        delete objDet;
        objDet = NULL;
    }
    OVMS_ServerDelete(srv);
    OVMS_ModelsSettingsDelete(modelsSettings);
    OVMS_ServerSettingsDelete(serverSettings);

    fprintf(stdout, "main() exit\n");
    return 0;
}
