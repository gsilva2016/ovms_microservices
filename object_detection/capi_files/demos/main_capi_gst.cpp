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
#include <regex>
#include <atomic>
#include <mutex>
#include <condition_variable>

#include <signal.h>
#include <stdio.h>

// Utilized for GStramer hardware accelerated decode and pre-preprocessing
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>

// Utilized for OpenCV based Rendering only
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

// Utilized for infernece output layer post-processing
#include <cmath>

#include "ovms.h"  // NOLINT

//using namespace std;
//using namespace cv;

std::mutex _mtx;
std::mutex _infMtx;
std::mutex _drawingMtx;
std::condition_variable _cvAllDecodersInitd;
bool _allDecodersInitd = false;

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

class MediaPipelineServiceInterface {
public:
    enum VIDEO_TYPE {
        H264,
        H265
    };

    virtual ~MediaPipelineServiceInterface() {}
    virtual const std::string getVideoDecodedPreProcessedPipeline(std::string mediaLocation, VIDEO_TYPE videoType, int video_width, int video_height) = 0;
    virtual const std::string getBroadcastPipeline() = 0;
    virtual const std::string getRecordingPipeline() = 0;

    const std::string updateVideoDecodedPreProcessedPipeline(int video_width, int video_height)
    {
        return getVideoDecodedPreProcessedPipeline(m_mediaLocation, m_videoType, video_width, video_height);
    }

protected:
    std::string m_mediaLocation;
    VIDEO_TYPE m_videoType;
    int m_videoWidth;
    int m_videoHeight;
};

OVMS_Server* _srv;
int _server_grpc_port;
int _server_http_port;

/* OpenCV media decode */
std::atomic<int> _mediaPipelinePaused;
std::string _videoStreamPipeline;
MediaPipelineServiceInterface::VIDEO_TYPE _videoType = MediaPipelineServiceInterface::VIDEO_TYPE::H264;
bool _detectorModel = 0;
bool _includeOCR = 0;
bool _render = 0;
cv::Mat _presentationImg;
int _video_input_width = 0;  // Get from media _img
int _video_input_height = 0; // Get from media _img
std::vector<cv::VideoCapture> _vidcaps;
int _window_width = 1280;
int _window_height = 720;

class GStreamerMediaPipelineService : public MediaPipelineServiceInterface {
public:
    const std::string getVideoDecodedPreProcessedPipeline(std::string mediaLocation, VIDEO_TYPE videoType, int video_width, int video_height) {
        m_mediaLocation = mediaLocation;
        m_videoType = videoType;
        m_videoWidth = video_width;
        m_videoHeight = video_height;

        if (mediaLocation.find("rtsp") != std::string::npos ) {
        // video/x-raw(memory:VASurface),format=NV12
            switch (videoType)
            {
                case H264:
//                    return "rtspsrc location=" + mediaLocation + " ! rtph264depay ! vaapidecodebin ! video/x-raw(memory:VASurface),format=BGRA ! appsink drop=1 sync=0";
                    
                    return "rtspsrc location=" + mediaLocation + " ! rtph264depay ! vaapidecodebin ! video/x-raw(memory:VASurface),format=NV12 ! vaapipostproc" +
                    " width=" + std::to_string(video_width) +
                    " height=" + std::to_string(video_height) +
                    " scale-method=fast ! videoconvert ! video/x-raw,format=BGR ! queue ! appsink drop=1 sync=0";
                case H265:
                    return "rtspsrc location=" + mediaLocation + " ! rtph265depay ! vaapidecodebin ! vaapipostproc" +
                    " width=" + std::to_string(video_width) +
                    " height=" + std::to_string(video_height) +
                    " scale-method=fast ! videoconvert ! video/x-raw,format=BGR ! appsink sync=0 drop=1";
                default:
                    std::cout << "Video type not supported!" << std::endl;
                    return "";
            }
        }
        else if (mediaLocation.find(".mp4") != std::string::npos ) {
            switch (videoType)
            {
                case H264:
                    return "filesrc location=" + mediaLocation + " ! qtdemux ! h264parse ! vaapidecodebin ! vaapipostproc" +
                    " width=" + std::to_string(video_width) +
                    " height=" + std::to_string(video_height) +
                    " scale-method=fast ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1";
                case H265:
                    return "filesrc location=" + mediaLocation + " ! qtdemux ! h265parse ! vaapidecodebin ! vaapipostproc" +
                    " width=" + std::to_string(video_width) +
                    " height=" + std::to_string(video_height) +
                    " scale-method=fast ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1";
                default:
                    std::cout << "Video type not supported!" << std::endl;
                    return "";
            }
        }
        else {
            std::cout << "Unknown media source specified " << mediaLocation << " !!" << std::endl;
            return "";
        }
    }

    const std::string getBroadcastPipeline() {
        // TODO: Not implemented
        return "videotestsrc ! videoconvert,format=BGR ! video/x-raw ! appsink drop=1";
    }

    const std::string getRecordingPipeline() {
        // TODO: Not implemented
        return "videotestsrc ! videoconvert,format=BGR ! video/x-raw ! appsink drop=1";
    }
protected:

};

class ObjectDetectionInterface {
public:
    const static size_t MODEL_DIM_COUNT = 4;
    uint64_t model_input_shape[MODEL_DIM_COUNT] = { 0 };

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
        std::vector<int> vmodel_input_shape = getModelInputShape();
        std::copy(vmodel_input_shape.begin(), vmodel_input_shape.end(), model_input_shape);

        //std::cout << "Using object detection type person-detection-retail-0013" << std::endl;
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
        return MODEL_DIM_COUNT;
    }

    const std::vector<int> getModelInputShape() {
        std::vector<int> shape{1, 320, 544, 3};
        return shape;
    }

    const std::string getClassLabelText(int classIndex) {
        return (classIndex == 1 ? "Person" : "Unknown");
    }

    /*
    * Reference: SSD
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
        const float* outData = reinterpret_cast<const float*>(voutputData);
        std::vector<int> input_shape = getModelInputShape();
        float network_h = (float) input_shape[1];
        float network_w = (float) input_shape[2];
        //printf("Network %f %f numDets %d \n", network_h, network_w, numberOfDetections);

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
    const char* MODEL_NAME = "people-detection-retail-0013";
    const uint64_t MODEL_VERSION = 0;
    const char* INPUT_NAME = "data";
};

class Yolov5 : public ObjectDetectionInterface
{
public:

    Yolov5()
    {
        confidence_threshold = .5;
        classes = 80;
        std::vector<int> vmodel_input_shape = getModelInputShape();
        std::copy(vmodel_input_shape.begin(), vmodel_input_shape.end(), model_input_shape);

        //std::cout << "Using object detection type Yolov5" << std::endl;
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
        return MODEL_DIM_COUNT;
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

    // Yolov5
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

class TextDetection : public ObjectDetectionInterface {
public:

    TextDetection() {
        confidence_threshold = .2;
        classes = 1;
        std::vector<int> vmodel_input_shape = getModelInputShape();
        std::copy(vmodel_input_shape.begin(), vmodel_input_shape.end(), model_input_shape);

        //std::cout << "Using object detection type text-detection-00012" << std::endl;
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
        return MODEL_DIM_COUNT;
    }

    const std::vector<int> getModelInputShape() {
        std::vector<int> shape{1, 704, 704, 3};
        return shape;
    }

    const std::string getClassLabelText(int classIndex) {
        return "text";
    }

    /*
    * Reference: https://github.com/openvinotoolkit/model_server/blob/4d4c067baec66f01b1f17795406dd01e18d8cf6a/demos/horizontal_text_detection/python/horizontal_text_detection.py
    * TODO: Move a shared lib.
    */
    void postprocess(const uint64_t* output_shape, const void* voutputData, const size_t bytesize, const uint32_t dimCount, std::vector<DetectedResult> &detectedResults)
    {
        if (!voutputData || !output_shape) {
            // nothing to do
            return;
        }
        // boxes shape - N,5 or 100,5
        const int numberOfDetections = output_shape[1];
        const int objectSize = output_shape[2];
        const float* outData = reinterpret_cast<const float*>(voutputData);
        std::vector<int> input_shape = getModelInputShape();
        float network_h = (float) input_shape[1];
        float network_w = (float) input_shape[2];
        float scaleW = 1.0;
        float scaleH = 1.0;

        if (_render) {
            scaleW = (float)_window_width / network_w;
            scaleH = (float)_window_height / network_w;
        }

        //printf("----->Network %f %f numDets %d objsize %d \n", network_h, network_w, numberOfDetections, objectSize);

        for (int i = 0; i < numberOfDetections; i++)
        {
            float confidence = outData[i * objectSize + 4];
            //printf("------>text conf: %f\n", outData[i * objectSize + 4]);

            if (confidence > confidence_threshold ) {
                DetectedResult obj;
                obj.x = outData[i * objectSize + 0] * scaleW;
                obj.y = outData[i * objectSize + 1] * scaleH;
                // Yolo/SSD is not bottom-left/bottom-right so make consistent by subtracking
                obj.width = outData[i * objectSize + 2] * scaleW - obj.x;
                obj.height = outData[i * objectSize + 3] * scaleH - obj.y;
                obj.confidence = confidence;
                obj.classId = 0; // only text can be detected
                obj.classText = getClassLabelText(obj.classId).c_str();
                //printf("Adding obj %f %f %f %f with label %s\n",obj.x, obj.y, obj.width, obj.height, obj.classText);
                detectedResults.push_back(obj);

            } // end if confidence
        } // end for
    } // End of Text-Det Post-Processing


private:
    /* Model Serving Info :
      https://github.com/dlstreamer/pipeline-zoo-models/blob/main/storage/horizontal-text-detection-0002/
      https://github.com/openvinotoolkit/model_server/blob/4d4c067baec66f01b1f17795406dd01e18d8cf6a/demos/horizontal_text_detection/python/horizontal_text_detection.py
    */
    const char* MODEL_NAME = "text-detect-0002";
    const uint64_t MODEL_VERSION = 0;
    const char* INPUT_NAME = "input";
};

GStreamerMediaPipelineService* _mediaService = NULL;
std::string _user_request;

namespace {
volatile sig_atomic_t shutdown_request = 0;
}

bool stringIsInteger(std::string strInput) {
    std::string::const_iterator it = strInput.begin();
    while (it != strInput.end() && std::isdigit(*it)) ++it;
    return !strInput.empty() && it == strInput.end();
}


bool openVideo(std::string videoPipeline, cv::VideoCapture& vidcap)
{

    const char* videoFileName = videoPipeline.c_str();
    
    // Default to GSTREAMER for media decode
    int retries = 5;
    for (int i = 0; i < retries; i++)
    {
        if (vidcap.open(videoFileName, cv::CAP_GSTREAMER))
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    //printf("OV 2\n");

    if (!vidcap.isOpened()) {
        printf("Unable to open the video stream %s\n", videoFileName);
    } else {
        _video_input_width  = vidcap.get(cv::CAP_PROP_FRAME_WIDTH);
        _video_input_height = vidcap.get(cv::CAP_PROP_FRAME_HEIGHT);
    }
    //printf("OV 3\n");

	return vidcap.isOpened();
}

void closeVideo(cv::VideoCapture& vidcap)
{
    int retries = 5;
    for (int i = 0; i < retries; i++)
    {
        if (!vidcap.isOpened())
            break;
        else
            vidcap.release();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

bool getVideoFrame(cv::VideoCapture& vidcap, cv::Mat &img)
{
    while (_mediaPipelinePaused)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

	int ret = vidcap.read(img);

	if (!ret)
    {
        std::cout << "Warning: Failed to read video frame!" << std::endl;

        //retry one time...
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ret = vidcap.read(img);
    }

	return ret;
}

bool setActiveModel(int detectionType, ObjectDetectionInterface** objDet)
{
    if (detectionType != 0 && detectionType != 1 ) {
        std::cout << "ERROR: detectionType option must be 0 (yolov5) or 1 (people-detection-retail-0013) or 3 (yolox-complex)" << std::endl;
        return false;
    }

    if (objDet == NULL)
        return false;

    if (detectionType == 0) {
        *objDet = new Yolov5();
    }
    else if(detectionType == 1) {
        *objDet = new SSD();
    }
    else if(detectionType == 3) {
        //TODO:
        std::cout << "TODO" << std::endl;
        return false;
    }
    return true;
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

// TODO: Multiple references state that imshow can't be used in any other thread than main!
void displayGUIInferenceResults(cv::Mat analytics_frame, std::vector<DetectedResult> &results, int latency, int througput)
{
    auto ttid = std::this_thread::get_id();
    std::stringstream ss;
    ss << ttid;
    std::string tid = ss.str();

    for (auto & obj : results) {
	    const float x0 = obj.x; 
        const float y0 = obj.y; 
        const float x1 = obj.x + obj.width; 
        const float y1 = obj.y + obj.height; 

        //printf("--------->coords: %f %f %f %f\n", x0, y0, x1, y1);
        cv::rectangle( analytics_frame,
            cv::Point( (int)(x0),(int)(y0) ),
            cv::Point( (int)x1, (int)y1 ),
            cv::Scalar(255, 0, 0),
            2, cv::LINE_8 );
    } // end for

    //latency
    std::string fps_msg = (througput == 0) ? "..." : std::to_string(througput) + "fps";
    std::string latency_msg = (latency == 0) ? "..." :  std::to_string(latency) + "ms";
    std::string roiCount_msg = std::to_string(results.size());
    std::string message = "E2E Pipeline Performance: " + latency_msg + " and " + fps_msg + " with ROIs#" + roiCount_msg;
    cv::putText(analytics_frame, message.c_str(), cv::Size(0, 20), cv::FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv::LINE_4);
    cv::putText(analytics_frame, tid, cv::Size(0, 40), cv::FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv::LINE_4);

    cv::Mat presenter;
    resize(analytics_frame, presenter, cv::Size(_window_width, _window_height), 0, 0, cv::INTER_LINEAR);

    {
    std::lock_guard<std::mutex> lock(_drawingMtx);
    cv::imshow("OpenVINO Results " + tid, presenter);
    cv::waitKey(1);
    }
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

// This function is responsible for generating a GST pipeline that
// decodes and resizes the video stream based on the desired window size or
// the largest analytics frame size needed if running headless
std::string getVideoPipelineText(std::string mediaPath, ObjectDetectionInterface* objDet, ObjectDetectionInterface* textDet) 
{

    std::vector<int> pipelineFrameShape = objDet->getModelInputShape();
    if (textDet) {
        pipelineFrameShape = textDet->getModelInputShape();
    }

    int frame_width = pipelineFrameShape[1];
    int frame_height = pipelineFrameShape[2];

    if (_render)
    {
        frame_width = _window_width;
        frame_height = _window_height;
    }

    return _mediaService->getVideoDecodedPreProcessedPipeline(
        mediaPath,
        _videoType,
        frame_width,
        frame_height);
}

void performTextDetectionInference(OVMS_Server* srv, std::vector<DetectedResult> &detectedResults, cv::Mat& img, ObjectDetectionInterface* textDet)
{
    OVMS_InferenceRequest* request{nullptr};
    {
        std::lock_guard<std::mutex> lock(_infMtx);
        OVMS_InferenceRequestNew(&request, srv, textDet->getModelName(), textDet->getModelVersion());
    }

    OVMS_InferenceRequestAddInput(
        request,
        textDet->getModelInputName(),
        OVMS_DATATYPE_FP32,
        textDet->model_input_shape,
        textDet->getModelDimCount()
    );

    // TODO: Which option is the best? Using option #3
    // 1. traverse all rois and scale as needed
    // 2. use the original detection image  and scale as needed
    // 3. use  the original video stream with no resizing. However,  detection will need to resize since 
    //    it expects a smaller frame than OCR
    
    // Since OCR requires the largest video shape in the pipeline there is no need to resize unless we are 
    // rendering. This is due to the fact that rendering is poor when using a video shape based on the OCR network input shape.

    cv::Mat floatImage;
    if (_render)
    {
        cv::Mat analytics_frame;
        std::vector<int> inputShape = textDet->getModelInputShape();
        resize(img, analytics_frame, cv::Size(inputShape[1], inputShape[2]), 0, 0, cv::INTER_CUBIC);
        analytics_frame.convertTo(floatImage, CV_32F);
    }
    else // path is optimized when not having to render for demo purposes
        img.convertTo(floatImage, CV_32F);

    const int DATA_SIZE = floatImage.step[0] * floatImage.rows;

    OVMS_InferenceRequestInputSetData(
        request,
        textDet->getModelInputName(),
        reinterpret_cast<void*>(floatImage.data),
        DATA_SIZE ,
        OVMS_BUFFERTYPE_CPU,
        0
    );

    // run sync request
    OVMS_Status* res = nullptr;
    OVMS_InferenceResponse* response = nullptr;
    {
        std::lock_guard<std::mutex> lock(_infMtx);
        res = OVMS_Inference(srv, request, &response);
    }

    if (res != nullptr) {
        std::cout << "OVMS_Inference failed " << std::endl;
        uint32_t code = 0;
        const char* details = 0;
        OVMS_StatusGetCode(res, &code);
        OVMS_StatusGetDetails(res, &details);
        std::cout << "Error occured during inference. Code:" << code
                << ", details:" << details << std::endl;
        return;
    }

    uint32_t outputCount = 0;
    OVMS_InferenceResponseGetOutputCount(response, &outputCount);
    //printf("TextDet got %d model outputs\n", outputCount);

    const void* voutputData1;
    size_t bytesize1 = 0;
    uint32_t outputId = 0; // get detections only
    OVMS_DataType datatype1 = (OVMS_DataType)42;
    const uint64_t* shape1{nullptr};
    uint32_t dimCount1 = 0;
    OVMS_BufferType bufferType1 = (OVMS_BufferType)42;
    uint32_t deviceId1 = 42;
    const char* outputName1{nullptr};

    OVMS_InferenceResponseGetOutput(response, outputId, &outputName1, &datatype1, &shape1, &dimCount1, &voutputData1, &bytesize1, &bufferType1, &deviceId1);
    //std::cout << "TextDet------------> " << outputName1  << " " << dimCount1 << " " << shape1[0] << " " << shape1[1] << " " << shape1[2] << " " << shape1[3] << " " << bytesize1 << " " << outputCount << std::endl;

    std::vector<DetectedResult> detectedTextResults;
    textDet->postprocess(shape1, voutputData1, bytesize1, dimCount1, detectedTextResults);
    textDet->postprocess(detectedTextResults, detectedResults);

}

bool createModelServer(OVMS_Server* srv, OVMS_ServerSettings* serverSettings, OVMS_ModelsSettings* modelsSettings) 
{

    if (srv == NULL)
        return false;

    OVMS_Status* res = OVMS_ServerStartFromConfigurationFile(srv, serverSettings, modelsSettings);

    if (res) {
        uint32_t code = 0;
        const char* details = nullptr;

        OVMS_StatusGetCode(res, &code);
        OVMS_StatusGetDetails(res, &details);
        std::cerr << "ERROR: during start: code:" << code << "; details:" << details
                  << "; grpc_port: " << _server_grpc_port 
                  << "; http_port: " << _server_http_port 
                  << ";" << std::endl;

        OVMS_StatusDelete(res);

        OVMS_ServerDelete(srv);
        OVMS_ModelsSettingsDelete(modelsSettings);
        OVMS_ServerSettingsDelete(serverSettings);
        return false;
    }    

    return true;
}

// Bus messages processing, similar to all gstreamer examples
gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
	GMainLoop *loop = (GMainLoop *) data;

	switch (GST_MESSAGE_TYPE (msg))
	{

	case GST_MESSAGE_EOS:
		fprintf(stderr, "End of stream\n");
		g_main_loop_quit(loop);
		break;

	case GST_MESSAGE_ERROR:
	{
		gchar *debug;
		GError *error;

		gst_message_parse_error(msg, &error, &debug);
		g_free(debug);

		g_printerr("Error: %s\n", error->message);
		g_error_free(error);

		g_main_loop_quit(loop);

		break;
	}
	default:
		break;
	}

	return TRUE;
}


bool loadGStreamer(GstElement** pipeline,  GstElement** appsink, std::string mediaPath, ObjectDetectionInterface* _objDet, ObjectDetectionInterface* _textDet) 
{    
    static int threadCnt = 0;
    
    std::string videoPipelineText = getVideoPipelineText(mediaPath, _objDet, ((_includeOCR) ? _textDet : NULL));
    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "Opening Media Pipeline: " << videoPipelineText << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;

    *pipeline = gst_parse_launch (videoPipelineText.c_str(), NULL);
    if (*pipeline == NULL) {
        std::cout << "ERROR: Failed to parse GST pipeline. Quitting." << std::endl;
        return false;
    }

    std::string appsinkName = "appsink" + std::to_string(threadCnt++);

    *appsink = gst_bin_get_by_name (GST_BIN (*pipeline), appsinkName.c_str());

    // Check if all elements were created
    if (!(*appsink))
    {
        printf("ERROR: Failed to initialize GST pipeline (missing %s) Quitting.\n", appsinkName.c_str());
        return false;
    }

    GstStateChangeReturn gst_res;

    // Start pipeline so it could process incoming data
    gst_res = gst_element_set_state(*pipeline, GST_STATE_PLAYING);
    
    if (gst_res != GST_STATE_CHANGE_SUCCESS && gst_res != GST_STATE_CHANGE_ASYNC  ) {
        printf("ERROR: StateChange not successful. Error Code: %d\n", gst_res);
        return false;
    }
    
    return true;
}

// OVMS C-API is a global process (singleton design) wide server so can't create many of them
bool loadOVMS(OVMS_Server* srv) 
{
     OVMS_Status* res = NULL;
     
     OVMS_ServerSettings* serverSettings = 0;
     OVMS_ModelsSettings* modelsSettings = 0;
    
     OVMS_ServerSettingsNew(&serverSettings);
     OVMS_ModelsSettingsNew(&modelsSettings);
     OVMS_ServerNew(&_srv);
     OVMS_ServerSettingsSetGrpcPort(serverSettings, _server_grpc_port);
     OVMS_ServerSettingsSetRestPort(serverSettings, _server_http_port);
     OVMS_ServerSettingsSetLogLevel(serverSettings, OVMS_LOG_ERROR);

     // // Loads models in single OVMS C-API server
     OVMS_ModelsSettingsSetConfigPath(modelsSettings, "configs/config_object_detection.json");

     if (!createModelServer(_srv, serverSettings, modelsSettings)) {
         std::cout << "Failed to create model server\n" << std::endl;
         return false;
     }
     else {
         std::cout << "--------------------------------------------------------------" << std::endl;
         std::cout << "Server ready for inference C-API ports " << _server_grpc_port << " " << _server_http_port << std::endl;
         std::cout << "--------------------------------------------------------------" << std::endl;
         _server_http_port+=1;
         _server_grpc_port+=1;
     }
     return true;
}

bool getMAPipeline(std::string mediaPath, GstElement** pipeline,  GstElement** appsink, ObjectDetectionInterface** _objDet, ObjectDetectionInterface** _textDet) 
{
    
    *_textDet = new TextDetection();
    
    if (!setActiveModel(_detectorModel, _objDet)) {
        std::cout << "Unable to set active detection model" << std::endl;
        return false;
    }

    return loadGStreamer(pipeline, appsink, mediaPath, *_objDet, *_textDet);
}

void run_stream(std::string mediaPath, GstElement* pipeline, GstElement* appsink, ObjectDetectionInterface* objDet, ObjectDetectionInterface* textDet)
{

    auto ttid = std::this_thread::get_id();
    std::stringstream ss;
    ss << ttid;
    std::string tid = ss.str();

    // Wait for all decoder streams to init...otherwise causes a segfault when OVMS loads
    // https://stackoverflow.com/questions/48271230/using-condition-variablenotify-all-to-notify-multiple-threads
    std::unique_lock<std::mutex> lk(_mtx);
    _cvAllDecodersInitd.wait(lk, [] { return _allDecodersInitd;} ); 
    lk.unlock();

    printf("Starting thread: %s\n", tid.c_str()) ;

    auto initTime = std::chrono::high_resolution_clock::now();
    unsigned long numberOfFrames = 0;
    long long numberOfSkipFrames = 0;
    OVMS_Status* res = NULL;
    
    while (!shutdown_request) {
        auto startTime = std::chrono::high_resolution_clock::now();        
        
        const void* voutputData3;
        const void* voutputData2;
        const void* voutputData1;
        size_t bytesize1 = 0;
        size_t bytesize2 = 0;
        size_t bytesize3 = 0;
        uint32_t outputCount = 0;
        uint32_t outputId;
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

        GstSample *sample;
        GstStructure *s;
        GstBuffer *buffer;
        GstMapInfo m;
      
        std::vector<DetectedResult> detectedResults;
        std::vector<DetectedResult> detectedResultsFiltered;        

        if (gst_app_sink_is_eos(GST_APP_SINK(appsink))) {
            std::cout << "INFO: EOS " << std::endl;
            return;
        }

        sample = gst_app_sink_try_pull_sample (GST_APP_SINK(appsink), 1 * GST_SECOND);

        if (sample == nullptr) {
            std::cout << "ERROR: No sample found" << std::endl;
            return;
        }

        GstCaps *caps;
        caps = gst_sample_get_caps(sample);

        if (caps == nullptr) {
            std::cout << "ERROR: No caps found for sample" << std::endl;
            return;
        }

        s = gst_caps_get_structure(caps, 0);
        gst_structure_get_int(s, "width", &_video_input_width);
        gst_structure_get_int(s, "height", &_video_input_height);

        buffer = gst_sample_get_buffer(sample);
        gst_buffer_map(buffer, &m, GST_MAP_READ); 

        if (m.size <= 0) {
            std::cout << "ERROR: Invalid buffer size" << std::endl;
            return;
        }
        
        // Final FP32 image for inference
        cv::Mat analytics_frame;
        cv::Mat floatImage;
        std::vector<int> inputShape;

        inputShape = objDet->getModelInputShape();

        cv::Mat img(_video_input_height, _video_input_width, CV_8UC3, (void *) m.data);
        
        // Resize not needed if not rendering and no OCR is used. This is true given the frame will already be scaled
        // to the detector input shape for optimization purposes.
        if (_render || _includeOCR) {
            resize(img, analytics_frame, cv::Size(inputShape[1], inputShape[2]), 0, 0, cv::INTER_LINEAR);
            analytics_frame.convertTo(floatImage, CV_32F);
        }
        else {
            img.convertTo(floatImage, CV_32F);
        }
        
        const int DATA_SIZE = floatImage.step[0] * floatImage.rows;       

	    OVMS_InferenceResponse* response = nullptr;
        OVMS_InferenceRequest* request{nullptr};

    // Inference        
    {
        std::lock_guard<std::mutex> lock(_infMtx);

        OVMS_InferenceRequestNew(&request, _srv, objDet->getModelName(), objDet->getModelVersion());

        OVMS_InferenceRequestAddInput(
             request,
             objDet->getModelInputName(),
             OVMS_DATATYPE_FP32,
             objDet->model_input_shape,
             objDet->getModelDimCount()
        );
        
        // run sync request
        OVMS_InferenceRequestInputSetData(
            request,
            objDet->getModelInputName(),
            reinterpret_cast<void*>(floatImage.data),
            DATA_SIZE ,
            OVMS_BUFFERTYPE_CPU,
            0
        );

        res = OVMS_Inference(_srv, request, &response);

        if (res != nullptr) {
            std::cout << "OVMS_Inference failed " << std::endl;
            uint32_t code = 0;
            const char* details = 0;
            OVMS_StatusGetCode(res, &code);
            OVMS_StatusGetDetails(res, &details);
            std::cout << "Error occured during inference. Code:" << code
                    << ", details:" << details << std::endl;
            if (request)
                OVMS_InferenceRequestDelete(request);
            break;
        }
        else
        {
            OVMS_StatusDelete(res);
            printf("INference successful\n");
        }
    } // end lock on inference request to server

        // Read output. Yolov5s has 3 outputs...only processing one.
        // TODO: process remaining        
        OVMS_InferenceResponseGetOutputCount(response, &outputCount);
        outputId = outputCount - 1;

        OVMS_InferenceResponseGetOutput(response, outputId, &outputName1, &datatype1, &shape1, &dimCount1, &voutputData1, &bytesize1, &bufferType1, &deviceId1);

        // std::cout << "------------>" << tid << " : " << "DeviceID " << deviceId1
        //  << ", OutputName " << outputName1
        //  << ", DimCount " << dimCount1
        //  << ", shape " << shape1[0] << " " << shape1[1] << " " << shape1[2] << " " << shape1[3]
        //  << ", byteSize " << bytesize1
        //  << ", OutputCount " << outputCount << std::endl;

        //OVMS_InferenceResponseGetOutput(response, outputId - 1, &outputName2, &datatype2, &shape2, &dimCount2, &voutputData2, &bytesize2, &bufferType2, &deviceId2);
        //std::cout << "------------> " << outputName2  << " " << dimCount2 << " " << shape2[0] << " " << shape2[1] << " " << shape2[2] << " " << shape2[3] << " " << bytesize2 << " " << outputCount << std::endl;

        //OVMS_InferenceResponseGetOutput(response, outputId-2, &outputName3, &datatype3, &shape3, &dimCount3, &voutputData3, &bytesize3, &bufferType3, &deviceId3);
        //std::cout << "------------> " << outputName3  << " " << dimCount3 << " " << shape3[0] << " " << shape3[1] << " " << shape3[2] << " " << shape3[3] << " " << bytesize3 << " " << outputCount << std::endl;

        objDet->postprocess(shape1, voutputData1, bytesize1, dimCount1, detectedResults);
        objDet->postprocess(detectedResults, detectedResultsFiltered);

        // perform text detection as an option
        if (_includeOCR)
        {        
            performTextDetectionInference(_srv, detectedResultsFiltered, img, textDet);
        }

        numberOfSkipFrames++;
        float fps = 0;
        if (numberOfSkipFrames <= -1) // skip warm up
        {
            initTime = std::chrono::high_resolution_clock::now();
            numberOfFrames = 1;

            if (response) {
              OVMS_InferenceResponseDelete(response);
            }
        
            if (request) {
              OVMS_InferenceRequestInputRemoveData(request, objDet->getModelInputName()); // doesn't help
              OVMS_InferenceRequestRemoveInput(request, objDet->getModelInputName());
              OVMS_InferenceRequestDelete(request);
            }

            //printf("Too early...Skipping frames..\n");
            continue;
        }
        else if (numberOfSkipFrames > 120)
            numberOfFrames++;

        auto endTime = std::chrono::high_resolution_clock::now();
        auto latencyTime = ((std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime)).count());
        auto runningLatencyTime = ((std::chrono::duration_cast<std::chrono::milliseconds>(endTime-initTime)).count());
        if (runningLatencyTime > 0) { // skip a few to account for init
            fps = (float)numberOfFrames/(float)(runningLatencyTime/1000); // convert to seconds
        }

        //printInferenceResults(detectedResultsFiltered);
        if (_render) {
            displayGUIInferenceResults(img, detectedResultsFiltered, latencyTime, fps);
            printf("quitting %s\n", tid.c_str());
            //return;
        }
        else
        {
            std::cout << "Pipeline Throughput FPS: " << fps << std::endl;
            std::cout << "Pipeline Latency (ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() << std::endl;
        }
        //saveInferenceResultsAsVideo(img, detectedResultsFiltered);

        
        if (request) {
           OVMS_InferenceRequestInputRemoveData(request, objDet->getModelInputName()); // doesn't help
           OVMS_InferenceRequestRemoveInput(request, objDet->getModelInputName());
           OVMS_InferenceRequestDelete(request);           
        }
        
        if (response) {
           OVMS_InferenceResponseDelete(response);
        }
        
        gst_buffer_unmap(buffer, &m);
        gst_sample_unref(sample);

        if (shutdown_request > 0)
            break;
    } // end while get frames    

    std::cout << "Goodbye..." << std::endl;

  //TODO make global  
//    OVMS_ModelsSettingsDelete(modelsSettings);
//    OVMS_ServerSettingsDelete(serverSettings);

       
    if (res != NULL) {
        OVMS_StatusDelete(res);
        res = NULL;
    }

    if (objDet) {
        delete objDet;
        objDet = NULL;
    }

    if (textDet) {
        delete textDet;
        textDet = NULL;
    }

    //gst_object_unref(bus);
//    if (loop)
//        g_main_loop_unref(loop);
    gst_element_set_state (pipeline, GST_STATE_NULL);
    if (pipeline)
        gst_object_unref(pipeline);
//    if (context)
//        g_main_context_unref (context);
}

void print_usage(const char* programName) {
    std::cout << "Usage: ./" << programName << " mediaLocation video_type detector_model include_ocr render\n\n"
        << "Required: mediaLocation is an rtsp://127.0.0.1:8554/camera_0 url or a path to an *.mp4 file\n"
        << "Required: video_type is 0 for AVC or 1 for HEVC\n"
        << "Required: 0 for yolov5 or 1 for person-detection-retail-0013\n"
        << "Optional: include_ocr is 1 to include OCR or 0 (default) to disable it\n"
        << "Optional: render is 1 to launch render window or 0 (default) for headless\n";
}


int main(int argc, char** argv) {
    std::cout << std::setprecision(2) << std::fixed;

    // Use GST pipelines with OpenCV for media HWA decode and pre-procesing
    _mediaService = new GStreamerMediaPipelineService();

    _server_grpc_port = 9178;
    _server_http_port = 11338;

    _videoStreamPipeline = "people-detection.mp4";

    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    if (!stringIsInteger(argv[2]) || !stringIsInteger(argv[3]) ) {
        print_usage(argv[0]);
        return 1;
    } else {
        _videoStreamPipeline = argv[1];
        _videoType = (MediaPipelineServiceInterface::VIDEO_TYPE) std::stoi(argv[2]);
        _detectorModel = std::stoi(argv[3]);
    }

    if (stringIsInteger(argv[4]))
        _includeOCR = std::stoi(argv[4]);
    if (stringIsInteger(argv[5]))
        _render = std::stoi(argv[5]);

    if (argc > 6) {
	      _server_grpc_port = std::stoi(argv[6]);
	      _server_http_port = std::stoi(argv[7]);
    }

    // fails if init fails
    gst_init(NULL, NULL);
    
    std::vector<std::thread> running_streams;
    _allDecodersInitd = false;

    // TODO: Refactor to load based on config file
    GstElement *pipeline;
    GstElement *appsink;
    ObjectDetectionInterface* objDet;
    ObjectDetectionInterface* textDet;    
    getMAPipeline(_videoStreamPipeline, &pipeline,  &appsink, &objDet, &textDet);
    running_streams.emplace_back(run_stream, _videoStreamPipeline, pipeline, appsink, objDet, textDet);
    
    GstElement *pipeline2;
    GstElement *appsink2;
    ObjectDetectionInterface* objDet2;
    ObjectDetectionInterface* textDet2;
    _videoStreamPipeline = "rtsp://127.0.0.1:8554/camera_2";    
    getMAPipeline(_videoStreamPipeline, &pipeline2,  &appsink2, &objDet2, &textDet2);
    running_streams.emplace_back(run_stream, _videoStreamPipeline, pipeline2, appsink2, objDet2, textDet2);

    if (!loadOVMS(_srv))
        return -1;

    _allDecodersInitd = true;
    _cvAllDecodersInitd.notify_all();;


    while (!shutdown_request) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100000));

        std::cout << "TODO:::Available Commands: \n"
        << "  add_stream [Optional: mediaLocation is an rtsp://127.0.0.1:8554/camera_0 url or a path to an *.mp4 file]\n"
        << "  del_stream\n"
        << "  change_det <0=yolov5, 1=people-detection-retail-0013, 2=yolox-complex>\n"
        << "  change_video [Required: mediaLocation is an rtsp://127.0.0.1:8554/camera_0 url or a path to an *.mp4 file]\n"
        << "  toggle_ocr \n"
        << "  toggle_render \n"
        << "  r      - (refresh prompt)\n"
        << "  ctrl+c - (quit/exit)\n"
        << "\n";
        std::cout << "Enter a command: ";
        std::getline(std::cin >> std::ws, _user_request);

        if (_user_request.find("add_stream") != std::string::npos) {
            std::string mediaPath = _videoStreamPipeline;
            if (_user_request.length() > 10 )
                mediaPath = _user_request.substr(11);
            //std::cout << "new stream: " << mediaPath << std::endl;
            //running_streams.emplace_back(run_stream, mediaPath,2);

            std::cout << "--------------------------------------------------------------\n";
            std::cout << "SUCESS: Added a client thread" << std::endl;
            std::cout << "--------------------------------------------------------------\n\n";
        }
        else if (_user_request.find("del_stream") != std::string::npos) {
            std::cout << "--------------------------------------------------------------\n";
            std::cout << "SUCESS: Removed a client thread" << std::endl;
            std::cout << "--------------------------------------------------------------\n\n";
        }
        else if (_user_request.find("toggle_ocr") != std::string::npos) {
            // int vidCount = 0;
            // std::vector<int> modelInputShape = _objDet->getModelInputShape();
            // update model
            // if (_textDet == NULL) {
            //     _textDet = new TextDetection();
            //     modelInputShape = _textDet->getModelInputShape();
            //     _includeOCR = true;
            // }
            // else {
            //     _includeOCR = false;
            //     delete _textDet;
            //     _textDet = NULL;
            // }
            _includeOCR = !_includeOCR;

            if (_render) {
                std::cout << "--------------------------------------------------------------\n";
                std::cout << "SUCESS: Toggled OCR" << std::endl;
                std::cout << "--------------------------------------------------------------\n\n";
                continue;
            }
            else
            {
                std::cout << "--------------------------------------------------------------\n";
                std::cout << "Not Supported" << std::endl;
                std::cout << "--------------------------------------------------------------\n\n";
                continue;
            }

            // TODO: below is not used
            // if not rendering then pipeline uses biggest image of the pipeline so needs to reload
            // this is the save on needed to perform resize() operations on the incoming frames
            // std::lock_guard<std::mutex> lock(_mtx);
            // _mediaPipelinePaused = 1;
            // vidCount = _vidcaps.size();

            // for (int i = 0; i < vidCount; i++) {
            //     cv::VideoCapture vidcap;
            //     //{
            //         // std::lock_guard<std::mutex> lock(_mtx);
            //     vidcap  = _vidcaps[i];
            //     //}
            //     // client updates - note OVMS is not changed since it manages models independently
            //     std::cout << "Stopping current video stream..." << std::endl;

            //     std::this_thread::sleep_for(std::chrono::milliseconds(100));

            //     closeVideo(vidcap);

            //     // update media-preprocessing for the new model
            //     std::cout << "Updating media pre-processing..." << std::endl;
            //     std::string videoPipeline = _mediaService->updateVideoDecodedPreProcessedPipeline(modelInputShape[1], modelInputShape[2]);


            //     std::cout << "Restarting video stream with applied media pre-processing..." << videoPipeline << std::endl;
            //     // start video feed again
            //     if (!openVideo(videoPipeline, vidcap)) {
            //         std::cout << "--------------------------------------------------------------\n";
            //         std::cout << "ERROR:Failed to toggle OCR "  << std::endl;
            //         std::cout << "--------------------------------------------------------------\n\n";
            //     }
            //     else {
            //         std::cout << "--------------------------------------------------------------\n";
            //         std::cout << "SUCESS: Toggled OCR" << std::endl;
            //         std::cout << "--------------------------------------------------------------\n\n";
            //     }


            //     _vidcaps[i] = vidcap;
            // }

            //  _mediaPipelinePaused = 0;
        }
        else if (_user_request.find("toggle_render") != std::string::npos) {
            std::cout << "--------------------------------------------------------------\n";
            std::cout << "ERROR:Failed to toggle render "  << std::endl;
            std::cout << "--------------------------------------------------------------\n\n";
        }
        else if (_user_request.find("change_det") != std::string::npos) {
            std::string chgModelReq = _user_request.substr(11);

            // new models always need new PP? So far yes...
            // int oldDetType;
            // if  (typeid(_objDet) == typeid(Yolov5))
            //     oldDetType = 0;
            // else if (typeid(_objDet) == typeid(SSD))
            //     oldDetType = 1;
            // else {
            //     std::cout << "Unknown error...Starting Model type unknown..." << std::endl;
            //     continue;
            // }

            // if (chgModelReq == "" || !stringIsInteger(chgModelReq)) {
            //     std::cout << "--------------------------------------------------------------\n";
            //     std::cout << "ERROR: A valid detection model type is required." << std::endl;
            //     std::cout << "--------------------------------------------------------------\n\n";
            //     continue;
            // }

            // int detModelType = std::stoi(chgModelReq.c_str());

            // if (detModelType != 0 && detModelType != 1) {
            //     std::cout << "--------------------------------------------------------------\n";
            //     std::cout << "ERROR: A valid detection model type is required." << std::endl;
            //     std::cout << "--------------------------------------------------------------\n\n";
            //     continue;
            // }

            // cv::VideoCapture vidcap;
            // {
            //     std::lock_guard<std::mutex> lock(_mtx);
            //     vidcap = _vidcaps[0];
            // }

            // // client updates - note OVMS is not changed since it manages models independently
            // std::cout << "Stopping current video stream..." << std::endl;
            // closeVideo(vidcap);

            // // update model
            // std::cout << "Updating active model..." << std::endl;
            // if (!setActiveModel(detModelType)) {
            //     std::cout << "--------------------------------------------------------------\n";
            //     std::cout << "ERROR: Failed to change active model to " << detModelType << std::endl;
            //     std::cout << "--------------------------------------------------------------\n\n";
            //     continue;
            // }

            // // update media-preprocessing for the new model
            // std::cout << "Updating media pre-processing..." << std::endl;
            // std::vector<int> modelInputShape = _objDet->getModelInputShape();
            // std::string videoPipeline = _mediaService->updateVideoDecodedPreProcessedPipeline(modelInputShape[1], modelInputShape[2]);
            // //updateMediaPreProcessing();

            // std::cout << "Restarting video stream with applied media pre-processing..." << std::endl;
            // // start video feed again
            // if (!openVideo(videoPipeline, vidcap)) {
            //     std::cout << "--------------------------------------------------------------\n";
            //     std::cout << "ERROR:Failed to start video/media "  << std::endl;
            //     std::cout << "--------------------------------------------------------------\n\n";
            // }
            // else {
            //     std::cout << "--------------------------------------------------------------\n";
            //     std::cout << "SUCESS: Changed active model to " << detModelType << std::endl;
            //     std::cout << "--------------------------------------------------------------\n\n";
            // }
        }
        else if (_user_request.find("change_video") != std::string::npos) {
            // std::string video_name = _user_request.substr(12);
            // if (video_name == "") {
            //     std::cout << "--------------------------------------------------------------\n";
            //     std::cout << "ERROR: file_path_to_mp4 or rtsp url is required" << std::endl;
            //     std::cout << "--------------------------------------------------------------\n\n";
            // } else {
            //     std::cout << "--------------------------------------------------------------\n";
            //     std::cout << "TODO" << std::endl;
            //     std::cout << "--------------------------------------------------------------\n\n";
            //     //std::cout << "SUCESS: Changed active video to " << video_name << std::endl;
            // }
        }
        else if (_user_request == "r")
            continue;
        else {
            std::cout << "--------------------------------------------------------------\n";
            std::cout << "Invalid server command: '" << _user_request << "'" << std::endl;
            std::cout << "--------------------------------------------------------------\n\n";
        }

        std::cout << "--------------------------------------------------------------\n\n";

    } //end while


//    for(auto& running_stream : running_streams)
//        running_stream.join();

    // if ( _objDet != NULL) {
    //     delete _objDet;
    //     _objDet = NULL;
    // }
    // if (_textDet != NULL) {
    //     delete _textDet;
    //     _textDet = NULL;
    // }

    if (_mediaService != NULL) {
        delete _mediaService;
        _mediaService = NULL;
    }

    OVMS_ServerDelete(_srv);
    //OVMS_ModelsSettingsDelete(modelsSettings);
    //OVMS_ServerSettingsDelete(serverSettings);

    fprintf(stdout, "main() exit\n");
    return 0;
}
