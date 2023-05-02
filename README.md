# OVMS Microservices


Ref. OVCM C-API Blog: https://blog.openvino.ai/blog-posts/model-server-c-api

Ref. OVMS C-API: https://docs.openvino.ai/latest/ovms_demo_capi_inference_demo.html

Ref. OVMS C-API Limitations: https://docs.openvino.ai/latest/ovms_docs_c_api.html#preview-limitations

Phase 1
- Add instrumentation to measure total time taken per frame (latency) and calc overall FPS througput - DONE
- Compare this yolov5s workload througput vs. Native OpenVINO benchmark app in sandbox - DONE 
- Compare this yolov5s workload througput vs. dlstreamer workload in sandbox - DONE
  - If perf is not near DLStreamer perf. then investigation is needed to resolve - N/A
- Create post-processing library and demonstrate bounding boxes around objects in the video - DONE

Phase 2
- Create MLOps demo - DONE
  - Push Yolov5 version 2 model and show increased accuracy (slower performance) and backout with no downtime - DONE

Phase 3
- Add SSD OpenVINO model https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/person-detection-retail-0013 - DONE
- Improve postprocessing accuracy - DONE
- Add live rendering support which shows inference metadata overlayed - DONE
- Support multiple object detection model types in a single OVMS C-API server - DONE


## Build Container
```
./build.sh
```

## Run Object Detection Demo on Arc discrete A770m GPU (https://simplynuc.com/serpent-canyon/) 
```
cd object_detection
```

```
./docker-run.sh
```

```
make -f MakefileCapi cpp
```
<b>For Arc GPU media decode and inferencing on the client (person who executed the buildh.sh script) downloaded MP4 Video Files :</b>

```
VIDEO_FILE=people-detection.mp4
```

or

```
VIDEO_FILE=coca-cola-4465029.mp4
```


Yolov5s (416x416)

```
/ovms/bin/capi_cpp_example "filesrc location=./$VIDEO_FILE ! qtdemux ! h264parse ! vaapidecodebin ! vaapipostproc width=416 height=416 scale-method=fast ! videoconvert ! video/x-raw ! appsink drop=1" 9178 11338 yolov5
```


Person-detection-retail-0013 SSD (544x320)

```
/ovms/bin/capi_cpp_example "filesrc location=$VIDEO_FILE ! qtdemux ! h264parse ! vaapidecodebin ! vaapipostproc width=544 height=320 scale-method=fast ! videoconvert ! video/x-raw ! appsink drop=1" 9178 11338 person-detection-retail-0013
```

<b>For Arc GPU media decode and inferencing on an RTSP stream :</b>

Yolov5s (416x416)

```
/ovms/bin/capi_cpp_example "rtspsrc location=rtsp://127.0.0.1:8554/camera_0 ! rtph264depay ! vaapidecodebin ! vaapipostproc width=416 height=416 scale-method=fast ! videoconvert ! video/x-raw ! appsink drop=1" 9178 11338 yolov5
```

Person-detection-retail-0013 SSD (544x320)

```
/ovms/bin/capi_cpp_example "rtspsrc location=rtsp://127.0.0.1:8554/camera_0 ! rtph264depay ! vaapidecodebin ! vaapipostproc width=544 height=320 scale-method=fast ! videoconvert ! video/x-raw ! appsink drop=1" 9178 11338 person-detection-retail-0013
```


** NOTE: For performing inference on CPU modify the config_object_detection.json and update the CLI text above to not use vaapidecodebin and vaapipostproc GST elements.


## Run Grafana GPU Metrics Dashboard on Arc discrete A770m GPU and Intel 12th Gen Core integrated GPU (https://simplynuc.com/serpent-canyon/) 

Refer to https://github.com/gsilva2016/docker-intel-gpu-telegraf/tree/master
