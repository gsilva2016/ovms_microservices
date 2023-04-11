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


## Build Container
./build.sh


## Run Yolov5s Demo on Arc discrete A770m GPU (https://simplynuc.com/serpent-canyon/) 

`cd yolov5`

`./docker-run.sh`

For Arc GPU media decode and inferencing on the client (person who executed the buildh.sh script) downloaded Pexel MP4 Video File :

`/ovms/bin/capi_cpp_example "filesrc location=./coca-cola-4465029.mp4 ! qtdemux ! h264parse ! vaapidecodebin ! vaapipostproc width=416 height=416 scale-method=fast ! videoconvert ! video/x-raw ! appsink drop=1"`

For Arc GPU media decode and inferencing on an RTSP stream :

`/ovms/bin/capi_cpp_example "rtspsrc location=rtsp://127.0.0.1:8554/camera_0 ! rtph264depay ! vaapidecodebin ! vaapipostproc width=416 height=416 scale-method=fast ! videoconvert ! video/x-raw ! appsink drop=1"`


** NOTE: For running on CPU modify the config_yolo.json and update the CLI text above to not use vaapidecodebin and vaapipostproc GST elements.
