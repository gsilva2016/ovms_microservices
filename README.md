# OVMS Microservices


Ref OVCM C-API Blog: https://blog.openvino.ai/blog-posts/model-server-c-api

Ref OVMS C-API: https://docs.openvino.ai/latest/ovms_demo_capi_inference_demo.html

Ref OVMS C-API Limitations: https://docs.openvino.ai/latest/ovms_docs_c_api.html#preview-limitations


## TODOs

Phase 1
- Add instrumentation to measure total time taken per frame (latency) and calc overall FPS througput
- Compare this yolov5s workload througput vs. Native OpenVINO benchmark app in sandbox - DONE for inference only.
- Compare this yolov5s workload througput vs. dlstreamer workload in sandbox 
  - If perf is not near DLStreamer perf. then investigation is needed to resolve - 
- Create post-processing library and demonstrate bounding boxes around objects in the video

Phase 2
- Create MLOps demo - DONE
  - Push Yolov5 version 2 model and show increased accuracy (slower performance) and backout with no downtime - DONE



## Build (will run demo once)
./build.sh


## Make code changes and run dummy data demo
cd yolov5

make clean

make from_docker

make build_image


## Run Yolov5s CPU demo only
docker run -it --entrypoint /bin/bash --privileged openvino/model_server-capi:latest 

make -f MakefileCapi cpp


