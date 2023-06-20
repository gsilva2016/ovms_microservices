#!/bin/bash

xhost +
#docker run -it --user root --net host --entrypoint /bin/bash -v `pwd`:/savedir  -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --privileged ovms_capi_ocv_gst-people:latest
docker run -it --user root --net host --entrypoint /bin/bash -v `pwd`:/savedir -v `pwd`/configs:/ovms/demos/configs -v `pwd`/model_repo:/ovms/demos/model_repo  -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --privileged ovms_capi_ocv_gst-people:latest

