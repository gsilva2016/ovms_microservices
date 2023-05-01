#!/bin/bash

xhost +
docker run -it --user root --net host --entrypoint /bin/bash -v `pwd`:/savedir  -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --privileged ovms_capi_ocv_gst-people:latest
