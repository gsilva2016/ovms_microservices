#!/bin/bash

#cp /savedir/object_detection/capi_files/demos/main_capi_gst.cpp .; make -f MakefileCapi cpp

docker run -it --user root --net host --entrypoint /bin/bash -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -v `pwd`:/savedir --privileged ovms_capi_ocv_gst-people:latest
