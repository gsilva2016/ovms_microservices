#!/bin/bash

xhost +
docker run -it --user root --net host --entrypoint /bin/bash -v /tmp/.X12-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -v `pwd`:/savedir --privileged docker.io/openvino/model_server-capi:latest

