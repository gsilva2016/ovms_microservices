#!/bin/bash

make clean
make from_docker
make build_image
docker build -t ovms_capi_ocv_gst-people:latest .
