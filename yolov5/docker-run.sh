#!/bin/bash

docker run -it --user root --net host --entrypoint /bin/bash -v `pwd`:/savedir --privileged ovms_capi_ocv_gst:latest
