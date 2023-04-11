# Build an OVMS CAPI image with dGPU media and inference support
FROM openvino/model_server-capi:latest

USER root

# Install GPU drivers for Flex/Arc
RUN apt update; apt install -y wget curl vim gpg-agent; wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg; echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu focal-devel main' | tee  /etc/apt/sources.list.d/intel.gpu.focal.list; apt update; 

RUN apt-get install -y \
  intel-opencl-icd \
  intel-level-zero-gpu level-zero \
  intel-media-va-driver-non-free libmfx1 libmfxgen1 libvpl2 \
  libegl-mesa0 libegl1-mesa libegl1-mesa-dev libgbm1 libgl1-mesa-dev libgl1-mesa-dri \
  libglapi-mesa libgles2-mesa-dev libglx-mesa0 libigdgmm11 libxatracker2 mesa-va-drivers \
  mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all vainfo

# Build GST with GPU support
RUN apt -y update; DEBIAN_FRONTEND=noninteractive apt install -y cmake build-essential git-gui meson python3 python3-pip flex bison libva-dev autoconf libtool libpciaccess-dev libssl-dev pkg-config libdrm-dev libgbm-dev libcogl-pango-dev libudev-dev lld clang libx11-xcb-dev libpciaccess-dev nasm yasm
RUN pip3 install meson==0.55.0
RUN git clone https://github.com/GStreamer/gst-build.git; \
  cd gst-build; git checkout 1.19.2; \
  meson builddir --buildtype=release -Dvaapi=enabled -Dvaapi:=disabled: -Dbad=enabled -Dgst-plugins-bad:msdk=disabled -Dgst-plugins-good:soup=disabled -Dgst-plugin-good:ffmpeg=disabled -Dintrospection=disabled -Dgst-plugin-good:sqlite3=disabled -Dpygobject:gobject-introspection=disabled -Dgst-plugins-bad:openjpeg=disabled -Dgst-plugins-bad:tests=disabled -Dgst-plugins-bad:examples=disabled -Dtests=disabled -Dexamples=disabled -Dgst-examples=disabled; \
  ninja -C builddir; cd builddir; ninja install

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/x86_64-linux-gnu/:/usr/local/lib:/ovms/lib

# enable hw decoders if choosing to use FFMPEG. Note default in this POC is GST though.
ENV OPENCV_FFMPEG_CAPTURE_OPTIONS="hw_decoders_any;vaapi"


