FROM tensorflow/tensorflow:1.13.2-gpu-py3-jupyter

LABEL maintainer="ko.sugawara@ens-lyon.fr"

ARG NVIDIA_DRIVER_VERSION=430

RUN apt-get update && apt-get install -y --no-install-recommends \
    ocl-icd-dev \
    ocl-icd-opencl-dev \
    opencl-headers \
    clinfo \
    libnvidia-compute-${NVIDIA_DRIVER_VERSION} \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install stardist gputools edt