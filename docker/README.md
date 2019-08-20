## Run *StarDist* with Docker + NVIDIA Container Toolkit

With this Docker image, you can run *StarDist* with [GPU support](https://www.tensorflow.org/install/gpu).

[gputools](https://github.com/maweigert/gputools) and [Multi-Label Anisotropic 3D Euclidean Distance Transform (MLAEDT-3D)](https://github.com/seung-lab/euclidean-distance-transform-3d) are also included.

This Docker image is extended from `tensorflow/tensorflow:1.13.2-gpu-py3-jupyter`, which is based on `Ubuntu 18.04` with `CUDA 10.0` and `Python 3.6.8`.

### Prerequisites

- [Docker 19.03](https://docs.docker.com/install/)
- NVIDIA Driver
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)

### Building image

    # assuming that you are in the docker/ directory
    docker build -t stardist --build-arg NVIDIA_DRIVER_VERSION=430 .

Please change the `NVIDIA_DRIVER_VERSION` depending on your environment.

This argument is used to install an appropriate package for running OpenCL.

The default value is `430` and `390`|`410`|`418`|`430` are available.

### Usage example

    # assuming that you are in the project root directory
    docker run --rm -it --gpus 1 -v $(pwd):/tf/stardist -p 8888:8888 stardist

This command will launch a jupyter notebook as root, binding the project root directory to `/tf/stardist`.

### Notes

- Please see the [Dockerfile](Dockerfile) and [its parent image](https://github.com/tensorflow/tensorflow/tree/v1.13.2/tensorflow/tools/dockerfiles) for details.
- Please follow the instructions for the usage of [Docker + NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker).
- Previous versions of Docker + nvidia-docker2 might work as well but we have not tested.