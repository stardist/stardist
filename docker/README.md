## Run *StarDist* with Docker + NVIDIA Container Toolkit

With this Docker image, you can run *StarDist* with [GPU support](https://www.tensorflow.org/install/gpu).

[gputools](https://github.com/maweigert/gputools) and [Multi-Label Anisotropic 3D Euclidean Distance Transform (MLAEDT-3D)](https://github.com/seung-lab/euclidean-distance-transform-3d) are also included.

This Docker image is extended from `tensorflow/tensorflow:2.11.0-gpu-jupyter`, which is based on `Ubuntu 20.04.5 LTS` with `CUDA 11.2` and `Python 3.8.10`.

### Prerequisites

- [Docker 19.03](https://docs.docker.com/install/)
- NVIDIA Driver
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)

For further information, see the [TensorFlow Docker documentation](https://www.tensorflow.org/install/docker).

### Building image

    # assuming that you are in the docker/ directory
    docker build -t stardist --build-arg NVIDIA_DRIVER_VERSION=470 .

Please change the `NVIDIA_DRIVER_VERSION` depending on your environment.

Try `nvidia-smi | grep "Driver Version"` or `cat /proc/driver/nvidia/version` to find your driver version.

This argument is used to install an appropriate package for running OpenCL.

### Usage example

    # assuming that you are in the project root directory
    docker run --rm -it --gpus 1 -v $(pwd):/tf/stardist -p 8888:8888 stardist

This command will launch a jupyter notebook as root, binding the project root directory to `/tf/stardist`.

### Notes

- Please see the [Dockerfile](Dockerfile) and [its parent image](https://hub.docker.com/layers/tensorflow/tensorflow/2.11.0-gpu-jupyter/images/sha256-fc519621eb9a54591721e9019f1606688c9abb329b16b00cc7107c23f14a6f24) for details.
- Please follow the instructions for the usage of [Docker + NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker).
- Previous versions of Docker + nvidia-docker2 might work as well but have not been tested.