# StarDist

![](https://github.com/mpicbg-csbd/stardist/raw/master/images/overview.png)


The code in this repository implements object detection with star-convex polygons as described in the paper:

Uwe Schmidt, Martin Weigert, Coleman Broaddus, and Gene Myers.  
[*Cell Detection with Star-convex Polygons*](https://arxiv.org/abs/1806.03535).  
International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), Granada, Spain, September 2018.

Please cite the paper if you are using this code in your research.



``` 
@inproceedings{schmidt2018,
  author    = {Uwe Schmidt and Martin Weigert and Coleman Broaddus and Gene Myers},
  title     = {Cell Detection with Star-Convex Polygons},
  booktitle = {Medical Image Computing and Computer Assisted Intervention - {MICCAI} 
  2018 - 21st International Conference, Granada, Spain, September 16-20, 2018, Proceedings, Part {II}},
  pages     = {265--273},
  year      = {2018},
  doi       = {10.1007/978-3-030-00934-2\_30},
}
```




## Installation

This package requires Python 3.5 (or newer) and can be installed with `pip`:

    pip install stardist

### Notes

- Depending on your Python installation, you may need to use `pip3` instead of `pip`.
- Since this package relies on a C++ extension, you could run into compilation problems (see [Troubleshooting](#troubleshooting) below). We currently do not provide pre-compiled binaries.
- StarDist uses the deep learning library [Keras](https://keras.io), which requires a suitable [backend](https://keras.io/backend/#keras-backends) (we only tested [TensorFlow](http://www.tensorflow.org/)).


## Usage

We provide several Jupyter [notebooks](https://github.com/mpicbg-csbd/stardist/tree/master/examples) that illustrate how this package can be used.

![](https://github.com/mpicbg-csbd/stardist/raw/master/images/example_steps.png)


## Troubleshooting

Installation requires Python 3.5 (or newer) and a working C++ compiler. We have only tested [GCC](http://gcc.gnu.org) (macOS, Linux), [Clang](https://clang.llvm.org) (macOS), and [Visual Studio](https://visualstudio.microsoft.com) (Windows 10). Please [open an issue](https://github.com/mpicbg-csbd/stardist/issues) if you have problems that are not resolved by the information below.

If available, the C++ code will make use of [OpenMP](https://en.wikipedia.org/wiki/OpenMP) to exploit multiple CPU cores for substantially reduced runtime on modern CPUs. This can be important to prevent the function `star_dist` ([utils.py](https://github.com/mpicbg-csbd/stardist/blob/master/stardist/utils.py)) from slowing down model training.


### macOS
Although Apple provides the Clang C/C++ compiler via [Xcode](https://developer.apple.com/xcode/), it does not come with OpenMP support.
Hence, we suggest to install the OpenMP-enabled GCC compiler, e.g. via [Homebrew](https://brew.sh) with `brew install gcc`. After that, you can install the package like this (adjust names/paths as necessary):

    CC=/usr/local/bin/gcc-8 CXX=/usr/local/bin/g++-8 pip install stardist


### Windows
Please install the [Build Tools for Visual Studio 2017](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017) from Microsoft to compile extensions for Python 3.5 and 3.6 (see [this](https://wiki.python.org/moin/WindowsCompilers) for further information). During installation, make sure to select the *Visual C++ build tools*. Note that the compiler comes with OpenMP support.
