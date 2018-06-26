# StarDist

The code in this repository implements object detection with star-convex polygons as described in the paper:

Uwe Schmidt, Martin Weigert, Coleman Broaddus, and Gene Myers.  
[*Cell Detection with Star-convex Polygons*](https://arxiv.org/abs/1806.03535).  
International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), Granada, Spain, September 2018.

Please cite the paper if you are using this code in your research.


## Installation

This package requires Python 3 and can be installed with `pip`:

    pip install stardist

### Notes
- Depending on your Python installation, you may need to use `pip3` instead of `pip`.
- Since this package relies on a C++ extension, you may run into compilation problems on some platforms. We currently do not provide pre-compiled binaries.


## Usage

We provide several Jupyter [notebooks](https://github.com/mpicbg-csbd/stardist/examples) that illustrate how this package can be used.