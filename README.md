[![PyPI version](https://badge.fury.io/py/stardist.svg)](https://pypi.org/project/stardist)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/stardist/badges/version.svg)](https://anaconda.org/conda-forge/stardist)
[![Test](https://github.com/stardist/stardist/workflows/Test/badge.svg)](https://github.com/stardist/stardist/actions?query=workflow%3ATest)
[![Test (PyPI)](https://github.com/stardist/stardist/workflows/Test%20(PyPI)/badge.svg)](https://github.com/stardist/stardist/actions?query=workflow%3A%22Test+%28PyPI%29%22)
[![Image.sc forum](https://img.shields.io/badge/dynamic/json.svg?label=forum&url=https%3A%2F%2Fforum.image.sc%2Ftags%2Fstardist.json&query=%24.topic_list.tags.0.topic_count&colorB=brightgreen&suffix=%20topics&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAABPklEQVR42m3SyyqFURTA8Y2BER0TDyExZ+aSPIKUlPIITFzKeQWXwhBlQrmFgUzMMFLKZeguBu5y+//17dP3nc5vuPdee6299gohUYYaDGOyyACq4JmQVoFujOMR77hNfOAGM+hBOQqB9TjHD36xhAa04RCuuXeKOvwHVWIKL9jCK2bRiV284QgL8MwEjAneeo9VNOEaBhzALGtoRy02cIcWhE34jj5YxgW+E5Z4iTPkMYpPLCNY3hdOYEfNbKYdmNngZ1jyEzw7h7AIb3fRTQ95OAZ6yQpGYHMMtOTgouktYwxuXsHgWLLl+4x++Kx1FJrjLTagA77bTPvYgw1rRqY56e+w7GNYsqX6JfPwi7aR+Y5SA+BXtKIRfkfJAYgj14tpOF6+I46c4/cAM3UhM3JxyKsxiOIhH0IO6SH/A1Kb1WBeUjbkAAAAAElFTkSuQmCC)](https://forum.image.sc/tags/stardist)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/stardist)](https://pypistats.org/packages/stardist)

# *StarDist* - Object Detection with Star-convex Shapes

![](https://github.com/stardist/stardist/raw/main/images/stardist_overview.png)

This repository contains the Python implementation of star-convex object detection for 2D and 3D images, as described in the papers:

<img src="https://github.com/stardist/stardist/raw/main/images/stardist_logo.jpg" title="siân is the king of the universe" width="25%" align="right">

- Uwe Schmidt, Martin Weigert, Coleman Broaddus, and Gene Myers.  
[*Cell Detection with Star-convex Polygons*](https://arxiv.org/abs/1806.03535).  
International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), Granada, Spain, September 2018.

- Martin Weigert, Uwe Schmidt, Robert Haase, Ko Sugawara, and Gene Myers.  
[*Star-convex Polyhedra for 3D Object Detection and Segmentation in Microscopy*](http://openaccess.thecvf.com/content_WACV_2020/papers/Weigert_Star-convex_Polyhedra_for_3D_Object_Detection_and_Segmentation_in_Microscopy_WACV_2020_paper.pdf).  
The IEEE Winter Conference on Applications of Computer Vision (WACV), Snowmass Village, Colorado, March 2020.

- Martin Weigert and Uwe Schmidt.  
[*Nuclei Instance Segmentation and Classification in Histopathology Images with Stardist*](https://arxiv.org/abs/2203.02284).  
The IEEE International Symposium on Biomedical Imaging Challenges (ISBIC), Kolkata, India, March 2022.

Please [cite the paper(s)](#how-to-cite) if you are using this code in your research.


## Overview

The following figure illustrates the general approach for 2D images. The training data consists of corresponding pairs of input (i.e. raw) images and fully annotated label images (i.e. every pixel is labeled with a unique object id or 0 for background).
A model is trained to densely predict the distances (r) to the object boundary along a fixed set of rays and object probabilities (d), which together produce an overcomplete set of candidate polygons for a given input image. The final result is obtained via non-maximum suppression (NMS) of these candidates.

![](https://github.com/stardist/stardist/raw/main/images/overview_2d.png)

The approach for 3D volumes is similar to the one described for 2D, using pairs of input and fully annotated label volumes as training data.

![](https://github.com/stardist/stardist/raw/main/images/overview_3d.png)

## Webinar/Tutorial

If you want to know more about the concepts and practical applications of StarDist, please have a look at the following webinar that was given at NEUBIAS Academy @Home 2020:

[![webinar video](http://img.youtube.com/vi/Amn_eHRGX5M/0.jpg)](http://www.youtube.com/watch?v=Amn_eHRGX5M "Webinar")


## Installation

This package is compatible with Python 3.6 - 3.12.

If you only want to use a StarDist plugin for a GUI-based software, please read [this](#plugins-for-other-software).

1. Please first [install TensorFlow](https://www.tensorflow.org/install)
(either TensorFlow 1 or 2) by following the official instructions.
For [GPU support](https://www.tensorflow.org/install/gpu), it is very
important to install the specific versions of CUDA and cuDNN that are
compatible with the respective version of TensorFlow. (If you need help and can use `conda`, take a look at [this](https://github.com/CSBDeep/CSBDeep/tree/main/extras#conda-environment).)

2. *StarDist* can then be installed with `pip`:

    - If you installed TensorFlow 2 (version *2.x.x*):

          pip install stardist

    - If you installed TensorFlow 1 (version *1.x.x*):

          pip install "stardist[tf1]"


#### Notes

- Depending on your Python installation, you may need to use `pip3` instead of `pip`.
- You can find out which version of TensorFlow is installed via `pip show tensorflow`.
- We provide pre-compiled binaries ("wheels") that should work for most Linux, Windows, and macOS platforms. If you're having problems, please see the [troubleshooting](#installation-1) section below.
- *(Optional)* You need to install [gputools](https://github.com/maweigert/gputools) if you want to use OpenCL-based computations on the GPU to speed up training.
- *(Optional)* You might experience improved performance during training if you additionally install the [Multi-Label Anisotropic 3D Euclidean Distance Transform (MLAEDT-3D)](https://github.com/seung-lab/euclidean-distance-transform-3d).


## Usage

We provide example workflows for 2D and 3D via Jupyter [notebooks](https://github.com/stardist/stardist/tree/main/examples) that illustrate how this package can be used.

![](https://github.com/stardist/stardist/raw/main/images/example_steps.png)

### Pretrained Models for 2D

Currently we provide some pretrained models in 2D that might already be suitable for your images:


| key | Modality (Staining) | Image format | Example Image    | Description  |
| :-- | :-: | :-:| :-:| :-- |
| `2D_versatile_fluo` `2D_paper_dsb2018`| Fluorescence (nuclear marker) | 2D single channel| <img src="https://github.com/stardist/stardist/raw/main/images/example_fluo.jpg" title="example image fluo" width="120px" align="center">       | *Versatile (fluorescent nuclei)* and *DSB 2018 (from StarDist 2D paper)* that were both trained on a [subset of the DSB 2018 nuclei segmentation challenge dataset](https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip). |
|`2D_versatile_he` | Brightfield (H&E) | 2D RGB  | <img src="https://github.com/stardist/stardist/raw/main/images/example_histo.jpg" title="example image histo" width="120px" align="center">       | *Versatile (H&E nuclei)* that was trained on images from the [MoNuSeg 2018 training data](https://monuseg.grand-challenge.org/Data/) and the [TNBC dataset from Naylor et al. (2018)](https://zenodo.org/record/1175282#.X6mwG9so-CN). |


You can access these pretrained models from `stardist.models.StarDist2D`

```python
from stardist.models import StarDist2D

# prints a list of available models
StarDist2D.from_pretrained()

# creates a pretrained model
model = StarDist2D.from_pretrained('2D_versatile_fluo')
```

And then try it out with a test image:

```python
from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
import matplotlib.pyplot as plt

img = test_image_nuclei_2d()

labels, _ = model.predict_instances(normalize(img))

plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("input image")

plt.subplot(1,2,2)
plt.imshow(render_label(labels, img=img))
plt.axis("off")
plt.title("prediction + input overlay")
```

![](images/pretrained_example.png)


### Annotating Images

To train a *StarDist* model you will need some ground-truth annotations: for every raw training image there has to be a corresponding label image where all pixels of a cell region are labeled with a distinct integer (and background pixels are labeled with 0).
To create such annotations in 2D, there are several options, among them being [Fiji](http://fiji.sc/), [Labkit](https://imagej.net/Labkit), or [QuPath](https://qupath.github.io). In 3D, there are fewer options: [Labkit](https://github.com/maarzt/imglib2-labkit) and [Paintera](https://github.com/saalfeldlab/paintera) (the latter being very sophisticated but having a steeper learning curve).

Although each of these provide decent annotation tools, we currently recommend using Labkit (for 2D or 3D images) or QuPath (for 2D):

#### Annotating with LabKit (2D or 3D)

1. Install [Fiji](https://fiji.sc) and the [Labkit](https://imagej.net/Labkit) plugin
2. Open the (2D or 3D) image and start Labkit via `Plugins > Labkit > Open Current Image With Labkit`
3. Successively add a new label and annotate a single cell instance with the brush tool until *all* cells are labeled.  
   (Always disable `allow overlapping labels` or – in older versions of LabKit – enable the `override` option.) 
4. Export the label image via `Labeling > Save Labeling ...` with `Files of Type > TIF Image` making sure that the file name ends with `.tif` or `.tiff`.

![](https://github.com/stardist/stardist/raw/main/images/labkit_2d_labkit.png)


Additional tips:

* The Labkit viewer uses [BigDataViewer](https://imagej.net/BigDataViewer) and its keybindings (e.g. <kbd>s</kbd> for contrast options, <kbd>CTRL</kbd>+<kbd>Shift</kbd>+<kbd>mouse-wheel</kbd> for zoom-in/out etc.)
* For 3D images (XYZ) it is best to first convert it to a (XYT) timeseries (via `Re-Order Hyperstack` and swapping `z` and `t`) and then use <kbd>[</kbd> and <kbd>]</kbd> in Labkit to walk through the slices.

#### Annotating with QuPath (2D)

1. Install [QuPath](https://qupath.github.io/)
2. Create a new project (`File -> Project...-> Create project`) and add your raw images
3. Annotate nuclei/objects
4. Run [this script](https://raw.githubusercontent.com/stardist/stardist/main/extras/qupath_export_annotations.groovy) to export the annotations (save the script and drag it on QuPath. Then execute it with `Run for project`). The script will create a `ground_truth` folder within your QuPath project that includes both the `images` and `masks` subfolder that then can directly be used with *StarDist*.

To see how this could be done, have a look at the following [example QuPath project](https://raw.githubusercontent.com/stardist/stardist/main/extras/qupath_example_project.zip) (data courtesy of Romain Guiet, EPFL).

![](https://github.com/stardist/stardist/raw/main/images/qupath.png)


### Multi-class Prediction

StarDist also supports multi-class prediction, i.e. each found object instance can additionally be classified into a fixed number of discrete object classes (e.g. cell types):

![](https://github.com/stardist/stardist/raw/main/images/stardist_multiclass.png)

Please see the [multi-class example notebook](https://nbviewer.jupyter.org/github/stardist/stardist/blob/main/examples/other2D/multiclass.ipynb) if you're interested in this.

## Instance segmentation metrics

StarDist contains the `stardist.matching` submodule that provides functions to compute common instance segmentation metrics between ground-truth label masks and predictions (not necessarily from StarDist). Currently available metrics are

* `tp`, `fp`, `fn`
* `precision`, `recall`, `accuracy`, `f1`
* `panoptic_quality`
* `mean_true_score`, `mean_matched_score`

which are computed by matching ground-truth/prediction objects if their IoU exceeds a threshold (by default 50%). See the documentation of `stardist.matching.matching` for a detailed explanation.

Here is an example how to use it:

```python

# create some example ground-truth and dummy prediction data
from stardist.data import test_image_nuclei_2d
from scipy.ndimage import rotate
_, y_true = test_image_nuclei_2d(return_mask=True)
y_pred = rotate(y_true, 2, order=0, reshape=False)

# compute metrics between ground-truth and prediction
from stardist.matching import matching

metrics =  matching(y_true, y_pred)

print(metrics)
```
```
Matching(criterion='iou', thresh=0.5, fp=88, tp=37, fn=88, precision=0.296, 
       recall=0.296, accuracy=0.1737, f1=0.296, n_true=125, n_pred=125, 
       mean_true_score=0.19490, mean_matched_score=0.65847, panoptic_quality=0.19490)
```

If you want to compare a list of images you can use `stardist.matching.matching_dataset`:

```python

from stardist.matching import matching_dataset

metrics = matching_dataset([y_true, y_true], [y_pred, y_pred])

print(metrics)
```
```
DatasetMatching(criterion='iou', thresh=0.5, fp=176, tp=74, fn=176, precision=0.296, 
    recall=0.296, accuracy=0.1737, f1=0.296, n_true=250, n_pred=250, 
    mean_true_score=0.19490, mean_matched_score=0.6584, panoptic_quality=0.1949, by_image=False)
```



## Troubleshooting & Support

1. Please first take a look at the [frequently asked questions (FAQ)]( https://stardist.net/docs/faq.html).
2. If you need further help, please go to the [image.sc forum](https://forum.image.sc) and try to find out if the issue you're having has already been discussed or solved by other people. If not, feel free to create a new topic there and make sure to use the tag `stardist` (we are monitoring all questions with this tag). When opening a new topic, please provide a clear and concise description to understand and ideally reproduce the issue you're having (e.g. including a code snippet, Python script, or Jupyter notebook).
3. If you have a technical question related to the source code or believe to have found a bug, feel free to [open an issue](https://github.com/stardist/stardist/issues), but please check first if someone already created a similar issue.

### Installation

If `pip install stardist` fails, it could be because there are no compatible wheels (`.whl`) for your platform ([see list](https://pypi.org/project/stardist/#files)). In this case, `pip` tries to compile a C++ extension that our Python package relies on (see below). While this often works on Linux out of the box, it will likely fail on Windows and macOS without installing a suitable compiler. (Note that you can enforce compilation by installing via `pip install stardist --no-binary :stardist:`.)

Installation without using wheels requires Python 3.6 (or newer) and a working C++ compiler. We have only tested [GCC](http://gcc.gnu.org) (macOS, Linux), [Clang](https://clang.llvm.org) (macOS), and [Visual Studio](https://visualstudio.microsoft.com) (Windows 10). Please [open an issue](https://github.com/stardist/stardist/issues) if you have problems that are not resolved by the information below.

If available, the C++ code will make use of [OpenMP](https://en.wikipedia.org/wiki/OpenMP) to exploit multiple CPU cores for substantially reduced runtime on modern CPUs. This can be important to prevent slow model training.

#### macOS
The default C/C++ compiler Clang that comes with the macOS command line tools (installed via `xcode-select --install`) does not support OpenMP out of the box, but it can be added. Alternatively, a suitable compiler can be installed from [conda-forge](https://conda-forge.org). Please see this [detailed guide](https://scikit-learn.org/stable/developers/advanced_installation.html#macos)  for more information on both strategies (although written for [scikit-image](https://scikit-learn.org), it also applies here).

A third alternative (and what we did until StarDist 0.8.1) is to install the OpenMP-enabled GCC compiler via [Homebrew](https://brew.sh) with `brew install gcc` (e.g. installing `gcc-12`/`g++-12` or newer). After that, you can build the package like this (adjust compiler names/paths as necessary):

    CC=gcc-12 CXX=g++-12 pip install stardist

If you use `conda` on macOS and after `import stardist` see errors similar to `Symbol not found: _GOMP_loop_nonmonotonic_dynamic_next`, please see [this issue](https://github.com/stardist/stardist/issues/19#issuecomment-535610758) for a temporary workaround.

##### MacOS OpenMP symbol not found Error 

If you encounter an `ImportError: dlopen(...): symbol not found in flat namespace ...` error on `import stardist`, you may try to install it like so:

```
brew install libomp

libomp_root=$(brew --prefix libomp)

export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
export CFLAGS="$CFLAGS -I$libomp_root/include"
export CXXFLAGS="$CXXFLAGS -I$libomp_root/include"
export LDFLAGS="$LDFLAGS -Wl,-rpath,$libomp_root/lib -L$libomp_root/lib -lomp"
pip install stardist --no-binary :all:
```
##### Apple Silicon

As of StarDist 0.8.2, we provide `arm64` wheels that should work with [macOS on Apple Silicon](https://support.apple.com/en-us/HT211814) (M1 chip or newer). 
We recommend setting up an `arm64` `conda` environment with GPU-accelerated TensorFlow following [Apple's instructions](https://developer.apple.com/metal/tensorflow-plugin/) (ensure you are using macOS 12 Monterey or newer) using [conda-forge miniforge3 or mambaforge](https://github.com/conda-forge/miniforge). Then install `stardist` using `pip`.
```
conda create -y -n stardist-env python=3.9   
conda activate stardist-env
conda install -c apple tensorflow-deps
pip install tensorflow-macos tensorflow-metal
pip install stardist
```

#### Windows
Please install the [Build Tools for Visual Studio 2019](https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2019) (or newer) from Microsoft to compile extensions for Python 3.6+ (see [this](https://wiki.python.org/moin/WindowsCompilers) for further information). During installation, make sure to select the *C++ build tools*. Note that the compiler comes with OpenMP support.

## Plugins for other software

### ImageJ/Fiji

We currently provide a ImageJ/Fiji plugin that can be used to run pretrained StarDist models on 2D or 2D+time images. Installation and usage instructions can be found at the [plugin page](https://imagej.net/StarDist).

### Napari

We made a plugin for the Python-based multi-dimensional image viewer [napari](https://napari.org). It directly uses the StarDist Python package and works for 2D and 3D images. Please see the [code repository](https://github.com/stardist/stardist-napari) for further details.

### QuPath

Inspired by the Fiji plugin, [Pete Bankhead](https://github.com/petebankhead) made a custom implementation of StarDist 2D for [QuPath](https://qupath.github.io) to use pretrained models. Please see [this page](https://qupath.readthedocs.io/en/latest/docs/deep/stardist.html) for documentation and installation instructions.

### Icy

Based on the Fiji plugin, [Deborah Schmidt](https://github.com/frauzufall) made a StarDist 2D plugin for [Icy](https://github.com/stardist/stardist-icy) to use pretrained models. Please see the [code repository](https://github.com/stardist/stardist-icy) for further details.

### KNIME

[Stefan Helfrich](https://github.com/stelfrich) has modified the Fiji plugin to be compatible with [KNIME](https://www.knime.com). Please see [this page](https://hub.knime.com/stelfrich/spaces/Public/latest/StarDist/StarDist%202D) for further details.

## How to cite
```bibtex
@inproceedings{schmidt2018,
  author    = {Uwe Schmidt and Martin Weigert and Coleman Broaddus and Gene Myers},
  title     = {Cell Detection with Star-Convex Polygons},
  booktitle = {Medical Image Computing and Computer Assisted Intervention - {MICCAI} 
  2018 - 21st International Conference, Granada, Spain, September 16-20, 2018, Proceedings, Part {II}},
  pages     = {265--273},
  year      = {2018},
  doi       = {10.1007/978-3-030-00934-2_30}
}

@inproceedings{weigert2020,
  author    = {Martin Weigert and Uwe Schmidt and Robert Haase and Ko Sugawara and Gene Myers},
  title     = {Star-convex Polyhedra for 3D Object Detection and Segmentation in Microscopy},
  booktitle = {The IEEE Winter Conference on Applications of Computer Vision (WACV)},
  month     = {March},
  year      = {2020},
  doi       = {10.1109/WACV45572.2020.9093435}
}

@inproceedings{weigert2022,
  author    = {Martin Weigert and Uwe Schmidt},
  title     = {Nuclei Instance Segmentation and Classification in Histopathology Images with Stardist},
  booktitle = {The IEEE International Symposium on Biomedical Imaging Challenges (ISBIC)},
  year      = {2022},
  doi       = {10.1109/ISBIC56247.2022.9854534}
}
```
