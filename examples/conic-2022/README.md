# CoNIC Challenge 2022

The Jupyter notebooks in this folder demonstrate model [training](train.ipynb) and [prediction](predict.ipynb) of our submitted entries to the 2022 [*Colon Nuclei Identification and Counting (CoNIC)* challenge](https://conic-challenge.grand-challenge.org). They make use of additional code in [conic.py](conic.py), which is not part of the StarDist package.

Please see [our paper](https://arxiv.org/abs/2203.02284) for more details.

## Installation

Running these example notebooks requires a custom version of StarDist.
Besides TensorFlow, you need to install StarDist from the `conic-2022` branch:

```
pip install git+https://github.com/stardist/stardist.git@conic-2022
```

Note that this requires you to build the StarDist package (wheel) yourself, i.e. you need a working compiler to do that. Please see [this](https://github.com/stardist/stardist/blob/master/README.md#installation-1) for more details.