"""

Command line script to perform prediction in 2D

"""


import os
import sys
import numpy as np
from tqdm import tqdm
import json 
import argparse
import pprint 
import pathlib
import warnings
from csbdeep.utils import normalize
from csbdeep.models.base_model import get_registered_models
from stardist.models import StarDist2D, Config2D
from tifffile import imread, imsave

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""
Prediction script for a 2D stardist model, usage: stardist-predict -i input.tif -m model_folder_or_pretrained_name -o output.tif

""")
    parser.add_argument("-i","--input", type=str, required=True, help = "input file (tiff)")
    parser.add_argument("-o","--output", type=str, default='label.tif', help = "output file (tiff)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-m', '--model', type=str, default=None, help = "model folder/ pretrained model to use")
    parser.add_argument("--axes", type=str, default = None, help = "axes to use for the input, e.g. 'XYC'")
    parser.add_argument("--n_tiles", type=int, nargs='+', default = None, help = "number of tiles to use for prediction")
    parser.add_argument("--pnorm", type=float, nargs=2, default = [3,99.8], help = "pmin/pmax to use for normalization")
    parser.add_argument("--prob_thresh", type=float, default=None, help = "prob_thresh for model (if not given use model default)")
    parser.add_argument("--nms_thresh", type=float, default=None, help = "nms_thresh for model (if not given use model default)")
    
    parser.add_argument("-v", "--verbose", action='store_true')
        
    args = parser.parse_args()

    if not pathlib.Path(args.input).suffix.lower() in (".tif", ".tiff"):
        raise ValueError('only tiff files supported for now')

    get_registered_models(StarDist2D, verbose=True);

    if args.verbose:
        print('reading image...')

    img = imread(args.input)

    if not img.ndim in (2,3):
        raise ValueError(f'currently only 2d and 3d images are supported by the prediction script')

    if args.axes is None:
        args.axes = {2:'YX',3:'ZYX'}
    
    if len(args.axes) != img.ndim:
        raise ValueError(f'dimension of input ({img.ndim}) not the same as length of given axes ({len(args.axes)})')

    try:
        if Path(args.model).is_dir():
            model = StarDist2D(None, name=args.model)
        else:
            model = StarDist2D.from_pretrained(args.model)
    except

    if args.verbose:
        print(f'loaded image of size {img.shape}')

    if args.verbose:
        print(f'normalizing...')
        
    img = normalize(img,*args.pnorm)

    labels, _ = model.predict_instances(img,
                            n_tiles=args.n_tiles,
                            prob_thresh=args.prob_thresh,
                            nms_thresh=args.nms_thresh)
    
    imsave(args.output, labels, compress=3)

    return img


if __name__ == '__main__':
    args = main()
