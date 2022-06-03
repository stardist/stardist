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

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""
Prediction script for a 2D stardist model, usage: stardist-predict -i input.tif -m model_folder_or_pretrained_name -o output_folder

""")
    parser.add_argument("-i","--input", type=str, nargs="+", required=True, help = "input file (tiff)")
    parser.add_argument("-o","--outdir", type=str, default='.', help = "output directory")
    parser.add_argument("--outname", type=str, nargs="+", default='{img}.stardist.tif', help = "output file name (tiff)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-m', '--model', type=str, default=None, help = "model folder / pretrained model to use")
    parser.add_argument("--axes", type=str, default = None, help = "axes to use for the input, e.g. 'XYC'")
    parser.add_argument("--n_tiles", type=int, nargs=2, default = None, help = "number of tiles to use for prediction")
    parser.add_argument("--pnorm", type=float, nargs=2, default = [1,99.8], help = "pmin/pmax to use for normalization")
    parser.add_argument("--prob_thresh", type=float, default=None, help = "prob_thresh for model (if not given use model default)")
    parser.add_argument("--nms_thresh", type=float, default=None, help = "nms_thresh for model (if not given use model default)")
    
    parser.add_argument("-v", "--verbose", action='store_true')
        
    args = parser.parse_args()


    from csbdeep.utils import normalize
    from csbdeep.models.base_model import get_registered_models
    from stardist.models import StarDist2D
    from imageio import imread
    from tifffile import imwrite

    get_registered_models(StarDist2D, verbose=True)

    if pathlib.Path(args.model).is_dir():
        model = StarDist2D(None, name=args.model)
    else:
        model = StarDist2D.from_pretrained(args.model)

    if model is None:
        raise ValueError(f"unknown model: {args.model}\navailable models:\n {get_registered_models(StarDist2D, verbose=True)}")
    
    for fname in args.input:
        if args.verbose:
            print(f'reading image {fname}')

        img = imread(fname)

        if not img.ndim in (2,3):
            raise ValueError(f'currently only 2d and 3d images are supported by the prediction script')

        if args.axes is None:
            args.axes = {2:'YX',3:'YXC'}[img.ndim]
        
        if len(args.axes) != img.ndim:
            raise ValueError(f'dimension of input ({img.ndim}) not the same as length of given axes ({len(args.axes)})')

        if args.verbose:
            print(f'loaded image of size {img.shape}')

        if args.verbose:
            print(f'normalizing...')
            
        img = normalize(img,*args.pnorm)

        labels, _ = model.predict_instances(img,
                                n_tiles=args.n_tiles,
                                prob_thresh=args.prob_thresh,
                                nms_thresh=args.nms_thresh)
        out = pathlib.Path(args.outdir)
        out.mkdir(parents=True,exist_ok=True)

        imwrite(out/args.outname.format(img=pathlib.Path(fname).with_suffix('').name), labels, compress=3)
        

if __name__ == '__main__':
    main()
