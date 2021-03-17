import numpy as np
from pathlib import Path
from stardist.models import Config3D, StarDist3D    
from csbdeep.utils.tf import keras_import
Sequence = keras_import('utils','Sequence')
LambdaCallback = keras_import('callbacks','LambdaCallback')
import argparse
import resource
import sys


def print_memory():
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    unit = 1e9 if sys.platform=="darwin" else 1e6
    print(f"\n >>>> total memory used: {mem/unit:.2f} GB \n",flush=True)

    

class LargeSequence(Sequence):
    def __init__(self, n=1000, size=256):
        self.n = n
        self.data = np.zeros((size,size,size), np.uint16)
        self.data[1:-1,1:-1,1:-1] = 1
        self.last = None
        
    def __getitem__(self,n):
        if self.last is None or self.last[0] !=n:
            self.last = (n,self.data.copy())        
        return self.last[1]
    
    def __len__(self):
        return self.n


    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("-s","--size", type=int, default=256)
    parser.add_argument("--nocache", action='store_true')
    
    args = parser.parse_args()
    

    X, Y = LargeSequence(100, size=args.size),LargeSequence(100, size=args.size)
        
    conf = Config3D(
        n_rays=8,
        backbone="unet",
        unet_n_depth=1,
        train_epochs=args.n,
        train_steps_per_epoch=20,
        train_batch_size=2,
        train_patch_size=(min(96,args.size),)*3,
        train_sample_cache = not args.nocache
    )

    model = StarDist3D(conf, None,None)
    model.prepare_for_training()
    model.callbacks.append(LambdaCallback(on_epoch_end=lambda a,b: print_memory()))
    model.train(X, Y, validation_data=(X[0][np.newaxis], Y[0][np.newaxis]))

    
