import sys
import numpy as np
import pytest
from pathlib import Path
from itertools import product
from stardist.sample_patches import get_valid_inds

@pytest.mark.parametrize('shape, patch_size', 
                        [
                            ((128,128), (128,128)),
                            ((77,11), (36,4)),
                            ((33,34,35), (33,11,1))
                        ])
def test_valid_inds(shape, patch_size):

    y = np.zeros(shape, np.uint16)
    inds = get_valid_inds(y, patch_size=patch_size)

    # just check whether at least the number is correct...
    assert len(inds[0]) == np.prod(1 + np.array(shape) - np.array(patch_size))

    return inds

if __name__ == '__main__':

    inds = test_valid_inds((128,128),(128,11))