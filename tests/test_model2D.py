import sys
import numpy as np
import pytest
from stardist.models import Config2D, StarDist2D
from utils import circle_image
import tempfile



@pytest.mark.parametrize('n_rays, grid, n_channel', [(17,(1,1),None), (32,(2,4),1), (4,(8,2),2)])
def test_model(n_rays, grid, n_channel):
    img = circle_image()
    imgs = np.repeat(img[np.newaxis], 10, axis=0)

    if n_channel is not None:
        imgs = np.repeat(imgs[...,np.newaxis], n_channel, axis=-1)
    else:
        n_channel = 1

    X = imgs+.6*np.random.uniform(0,1,imgs.shape)
    Y = (imgs if imgs.ndim==3 else imgs[...,0]).astype(int)

    conf = Config2D (
        n_rays                = n_rays,
        grid                  = grid,
        n_channel_in          = n_channel,
        use_gpu               = False,
        train_epochs          = 1,
        train_steps_per_epoch = 10,
        train_batch_size      = 2,
        train_loss_weights    = (4,1),
        train_patch_size      = (128,128),
    )

    with tempfile.TemporaryDirectory() as tmp:
        model = StarDist2D(conf, name='stardist', basedir=tmp)
        model.train(X, Y, validation_data=(X[:3],Y[:3]))



if __name__ == '__main__':
    test_model()
