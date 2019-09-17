import sys
import numpy as np
import pytest
from stardist.models import Config3D, StarDist3D
from utils import circle_image


@pytest.mark.parametrize('n_rays, grid, n_channel', [(73,(2,2,2),None), (33,(1,2,4),1), (7,(2,1,1),2)])
def test_model(tmpdir, n_rays, grid, n_channel):
    img = circle_image(shape=(64,80,96))
    imgs = np.repeat(img[np.newaxis], 3, axis=0)

    if n_channel is not None:
        imgs = np.repeat(imgs[...,np.newaxis], n_channel, axis=-1)
    else:
        n_channel = 1

    X = imgs+.6*np.random.uniform(0,1,imgs.shape)
    Y = (imgs if imgs.ndim==4 else imgs[...,0]).astype(int)

    conf = Config3D (
        rays                  = n_rays,
        grid                  = grid,
        n_channel_in          = n_channel,
        use_gpu               = False,
        train_epochs          = 1,
        train_steps_per_epoch = 2,
        train_batch_size      = 2,
        train_loss_weights    = (4,1),
        train_patch_size      = (48,64,64),
    )

    model = StarDist3D(conf, name='stardist', basedir=str(tmpdir))
    model.train(X, Y, validation_data=(X[:2],Y[:2]))



if __name__ == '__main__':
    test_model()
