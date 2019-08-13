import sys
import numpy as np
import pytest
from stardist.models import Config3D, StarDist3D
from utils import circle_image
import tempfile


@pytest.mark.parametrize('n_rays, grid', [(73,(2,2,2)),(33,(1,2,4))])
def test_model(n_rays, grid):
    img = circle_image(shape = (64,80,96))
    imgs = np.repeat(img[np.newaxis],10, axis = 0)

    X = imgs+.6*np.random.uniform(0,1,imgs.shape)
    Y = imgs.astype(np.uint16)

    conf = Config3D (
        rays       = n_rays,
        grid       = grid,
        use_gpu    = False,
        train_epochs     = 1,
        train_steps_per_epoch = 2,
        train_loss_weights = (4,1),
        train_patch_size = (48,64,64),
        n_channel_in = 1)

    with tempfile.TemporaryDirectory() as tmp:
        model = StarDist3D(conf, name='stardist', basedir=tmp)
        model.train(X, Y, validation_data=(X[:2],Y[:2]))



if __name__ == '__main__':
    test_model()
