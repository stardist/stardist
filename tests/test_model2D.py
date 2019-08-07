import sys
import numpy as np
from stardist.models import Config2D, StarDist2D, StarDistData2D
from utils import circle_image
import tempfile


def test_model():
    img = circle_image()
    imgs = np.repeat(img[np.newaxis],10, axis = 0)

    X = imgs+.6*np.random.uniform(0,1,imgs.shape)
    Y = imgs.astype(int)

    # X = X[..., np.newaxis]
    # Y = Y[..., np.newaxis]

    conf = Config2D (
        n_rays       = 16,
        grid         = (1,1),
        use_gpu      = False,
        train_epochs     = 1,
        train_steps_per_epoch = 10,
        train_loss_weights = (4,1),
        train_patch_size = (128,128),
        n_channel_in = 1)

    with tempfile.TemporaryDirectory() as tmp:
        model = StarDist2D(conf, name='stardist', basedir=tmp)
        model.train(X, Y, validation_data=(X[:3],Y[:3]))



if __name__ == '__main__':
    test_model()
