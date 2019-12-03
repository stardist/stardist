import sys
import numpy as np
import pytest
from stardist.models import Config2D, StarDist2D
from stardist.matching import matching
from csbdeep.utils import normalize
from utils import circle_image, real_image2d, path_model2d


@pytest.mark.parametrize('n_rays, grid, n_channel', [(17,(1,1),None), (32,(2,4),1), (4,(8,2),2)])
def test_model(tmpdir, n_rays, grid, n_channel):
    img = circle_image(shape=(160,160))
    imgs = np.repeat(img[np.newaxis], 3, axis=0)

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
        train_steps_per_epoch = 2,
        train_batch_size      = 2,
        train_loss_weights    = (4,1),
        train_patch_size      = (128,128),
    )

    model = StarDist2D(conf, name='stardist', basedir=str(tmpdir))
    model.train(X, Y, validation_data=(X[:2],Y[:2]))
    ref = model.predict(X[0])
    res = model.predict(X[0], n_tiles=((2,3) if X[0].ndim==2 else (2,3,1)))
    # assert all(np.allclose(u,v) for u,v in zip(ref,res))

    # ask to train only with foreground patches when there are none
    # include a constant label image that must trigger a warning
    conf.train_foreground_only = 1
    conf.train_steps_per_epoch = 1
    _X = X[:2]
    _Y = [np.zeros_like(Y[0]), np.ones_like(Y[1])]
    with pytest.warns(UserWarning):
        StarDist2D(conf, name='stardist', basedir=None).train(_X, _Y, validation_data=(X[-1:],Y[-1:]))


def test_load_and_predict():
    model_path = path_model2d()
    model = StarDist2D(None, name=model_path.name, basedir=str(model_path.parent))
    img, mask = real_image2d()
    x = normalize(img,1,99.8)
    prob, dist = model.predict(x, n_tiles=(2,3))
    assert prob.shape == dist.shape[:2]
    assert model.config.n_rays == dist.shape[-1]
    labels, polygons = model.predict_instances(x)
    assert labels.shape == img.shape[:2]
    assert labels.max() == len(polygons['coord'])
    assert len(polygons['coord']) == len(polygons['points']) == len(polygons['prob'])
    stats = matching(mask, labels, thresh=0.5)
    assert (stats.fp, stats.tp, stats.fn) == (1, 48, 17)


def test_stardistdata():
    from stardist.models import StarDistData2D
    img, mask = real_image2d()
    s = StarDistData2D([img,img], [mask,mask], batch_size=1, patch_size=(30,40), n_rays=32)
    (img,mask), (prob,dist) = s[0]
    return (img,mask), (prob,dist), s


if __name__ == '__main__':
    test_model("tmpdir",32,(1,1),1)
