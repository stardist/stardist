import sys
import numpy as np
import pytest
from stardist.models import Config3D, StarDist3D
from stardist.matching import matching
from csbdeep.utils import normalize
from utils import circle_image, real_image3d, path_model3d


@pytest.mark.parametrize('n_rays, grid, n_channel, backbone', [(73, (2, 2, 2), None, 'resnet'), (33, (1, 2, 4), 1, 'resnet'), (7, (2, 1, 1), 2, 'unet')])
def test_model(tmpdir, n_rays, grid, n_channel, backbone):
    img = circle_image(shape=(64, 80, 96))
    imgs = np.repeat(img[np.newaxis], 3, axis=0)

    if n_channel is not None:
        imgs = np.repeat(imgs[..., np.newaxis], n_channel, axis=-1)
    else:
        n_channel = 1

    X = imgs+.6*np.random.uniform(0, 1, imgs.shape)
    Y = (imgs if imgs.ndim == 4 else imgs[..., 0]).astype(int)

    conf = Config3D(
        backbone=backbone,
        rays=n_rays,
        grid=grid,
        n_channel_in=n_channel,
        use_gpu=False,
        train_epochs=1,
        train_steps_per_epoch=2,
        train_batch_size=2,
        train_loss_weights=(4, 1),
        train_patch_size=(48, 64, 64),
    )

    model = StarDist3D(conf, name='stardist', basedir=str(tmpdir))
    model.train(X, Y, validation_data=(X[:2], Y[:2]))
    ref = model.predict(X[0])
    res = model.predict(X[0], n_tiles=(
        (1, 2, 3) if X[0].ndim == 3 else (1, 2, 3, 1)))
    # assert all(np.allclose(u,v) for u,v in zip(ref,res))

    # ask to train only with foreground patches when there are none
    # include a constant label image that must trigger a warning
    conf.train_foreground_only = 1
    conf.train_steps_per_epoch = 1
    _X = X[:2]
    _Y = [np.zeros_like(Y[0]), np.ones_like(Y[1])]
    with pytest.warns(UserWarning):
        StarDist3D(conf, name='stardist', basedir=None).train(
            _X, _Y, validation_data=(X[-1:], Y[-1:]))


def test_load_and_predict():
    model_path = path_model3d()
    model = StarDist3D(None, name=model_path.name,
                       basedir=str(model_path.parent))
    img, mask = real_image3d()
    x = normalize(img, 1, 99.8)
    prob, dist = model.predict(x, n_tiles=(1, 2, 2))
    assert prob.shape == dist.shape[:3]
    assert model.config.n_rays == dist.shape[-1]
    labels, _ = model.predict_instances(x)
    assert labels.shape == img.shape[:3]
    stats = matching(mask, labels, thresh=0.5)
    assert (stats.fp, stats.tp, stats.fn) == (0, 30, 21)
    return model, labels


def test_load_and_predict_with_overlap():
    model_path = path_model3d()
    model = StarDist3D(None, name=model_path.name,
                       basedir=str(model_path.parent))
    img, mask = real_image3d()
    x = normalize(img, 1, 99.8)
    prob, dist = model.predict(x, n_tiles=(1, 2, 2))
    assert prob.shape == dist.shape[:3]
    assert model.config.n_rays == dist.shape[-1]
    labels, _ = model.predict_instances(x, nms_thresh=.5,
                                        overlap_label=-3)
    assert np.min(labels) == -3
    return model, labels


def test_predict_dense_sparse():
    model_path = path_model3d()
    model = StarDist3D(None, name=model_path.name,
                       basedir=str(model_path.parent))
    img, mask = real_image3d()
    x = normalize(img, 1, 99.8)
    labels1, res1 = model.predict_instances(x, n_tiles=(1, 2, 2))
    labels2, res2 = model.predict_instances(x, n_tiles=(1, 2, 2), sparse = True)
    assert np.allclose(labels1, labels2)
    assert all(np.allclose(res1[k], res2[k]) for k in set(res1.keys()).union(set(res2.keys())) )
    return labels2, labels2 


def test_load_and_export_TF():
    model_path = path_model3d()
    model = StarDist3D(None, name=model_path.name,
                       basedir=str(model_path.parent))
    assert any(g>1 for g in model.config.grid)
    # model.export_TF(single_output=False, upsample_grid=False)
    # model.export_TF(single_output=False, upsample_grid=True)
    model.export_TF(single_output=True, upsample_grid=False)
    model.export_TF(single_output=True, upsample_grid=True)


def test_optimize_thresholds():
    model_path = path_model3d()
    model = StarDist3D(None, name=model_path.name,
                       basedir=str(model_path.parent))
    img, mask = real_image3d()
    x = normalize(img, 1, 99.8)

    def _opt(model):
        return model.optimize_thresholds([x], [mask],
                                         nms_threshs=[.3, .5],
                                         iou_threshs=[.3, .5],
                                         optimize_kwargs=dict(tol=1e-1),
                                         save_to_json=False)

    t1 = _opt(model)
    # enforce implicit tiling
    model.config.train_batch_size = 1
    model.config.train_patch_size = tuple(s-1 for s in x.shape)
    t2 = _opt(model)
    assert all(np.allclose(t1[k], t2[k]) for k in t1.keys())
    return model


def test_stardistdata():
    from stardist.models import StarDistData3D
    from stardist import Rays_GoldenSpiral
    img, mask = real_image3d()
    s = StarDistData3D([img, img], [mask, mask], batch_size=1,
                       patch_size=(30, 40, 50), rays=Rays_GoldenSpiral(64))
    (img, mask), (prob, dist) = s[0]
    return (img, mask), (prob, dist), s


if __name__ == '__main__':
    # model, lbl = test_load_and_predict_with_overlap()
    y1,y2 = test_predict_dense_sparse()
