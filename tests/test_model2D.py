import sys
import numpy as np
import pytest
from pathlib import Path
from stardist.models import Config2D, StarDist2D
from stardist.matching import matching
from stardist.utils import export_imagej_rois
from stardist.plot import render_label, render_label_pred
from csbdeep.utils import normalize
from utils import circle_image, real_image2d, path_model2d

@pytest.mark.parametrize('n_rays, grid, n_channel', [(17, (1, 1), None), (32, (2, 4), 1), (4, (8, 2), 2)])
def test_model(tmpdir, n_rays, grid, n_channel):
    img = circle_image(shape=(160, 160))
    imgs = np.repeat(img[np.newaxis], 3, axis=0)

    if n_channel is not None:
        imgs = np.repeat(imgs[..., np.newaxis], n_channel, axis=-1)
    else:
        n_channel = 1

    X = imgs+.6*np.random.uniform(0, 1, imgs.shape)
    Y = (imgs if imgs.ndim == 3 else imgs[..., 0]).astype(int)

    conf = Config2D(
        n_rays=n_rays,
        grid=grid,
        n_channel_in=n_channel,
        use_gpu=False,
        train_epochs=1,
        train_steps_per_epoch=2,
        train_batch_size=2,
        train_loss_weights=(4, 1),
        train_patch_size=(128, 128),
    )

    model = StarDist2D(conf, name='stardist', basedir=str(tmpdir))
    model.train(X, Y, validation_data=(X[:2], Y[:2]))
    ref = model.predict(X[0])
    res = model.predict(X[0], n_tiles=(
        (2, 3) if X[0].ndim == 2 else (2, 3, 1)))
    # assert all(np.allclose(u,v) for u,v in zip(ref,res))

    # ask to train only with foreground patches when there are none
    # include a constant label image that must trigger a warning
    conf.train_foreground_only = 1
    conf.train_steps_per_epoch = 1
    _X = X[:2]
    _Y = [np.zeros_like(Y[0]), np.ones_like(Y[1])]
    with pytest.warns(UserWarning):
        StarDist2D(conf, name='stardist', basedir=None).train(
            _X, _Y, validation_data=(X[-1:], Y[-1:]))


def test_load_and_predict(model2d):
    model = model2d
    img, mask = real_image2d()
    x = normalize(img, 1, 99.8)
    prob, dist = model.predict(x, n_tiles=(2, 3))
    assert prob.shape == dist.shape[:2]
    assert model.config.n_rays == dist.shape[-1]
    labels, polygons = model.predict_instances(x)
    assert labels.shape == img.shape[:2]
    assert labels.max() == len(polygons['coord'])
    assert len(polygons['coord']) == len(
        polygons['points']) == len(polygons['prob'])
    stats = matching(mask, labels, thresh=0.5)
    assert (stats.fp, stats.tp, stats.fn) == (1, 48, 17)
    return labels


def test_load_and_export_TF(model2d):
    model = model2d
    assert any(g>1 for g in model.config.grid)
    # model.export_TF(single_output=False, upsample_grid=False)
    # model.export_TF(single_output=False, upsample_grid=True)
    model.export_TF(single_output=True, upsample_grid=False)
    model.export_TF(single_output=True, upsample_grid=True)


def test_optimize_thresholds(model2d):
    model = model2d
    img, mask = real_image2d()
    x = normalize(img, 1, 99.8)

    res = model.optimize_thresholds([x], [mask],
                              nms_threshs=[.3, .5],
                              iou_threshs=[.3, .5],
                              optimize_kwargs=dict(tol=1e-1),
                              save_to_json=False)

    np.testing.assert_almost_equal(res["prob"], 0.454617141955, decimal=3)
    np.testing.assert_almost_equal(res["nms"] , 0.3, decimal=3)


def test_stardistdata():
    from stardist.models import StarDistData2D
    img, mask = real_image2d()
    s = StarDistData2D([img, img], [mask, mask],
                       batch_size=1, patch_size=(30, 40), n_rays=32, length=1)
    (img,), (prob, dist) = s[0]
    return (img,), (prob, dist), s


def render_label_example(model2d):
    model = model2d
    img, y_gt = real_image2d()
    x = normalize(img, 1, 99.8)
    y, _ = model.predict_instances(x)
    # im =  render_label(y,img = x, alpha = 0.3, alpha_boundary=1, cmap = (.3,.4,0))
    im =  render_label(y,img = x, alpha = 0.3, alpha_boundary=1)
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.imshow(im)
    plt.show()
    return im


def render_label_pred_example(model2d):
    model = model2d
    img, y_gt = real_image2d()
    x = normalize(img, 1, 99.8)
    y, _ = model.predict_instances(x)

    im = render_label_pred(y_gt, y , img = x)
    import matplotlib.pyplot as plt
    plt.figure(1, figsize = (12,4))
    plt.subplot(1,4,1);plt.imshow(x);plt.title("img")
    plt.subplot(1,4,2);plt.imshow(render_label(y_gt, img = x));plt.title("gt")
    plt.subplot(1,4,3);plt.imshow(render_label(y, img = x));plt.title("pred")
    plt.subplot(1,4,4);plt.imshow(im);plt.title("tp (green) fp (red) fn(blue)")
    plt.tight_layout()
    plt.show()
    return im


def test_pretrained_scales():
    from scipy.ndimage import zoom
    from stardist.matching import matching
    from skimage.measure import regionprops

    model = StarDist2D.from_pretrained("2D_versatile_fluo")
    img, mask = real_image2d()
    x = normalize(img, 1, 99.8)

    def pred_scale(scale=2):
        x2 = zoom(x, scale, order=1)
        labels2, _ = model.predict_instances(x2)
        labels = zoom(labels2, tuple(_s1/_s2 for _s1, _s2 in zip(mask.shape, labels2.shape)), order=0)
        return labels

    scales = np.linspace(.5,5,10)
    accs = tuple(matching(mask, pred_scale(s)).accuracy for s in scales)
    print("scales   ", np.round(scales,2))
    print("accuracy ", np.round(accs,2))

    return accs


def test_stardistdata_sequence():
    from stardist.models import StarDistData2D
    from csbdeep.utils.tf import keras_import
    Sequence = keras_import('utils','Sequence')

    x = np.zeros((100,100), np.uint16)
    x[10:-10,10:-10] = 1

    class MyData(Sequence):
        def __init__(self, dtype):
            self.dtype = dtype
        def __getitem__(self,n):
            return ((1+n)*x).astype(self.dtype)
        def __len__(self):
            return 1000

    X = MyData(np.float32)
    Y = MyData(np.uint16)
    s = StarDistData2D(X,Y,
                       batch_size=1, patch_size=(100,100), n_rays=32, length=1)
    (img,), (prob, dist) = s[0]
    return (img,), (prob, dist), s


def test_imagej_rois_export(tmpdir, model2d):
    img = normalize(real_image2d()[0], 1, 99.8)
    labels, polys = model2d.predict_instances(img)
    export_imagej_rois(str(Path(tmpdir)/'img_rois.zip'), polys['coord'])


def print_receptive_fields():
    for backbone in ("unet",):
        for n_depth in (1,2,3):
            for grid in ((1,1),(2,2)):
                conf  = Config2D(backbone = backbone,
                                 grid = grid,
                                 unet_n_depth=n_depth)
                model = StarDist2D(conf, None, None)
                fov   = model._compute_receptive_field()
                print(f"backbone: {backbone} \t n_depth: {n_depth} \t grid {grid} -> fov: {fov}")


if __name__ == '__main__':
    from conftest import model2d
    # test_model("tmpdir", 32, (1, 1), 1)
    # im = render_label_pred_example(model2d())
    accs = test_pretrained_scales()

