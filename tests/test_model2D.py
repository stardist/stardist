import sys
import numpy as np
import pytest
from stardist.models import Config2D, StarDist2D
from stardist.matching import matching
from stardist.utils import _merge_multiclass
from stardist.plot import render_label, render_label_pred
from csbdeep.utils import normalize
from utils import circle_image, real_image2d, path_model2d


@pytest.mark.parametrize('n_rays, grid, n_channel', [(17, (1, 1), None), (32, (2, 4), 1), (4, (8, 2), 2)])
def test_model(tmpdir, n_rays, grid, n_channel, n_classes):
    radii = np.random.randint(10,30,n_classes)
    w  = 160
    def sample():
        mask = np.zeros((w,w,n_classes), np.uint16)
        for i,r in enumerate(radii):
            mask[...,i] = (.2*i+1)*circle_image(shape=(160, 160), radius = r, center = np.random.randint(-w//3,w//3,2))
        mask_merged = _merge_multiclass(mask)
        img = mask_merged.astype(np.float32)
        img += .6*np.random.uniform(0, 1, img.shape)
        img = img if n_channel is None else np.repeat(np.expand_dims(img,-1),n_channel, -1)
        if n_channel>1:
            img *= np.random.uniform(.4,1.6,(1,1,n_channel)) 
        mask = mask_merged if n_classes==1 else mask
        return img, mask

    X, Y = tuple(zip(*tuple(sample() for _ in range(16))))

    conf = Config2D(
        n_rays=n_rays,
        grid=grid,
        n_channel_in=n_channel,
        n_classes=n_classes,
        use_gpu=False,
        train_epochs=100,
        train_steps_per_epoch=20,
        train_batch_size=2,
        train_loss_weights=(1, 1, .2),
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
    return X,Y, model

def test_load_and_predict():
    model_path = path_model2d()
    model = StarDist2D(None, name=model_path.name,
                       basedir=str(model_path.parent))
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


def test_load_and_export_TF():
    model_path = path_model2d()
    model = StarDist2D(None, name=model_path.name,
                       basedir=str(model_path.parent))
    assert any(g>1 for g in model.config.grid)
    # model.export_TF(single_output=False, upsample_grid=False)
    # model.export_TF(single_output=False, upsample_grid=True)
    model.export_TF(single_output=True, upsample_grid=False)
    model.export_TF(single_output=True, upsample_grid=True)


def test_optimize_thresholds():
    model_path = path_model2d()
    model = StarDist2D(None, name=model_path.name,
                       basedir=str(model_path.parent))
    img, mask = real_image2d()
    x = normalize(img, 1, 99.8)

    res = model.optimize_thresholds([x], [mask],
                              nms_threshs=[.3, .5],
                              iou_threshs=[.3, .5],
                              optimize_kwargs=dict(tol=1e-1),
                              save_to_json=False)

    np.testing.assert_almost_equal(res["prob"], 0.454617141955, decimal=3)
    np.testing.assert_almost_equal(res["nms"] , 0.3, decimal=3)


def test_stardistdata(n_channel=1):
    from stardist.models import StarDistData2D
    img, y = real_image2d()
    if n_channel>1:
        img = np.repeat(np.expand_dims(img,-1),n_channel,-1)
        
    s = StarDistData2D([img, img], [y, y],
                       batch_size=1, patch_size=(30, 40), n_rays=32)
    (img, mask), (prob,prob_class,  dist) = s[0]
    return (img, mask), (prob, prob_class, dist), s

def test_stardistdata_multiclass(n_channel = 1):
    from stardist.models import StarDistData2D
    img, y = real_image2d()
    if n_channel>1:
        img = np.repeat(np.expand_dims(img,-1),n_channel,-1)

    # create 3 non-overlapping masks 
    y = np.stack([y*np.isin(y,np.arange(1+i,y.max()+1,3)) for i in range(3)], axis = -1)
    
    s = StarDistData2D([img, img], [y, y],
                       batch_size=1, patch_size=(30, 40), n_rays=32)
    (img, mask), (prob, prob_class, dist) = s[0]
    return (img, mask), (prob, prob_class, dist), s

def test_multiclass_backwardcompatibility():
    """checks whether n_classes=None gives old stardist model without class head"""

    conf_old = Config2D(n_classes=None, train_loss_weights=(1, 1, .2))

    model_old = StarDist2D(conf_old, name=  None, basedir = None)
    

def render_label_example():
    model_path = path_model2d()
    model = StarDist2D(None, name=model_path.name,
                       basedir=str(model_path.parent))
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

def render_label_pred_example():
    model_path = path_model2d()
    model = StarDist2D(None, name=model_path.name,
                       basedir=str(model_path.parent))
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


if __name__ == '__main__':
    # test_model("tmpdir", 32, (1, 1), 1)
    # im = render_label_pred_example()
    # accs = test_pretrained_scales()
    # (img, mask), (prob, prob_class, dist), s = test_stardistdata(3)    
    # (img, mask), (prob, prob_class, dist), s = test_stardistdata_multiclass(2)    

    # X,Y, model = test_model(".", 32, (2,2), n_channel = 3, n_classes = 5)

    # test_optimize_thresholds()


    test_multiclass_backwardcompatibility()
