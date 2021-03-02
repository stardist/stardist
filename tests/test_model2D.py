import sys
import numpy as np
import pytest
from pathlib import Path
from itertools import product
from stardist.models import Config2D, StarDist2D, StarDistData2D
from stardist.matching import matching
from stardist.utils import export_imagej_rois
from stardist.plot import render_label, render_label_pred
from csbdeep.utils import normalize
from utils import circle_image, real_image2d, path_model2d

@pytest.mark.parametrize('n_rays, grid, n_channel, workers', [(17, (1, 1), None, 1), (32, (2, 4), 1, 1), (4, (8, 2), 2, 1)])
def test_model(tmpdir, n_rays, grid, n_channel, workers):
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
        train_steps_per_epoch=1,
        train_batch_size=2,
        train_loss_weights=(4, 1),
        train_patch_size=(128, 128),
    )

    model = StarDist2D(conf, name='stardist', basedir=str(tmpdir))
    model.train(X, Y, validation_data=(X[:2], Y[:2]), workers=workers)
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

def test_load_and_predict_big():
    model_path = path_model2d()
    model = StarDist2D(None, name=model_path.name,
                       basedir=str(model_path.parent))
    img, _ = real_image2d()
    x = normalize(img, 1, 99.8)
    x = np.tile(x,(8,8))
    labels, polygons = model.predict_instances(x)
    return labels


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


def test_stardistdata(n_classes = None, classes = 1):
    np.random.seed(42)
    from stardist.models import StarDistData2D
    img, mask = real_image2d()
    s = StarDistData2D([img, img], [mask, mask],
                       grid = (2,2),
                       n_classes = n_classes, classes = (classes,classes), 
                       batch_size=1, patch_size=(30, 40), n_rays=32, length=1)
    a, b = s[0]
    return a,b, s


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




def _test_model_multiclass(n_classes = 1, classes = "auto", n_channel = None, basedir = None):
    from skimage.measure import regionprops    
    img, mask = real_image2d()
    img = normalize(img,1,99.8) 

    if n_channel is not None:
        img = np.repeat(img[..., np.newaxis], n_channel, axis=-1)
    else:
        n_channel = 1

    X, Y = [img, img, img], [mask, mask, mask]

    conf = Config2D(
        n_rays=48,
        grid=(2,2),
        n_channel_in=n_channel,
        n_classes = n_classes,
        use_gpu=False,
        train_epochs=1,
        train_steps_per_epoch=1,
        train_batch_size=1,
        train_dist_loss = "iou",
        train_patch_size=(128, 128),
    )

    if n_classes is not None and n_classes>1 and classes=="area":
        regs = regionprops(mask)
        areas = tuple(r.area for r in regs)
        inds = np.argsort(areas)
        ss = tuple(slice(n*len(regs)//n_classes,(n+1)*len(regs)//n_classes) for n in range(n_classes))
        classes = {}
        for i,s in enumerate(ss):
            for j in inds[s]:
                classes[regs[j].label] = i+1
        classes = (classes,)*len(X)
        
    model = StarDist2D(conf, name=None if basedir is None else "stardist", basedir=str(basedir))

    val_classes = {k:1 for k in set(mask[mask>0])}
    
    s = model.train(X, Y, classes = classes, epochs = 100, 
                validation_data=(X[:1], Y[:1]) if n_classes is None else (X[:1], Y[:1], (val_classes,))
                    )

    img = np.tile(img,(4,4) if img.ndim==2 else (4,4,1))

    kwargs = dict(prob_thresh=.2)
    labels1, res1 = model.predict_instances(img, **kwargs)
    labels2, res2 = model.predict_instances(img, sparse = True, **kwargs)
    labels3, res3 = model.predict_instances_big(img, axes="YX" if img.ndim==2 else "YXC",
                                                block_size=640, min_overlap=32, context=96, **kwargs)

    assert np.allclose(labels1, labels2)
    assert all([np.allclose(res1[k], res2[k]) for k in set(res1.keys()).union(set(res2.keys())) if isinstance(res1[k], np.ndarray)])
    
    return model, img, res1, res2

@pytest.mark.parametrize('n_classes, classes, n_channel',
                         [ (None, "auto", 1),
                           (1, "auto", 3),
                           (3, (1,2,3),3)]
                         )
def test_model_multiclass(tmpdir, n_classes, classes, n_channel):
    return _test_model_multiclass(n_classes=n_classes, classes=classes,
                                  n_channel=n_channel, basedir = tmpdir)


def test_classes():
    from stardist.utils import mask_to_categorical
    
    def _parse(n_classes, classes):
        model = StarDist2D(Config2D(n_classes = n_classes), None, None)
        classes =  model._parse_classes_arg(classes, length = 1)
        return classes

    def _check_single_val(n_classes, classes=1):
        img, y_gt = real_image2d()
        labels_gt = set(np.unique(y_gt[y_gt>0]))
        p, cls_dict = mask_to_categorical(y_gt,
                                          n_classes=n_classes,
                                          classes = classes, return_cls_dict = True)
        assert p.shape == y_gt.shape+(n_classes+1,)
        assert tuple(cls_dict.keys()) == (classes,) and  set(cls_dict[classes]) == labels_gt
        assert set(np.where(np.count_nonzero(p, axis = (0,1)))[0]) == set({0,classes})
        return p, cls_dict
        
    assert _parse(None,"auto") is None
    assert _parse(1,   "auto") == (1,)

    p, cls_dict = _check_single_val(1,1)
    p, cls_dict = _check_single_val(2,1)
    p, cls_dict = _check_single_val(7,6)
    
    return p
    

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

def test_predict_dense_sparse(model2d):
    model = model2d
    img, mask = real_image2d()
    x = normalize(img, 1, 99.8)
    labels1, res1 = model.predict_instances(x, n_tiles=(2, 2), sparse = False)
    labels2, res2 = model.predict_instances(x, n_tiles=(2, 2), sparse = True)
    assert np.allclose(labels1, labels2)
    assert all(np.allclose(res1[k], res2[k]) for k in set(res1.keys()).union(set(res2.keys())) if isinstance(res1[k], np.ndarray))
    return labels2, res1, labels2, res2



def test_speed(model2d):
    from time import time
    
    model = model2d
    img, mask = real_image2d()
    x = normalize(img, 1, 99.8)
    x = np.tile(x,(6,6))
    print(x.shape)
    
    stats = []
    
    for mode, n_tiles,sparse in product(("normal", "big"),(None,(2,2)),(True, False)):
           t = time()
           if mode=="normal":
               labels, res = model.predict_instances(x, n_tiles=n_tiles, sparse = sparse)
           else:
               labels, res = model.predict_instances_big(x,axes = "YX",
                                                         block_size = 1024+256,
                                                         context = 64, min_overlap = 64,
                                                         n_tiles=n_tiles, sparse = sparse)
               
           t = time()-t
           s = f"mode={mode}\ttiles={n_tiles}\tsparse={sparse}\t{t:.2f}s"
           print(s)
           stats.append(s)
    
    for s in stats:
        print(s) 


def render_label_pred_example2(model2d):
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


def test_pretrained_integration():
    from stardist.models import StarDist2D
    img = normalize(real_image2d()[0])

    model = StarDist2D.from_pretrained("2D_versatile_fluo")
    prob,dist = model.predict(img)

    y1, res1 = model._instances_from_prediction(img.shape,prob,dist, nms_thresh=.3)
    y2, res2 = model._instances_from_prediction_old(img.shape,prob,dist, nms_thresh=.3)

    # reorder as polygons is inverted in newer versions
    res2 = dict((k,v[::-1]) for k,v in res2.items())        
    y2[y2>0] = np.max(y2)-y2[y2>0]+1

    for k in res1.keys():
        if isinstance(res1[k], np.ndarray):
            assert np.allclose(res1[k],res2[k])
        
    assert np.allclose(y1,y2)
    
    return y1, res1, y2, res2



# this test has to be at the end of the model
def test_load_and_export_TF(model2d):
    model = model2d
    assert any(g>1 for g in model.config.grid)
    # model.export_TF(single_output=False, upsample_grid=False)
    # model.export_TF(single_output=False, upsample_grid=True)
    model.export_TF(single_output=True, upsample_grid=False)
    model.export_TF(single_output=True, upsample_grid=True)
    
if __name__ == '__main__':
    from conftest import _model2d
    # test_speed(_model2d())
    _test_model_multiclass(n_classes = 1, classes = "auto", n_channel = None, basedir = None)
    

    
    
