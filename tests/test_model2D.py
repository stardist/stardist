import sys
import numpy as np
import pytest
from pathlib import Path
import tensorflow as tf
from itertools import product
from stardist.data import test_image_nuclei_2d, test_image_he_2d
from stardist.models import Config2D, StarDist2D, StarDistData2D
from stardist.matching import matching
from stardist.utils import export_imagej_rois
from stardist.plot import render_label, render_label_pred
from csbdeep.utils import normalize
from utils import circle_image, path_model2d, NumpySequence, Timer


@pytest.mark.parametrize('n_rays, grid, n_channel, workers, use_sequence', [(17, (1, 1), None, 1, False), (32, (2, 4), 1, 1, False), (4, (8, 2), 2, 1, True)])
def test_model(tmpdir, n_rays, grid, n_channel, workers, use_sequence):
    img = circle_image(shape=(160, 160))
    imgs = np.repeat(img[np.newaxis], 3, axis=0)

    if n_channel is not None:
        imgs = np.repeat(imgs[..., np.newaxis], n_channel, axis=-1)
    else:
        n_channel = 1

    X = imgs+.6*np.random.uniform(0, 1, imgs.shape)
    Y = (imgs if imgs.ndim == 3 else imgs[..., 0]).astype(int)


    if use_sequence:
        X, Y = NumpySequence(X), NumpySequence(Y)

    conf = Config2D(
        n_rays=n_rays,
        grid=grid,
        n_channel_in=n_channel,
        use_gpu=False,
        train_epochs=2,
        train_steps_per_epoch=1,
        train_batch_size=2,
        train_loss_weights=(4, 1),
        train_patch_size=(128, 128),
        train_sample_cache = not use_sequence
    )

    model = StarDist2D(conf, name='stardist', basedir=str(tmpdir))
    model.train(X, Y, validation_data=(X[:2], Y[:2]), workers=workers)
    ref = model.predict(X[0])
    res = model.predict(X[0], n_tiles=(
        (2, 3) if X[0].ndim == 2 else (2, 3, 1)))

    # deactivate as order of labels might not be the same
    # assert all(np.allclose(u,v) for u,v in zip(ref,res))

    return model


def test_foreground_warning():
    # ask to train only with foreground patches when there are none
    # include a constant label image that must trigger a warning
    conf = Config2D(
        n_rays=32,
        train_patch_size=(96, 96),
        train_foreground_only = 1,
        train_steps_per_epoch = 1,
        train_epochs=1,
        train_batch_size=2,
    )
    X,Y = np.ones((2,100,100), np.float32), np.ones((2,100,100),np.uint16)

    with pytest.warns(UserWarning):
        StarDist2D(conf, None, None).train(
            X, Y, validation_data=(X[-1:], Y[-1:]))


def test_load_and_predict(model2d):
    model = model2d
    img, mask = test_image_nuclei_2d(return_mask=True)
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
    assert (stats.fp, stats.tp, stats.fn) == (5, 114, 11)
    return labels

def test_load_and_predict_big():
    model_path = path_model2d()
    model = StarDist2D(None, name=model_path.name,
                       basedir=str(model_path.parent))
    img = test_image_nuclei_2d()
    x = normalize(img, 1, 99.8)
    x = np.tile(x,(4,4))
    labels1, polygons1 = model.predict_instances(x)
    labels2, polygons2 = model.predict_instances(x, n_tiles=(4,4))
    assert np.allclose(labels1>0, labels2>0)
    return labels1


def test_optimize_thresholds(model2d):
    model = model2d
    img, mask = test_image_nuclei_2d(return_mask=True)
    x = normalize(img, 1, 99.8)

    res = model.optimize_thresholds([x], [mask],
                              nms_threshs=[.3, .5],
                              iou_threshs=[.3, .5],
                              optimize_kwargs=dict(tol=1e-1),
                              save_to_json=False)

    np.testing.assert_almost_equal(res["prob"], 0.549501654040, decimal=3)
    np.testing.assert_almost_equal(res["nms"] , 0.5, decimal=3)


@pytest.mark.parametrize('n_classes, classes', [(None,(1,1)),(2,(1,2))])
@pytest.mark.parametrize('shape_completion',(False,True))
def test_stardistdata(shape_completion, n_classes, classes):
    np.random.seed(42)
    from stardist.models import StarDistData2D
    img, mask = test_image_nuclei_2d(return_mask=True)
    s = StarDistData2D([img, img], [mask, mask],
                       grid = (2,2),
                       n_classes = n_classes, classes = classes,
                       shape_completion = shape_completion, b = 8,
                       batch_size=1, patch_size=(30, 40), n_rays=32, length=1)
    a, b = s[0]
    return a,b, s


def _edt_available():
    try:
        from edt import edt
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _edt_available(), reason="needs edt")
@pytest.mark.parametrize('anisotropy',(None,(3.2,0.75)))
def test_edt_prob(anisotropy):
    try:
        import edt
        from stardist.utils import _edt_prob_edt, _edt_prob_scipy

        masks = (np.tile(test_image_nuclei_2d(return_mask=True)[1],(2,2)),
                 np.zeros((117,92)),
                 np.ones((153,112)))
        dtypes = (np.uint16, np.int32)
        slices = (slice(None),)*2, (slice(1,-1),)*2
        for mask, dtype, sl in product(masks, dtypes, slices):
            mask = mask.astype(dtype)[sl]
            print(f"\nEDT {dtype.__name__} {mask.shape} slice {sl} ")
            with Timer("scipy "):
                ed1 = _edt_prob_scipy(mask, anisotropy=anisotropy)
            with Timer("edt:  "):
                ed2 = _edt_prob_edt(mask, anisotropy=anisotropy)
            assert np.percentile(np.abs(ed1-ed2), 99.9) < 1e-3

    except ImportError:
        print("Install edt to run test")


def render_label_example(model2d):
    model = model2d
    img, y_gt = test_image_nuclei_2d(return_mask=True)
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
    img, y_gt = test_image_nuclei_2d(return_mask=True)
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
    img, mask = test_image_nuclei_2d(return_mask=True)
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

    X = np.zeros((2,100,100), np.uint16)
    X[:, 10:-10,10:-10] = 1

    X,Y = NumpySequence(X), NumpySequence(X.astype(np.uint16))
    s = StarDistData2D(X,Y,
                       batch_size=1, patch_size=(100,100), n_rays=32, length=1)
    (img,), (prob, dist) = s[0]
    return (img,), (prob, dist), s


def test_stardistdata_multithreaded(workers=5):
    np.random.seed(42)
    from stardist.models import StarDistData2D
    from scipy.ndimage import rotate
    from concurrent.futures import ThreadPoolExecutor
    from time import sleep

    def augmenter(x,y):
        deg = int(np.sum(y)%117)
        print(deg)
        # return x,y
        x = rotate(x, deg, reshape=False, order=0)
        y = rotate(y, deg, reshape=False, order=0)
        # sleep(np.abs(deg)/180)
        return x,y

    n_samples = 4

    _ , mask = test_image_nuclei_2d(return_mask=True)
    Y = np.stack([mask+i for i in range(n_samples)])
    s = StarDistData2D(Y.astype(np.float32), Y,
                       grid = (1,1),
                       n_classes = None, augmenter=augmenter,
                       batch_size=1, patch_size=mask.shape, n_rays=32, length=len(Y))

    a1, b1 = tuple(zip(*tuple(s[i] for i in range(n_samples))))

    with ThreadPoolExecutor(max_workers=n_samples) as e:
        a2, b2 = tuple(zip(*tuple(e.map(lambda i: s[i], range(n_samples)))))

    assert all([np.allclose(_r1[0], _r2[0]) for _r1,_r2 in zip(a1, a2)])
    assert all([np.allclose(_r1[0], _r2[0]) for _r1,_r2 in zip(b1, b2)])
    assert all([np.allclose(_r1[1], _r2[1]) for _r1,_r2 in zip(b1, b2)])
    return a2, b2, s



def test_imagej_rois_export(tmpdir, model2d):
    img = normalize(test_image_nuclei_2d(), 1, 99.8)
    labels, polys = model2d.predict_instances(img)
    export_imagej_rois(str(Path(tmpdir)/'img_rois.zip'), polys['coord'])




def _test_model_multiclass(n_classes = 1, classes = "auto", n_channel = None, basedir = None):
    from skimage.measure import regionprops
    img, mask = test_image_nuclei_2d(return_mask=True)
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

    if n_classes is not None and n_classes>1 and classes=="auto":
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

    s = model.train(X, Y, classes = classes, epochs = 30,
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

    return model, img, res1, res2, res3

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
        img, y_gt = test_image_nuclei_2d(return_mask=True)
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
    img, mask = test_image_nuclei_2d(return_mask=True)
    x = normalize(img, 1, 99.8)
    labels1, res1 = model.predict_instances(x, n_tiles=(2, 2), sparse = False)
    labels2, res2 = model.predict_instances(x, n_tiles=(2, 2), sparse = True)
    assert np.allclose(labels1, labels2)
    assert all(np.allclose(res1[k], res2[k]) for k in set(res1.keys()).union(set(res2.keys())) if isinstance(res1[k], np.ndarray))
    return labels2, res1, labels2, res2


def test_speed(model2d):
    from time import time

    model = model2d
    img, mask = test_image_nuclei_2d(return_mask=True)
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
    img, y_gt = test_image_nuclei_2d(return_mask=True)
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
    img = normalize(test_image_nuclei_2d())

    model = StarDist2D.from_pretrained("2D_versatile_fluo")
    prob,dist = model.predict(img)

    y1, res1 = model._instances_from_prediction(img.shape,prob,dist, nms_thresh=.3)
    return y1, res1

    # y2, res2 = model._instances_from_prediction_old(img.shape,prob,dist, nms_thresh=.3)

    # # reorder as polygons is inverted in newer versions
    # res2 = dict((k,v[::-1]) for k,v in res2.items())
    # y2[y2>0] = np.max(y2)-y2[y2>0]+1

    # for k in res1.keys():
    #     if isinstance(res1[k], np.ndarray):
    #         assert np.allclose(res1[k],res2[k])

    # assert np.allclose(y1,y2)

    # return y1, res1, y2, res2


@pytest.mark.parametrize('scale', (0.5, 2.0, (0.34, 1.47)))
@pytest.mark.parametrize('mode', ('fluo', 'he'))
def test_predict_with_scale(scale, mode):
    from scipy.ndimage import zoom
    if np.isscalar(scale):
        scale = (scale,scale)
    if mode=='fluo':
        model = StarDist2D.from_pretrained('2D_versatile_fluo')
        x = test_image_nuclei_2d()
        _scale = tuple(scale)
    elif mode=='he':
        model = StarDist2D.from_pretrained('2D_versatile_he')
        x = test_image_he_2d()
        _scale = tuple(scale) + (1,)
    else:
        raise ValueError(mode)

    x = normalize(x)
    x = zoom(x, (0.5,0.5) if x.ndim==2 else (0.5,0.5,1), order=1) # to speed test up
    x_scaled = zoom(x, _scale, order=1)
    
    labels,        res        = model.predict_instances(x, scale=_scale)
    labels_scaled, res_scaled = model.predict_instances(x_scaled)

    assert x.shape[:2] == labels.shape
    assert np.allclose(res['points'] * np.asarray(scale).reshape(1,2),   res_scaled['points'])
    assert np.allclose(res['coord']  * np.asarray(scale).reshape(1,2,1), res_scaled['coord'])
    assert np.allclose(res['prob'], res_scaled['prob'])

    return x, labels


def test_tfdata():
    np.random.seed(42)
    from stardist.models import StarDistData2D
    from stardist.models.tfdata_wrapper import wrap_stardistdata_as_tfdata
    from time import sleep

    def augmenter(x,y):
        return x,y

    n_samples = 4

    _ , mask = test_image_nuclei_2d(return_mask=True)
    Y = np.stack([mask+i for i in range(n_samples)])
    s = StarDistData2D(Y.astype(np.float32), Y,
                       grid = (1,1),
                       n_classes = None, augmenter=augmenter,
                       batch_size=1, patch_size=mask.shape, n_rays=32, length=len(Y))

    data = wrap_stardistdata_as_tfdata(s, shuffle=True, num_parallel_calls=2, verbose=True)

    return data


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
    # _test_model_multiclass(n_classes = 1, classes = "auto", n_channel = None, basedir = None)
    # a,b,s = test_stardistdata_multithreaded()
    # test_model("foo", 32, (1,1), None, 4)

    # test_foreground_warning()

    # model = test_model("tmpdir", 32, (2, 2), 1, False, 1)

    # test_load_and_export_TF(model)

    # test_predict_dense_sparse(_model2d())

    data = test_tfdata()
    
