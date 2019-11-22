import sys
import numpy as np
import pytest
from stardist.models import Config2D, StarDist2D
from stardist.matching import matching
from tifffile import imread
from csbdeep.utils import normalize
from utils import circle_image, real_image2d, path_model2d, prob_dist_image2d


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
    return model



@pytest.mark.parametrize('affinity', [False,True])
def test_optimize_thresholds(affinity):
    model_path = path_model2d()
    model = StarDist2D(None, name=model_path.name, basedir=str(model_path.parent))
    img, mask = real_image2d()
    x = normalize(img,1,99.8)
    model.optimize_thresholds([x],[mask],
                                    nms_threshs = [.3,.5],
                                    iou_threshs = [.3,.5],
                                    affinity = affinity,
                                    optimize_kwargs = dict(tol=1e-1),
                                    save_to_json = False)
    return model


def test_affinity():
    img, prob, dist = prob_dist_image2d()
    
    conf = Config2D (n_rays = dist.shape[-1], grid =(2,2))
    
    model = StarDist2D(conf, name=None, basedir=None)

    labels0, _ = model._instances_from_prediction(img.shape, prob, dist,
                                                 affinity=False, affinity_thresh=.01)

    labels, _ = model._instances_from_prediction(img.shape, prob, dist,
                                                 affinity=True, affinity_thresh=.01)
    
    return labels0, labels

@pytest.mark.parametrize('n_channel', (None,2))
def test_stardistdata(n_channel):
    from stardist.models import StarDistData2D
    img, mask = real_image2d()

    if n_channel is not None:
        img = np.repeat(img[...,np.newaxis], n_channel, axis=-1)

    s = StarDistData2D([img, img],[mask, mask], batch_size = 2, patch_size=(30,40),
                       n_rays =22)
    (img,mask),(prob, dist) = s[0]
    return (img,mask),(prob, dist), s
     

if __name__ == '__main__':
    # labels0, labels = test_affinity()
    (img,mask),(prob, dist), s = test_stardistdata(2)
