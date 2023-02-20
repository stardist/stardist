import warnings
from collections.abc import Iterable
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from augmend.transforms import BaseTransform
from augmend.utils import _validate_rng
from csbdeep.utils import _raise
from skimage.draw import polygon
from skimage.measure import regionprops
from skimage.transform import resize
from sklearn.decomposition import NMF
from tqdm.auto import tqdm

CLASS_NAMES = {
    0: "BACKGROUND",
    1: "Neutrophil",
    2: "Epithelial",
    3: "Lymphocyte",
    4: "Plasma",
    5: "Eosinophil",
    6: "Connective",
}


### DATA


def cls_dict_from_label(y, y_class):
    return dict(
        (r.label, int(np.median(y_class[r.slice][y[r.slice] == r.label]))) for r in regionprops(y)
    )


def get_data(path, n=None, shuffle=True, normalize=True, seed=None):
    rng = np.random if seed is None else np.random.RandomState(seed)

    path = Path(path)
    X = np.load(path / "images.npy")
    Y0 = np.load(path / "labels.npy")
    assert len(X) == len(Y0)

    idx = np.arange(len(X))
    if shuffle:
        rng.shuffle(idx)
    idx = idx[:n]

    X = X[idx]
    Y0 = Y0[idx]

    if normalize:
        X = (X / 255).astype(np.float32)

    Y = Y0[..., 0]
    D = np.array(
        [
            cls_dict_from_label(y, y_class)
            for y, y_class in tqdm(zip(Y0[..., 0], Y0[..., 1]), total=len(Y0))
        ]
    )

    return X, Y, D, Y0, idx


def oversample_classes(X, Y, D, Y0, idx, n_extra_classes=4, seed=None):
    rng = np.random if seed is None else np.random.RandomState(seed)

    # get the most infrequent classes
    class_counts = np.bincount(Y0[:, ::4, ::4, 1].ravel(), minlength=len(CLASS_NAMES))
    extra_classes = np.argsort(class_counts)[:n_extra_classes]
    all(
        class_counts[c] > 0 or _raise(f"count 0 for class {c} ({CLASS_NAMES[c]})")
        for c in extra_classes
    )

    # how many extra samples (more for infrequent classes)
    n_extras = np.sqrt(np.sum(class_counts[1:]) / class_counts[extra_classes])
    n_extras = n_extras / np.max(n_extras)
    print("oversample classes", extra_classes)
    idx_take = np.arange(len(X))

    for c, n_extra in zip(extra_classes, n_extras):
        # oversample probability is ~ number of instances
        prob = np.sum(Y0[:, ::2, ::2, 1] == c, axis=(1, 2))
        prob = np.clip(prob, 0, np.percentile(prob, 99.8))
        prob = prob ** 2
        # prob[prob<np.percentile(prob,90)] = 0
        prob = prob / np.sum(prob)
        n_extra = int(n_extra * len(X))
        print(f"adding {n_extra} images of class {c} ({CLASS_NAMES[c]})")
        idx_extra = rng.choice(np.arange(len(X)), n_extra, p=prob)
        idx_take = np.append(idx_take, idx_extra)

    X, Y, D, Y0, idx = map(lambda x: x[idx_take], (X, Y, D, Y0, idx))
    return X, Y, D, Y0, idx


### AUGMENTATIONS


def _assert_uint8_image(x):
    assert x.ndim == 3 and x.shape[-1] == 3 and x.dtype.type is np.uint8


def rgb_to_density(x):
    _assert_uint8_image(x)
    x = np.maximum(x, 1)
    return np.maximum(-1 * np.log(x / 255), 1e-6)


def density_to_rgb(x):
    return np.clip(255 * np.exp(-x), 0, 255).astype(np.uint8)


def rgb_to_lab(x):
    _assert_uint8_image(x)
    return cv2.cvtColor(x, cv2.COLOR_RGB2LAB)


def lab_to_rgb(x):
    _assert_uint8_image(x)
    return cv2.cvtColor(x, cv2.COLOR_LAB2RGB)


def extract_stains(x, subsample=128, l1_reg=0.001, tissue_threshold=200):
    """Non-negative matrix factorization 
    
    Let x be the image as optical densities with shape (N,3) 

    then we want to decompose it as 

    x = W * H 

    with
        W: stain values of shape (N, 2)
        H: staining matrix of shape (2, 3) 
        
    Solve it as 
    
    min (x - W * H)^2 + |H|_1 

    with additonal sparsity prior on the stains W 
    """
    _assert_uint8_image(x)

    model = NMF(
        n_components=2, init="random", random_state=0, alpha_W=l1_reg, alpha_H=0, l1_ratio=1
    )

    # optical density
    density = rgb_to_density(x)

    # only select darker regions
    tissue_mask = rgb_to_lab(x)[..., 0] < tissue_threshold

    values = density[tissue_mask]

    # compute stain matrix on subsampled values (way faster)
    model.fit(values[::subsample])

    H = model.components_

    # normalize rows
    H = H / np.linalg.norm(H, axis=1, keepdims=True)
    if H[0, 0] < H[1, 0]:
        H = H[[1, 0]]

    # get stains on full image
    Hinv = np.linalg.pinv(H)
    stains = density.reshape((-1, 3)) @ Hinv
    stains = stains.reshape(x.shape[:2] + (2,))

    return H, stains


def stains_to_rgb(stains, stain_matrix):
    assert stains.ndim == 3 and stains.shape[-1] == 2
    assert stain_matrix.shape == (2, 3)
    return density_to_rgb(stains @ stain_matrix)


def augment_stains(x, amount_matrix=0.2, amount_stains=0.2, n_samples=1, subsample=128, rng=None):
    """ 
    create stain color augmented versions of x by 
    randomly perturbing the stain matrix by given amount

    1) extract stain matrix M and associated stains
    2) add uniform random noise (+- scale) to stain matrix
    3) reconstruct image 
    """
    _assert_uint8_image(x)
    if rng is None:
        rng = np.random

    M, stains = extract_stains(x, subsample=subsample)

    M = np.expand_dims(M, 0) + amount_matrix * rng.uniform(-1, 1, (n_samples, 2, 3))
    M = np.maximum(M, 0)

    stains = np.expand_dims(stains, 0) * (
        1 + amount_stains * rng.uniform(-1, 1, (n_samples, 1, 1, 2))
    )
    stains = np.maximum(stains, 0)

    if n_samples == 1:
        return stains_to_rgb(stains[0], M[0])
    else:
        return np.stack(tuple(stains_to_rgb(s, m) for s, m in zip(stains, M)), 0)


class HEStaining(BaseTransform):
    """HE staining augmentations"""

    @staticmethod
    def _augment(x, rng, amount_matrix, amount_stains):
        rng = _validate_rng(rng)
        x_rgb = (255 * np.clip(x, 0, 1)).astype(np.uint8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = augment_stains(
                    x_rgb,
                    amount_matrix=amount_matrix,
                    amount_stains=amount_stains,
                    subsample=128,
                    n_samples=1,
                    rng=rng,
                )
            except:
                res = x_rgb
        return (res / 255).astype(np.float32)

    def __init__(self, amount_matrix=0.15, amount_stains=0.4):
        super().__init__(
            default_kwargs=dict(amount_matrix=amount_matrix, amount_stains=amount_stains),
            transform_func=self._augment,
        )


class HueBrightnessSaturation(BaseTransform):
    """apply affine intensity shift where background is bright"""

    @staticmethod
    def hbs_adjust(x, rng, hue, brightness, saturation):
        def _prep(s, negate=True):
            s = (-s if negate else s, s) if np.isscalar(s) else tuple(s)
            assert len(s) == 2
            return s

        hue = _prep(hue)
        brightness = _prep(brightness)
        saturation = _prep(saturation, False)
        assert x.ndim == 3 and x.shape[-1] == 3
        rng = _validate_rng(rng)
        h_hue = rng.uniform(*hue)
        h_brightness = rng.uniform(*brightness)
        h_saturation = rng.uniform(*saturation)
        x = tf.image.adjust_hue(x, h_hue)
        x = tf.image.adjust_brightness(x, h_brightness)
        x = tf.image.adjust_saturation(x, h_saturation)
        return x.numpy()

    def __init__(self, hue=0.1, brightness=0, saturation=1):
        """
        hue:        add +- value
        brightness: add +- value
        saturation: multiply by value 
        -> set hue=0, brightness=0, saturation=1 for no effect
        """
        super().__init__(
            default_kwargs=dict(hue=hue, brightness=brightness, saturation=saturation),
            transform_func=self.hbs_adjust,
        )


### PREDICTION


def refine(labels, polys, thr=0.5, w_winner=2, progress=False):
    """shape refinement"""
    thr = float(thr)
    assert 0 <= thr <= 1, f"required: 0 <= {thr} <= 1"
    if thr == 1:
        # to include only pixels where all polys agree
        # because we take mask > thr below
        thr -= np.finfo(float).eps
    nms = polys["nms"]
    obj_ind = np.flatnonzero(nms["suppressed"] == -1)
    assert np.allclose(nms["scores"][obj_ind], sorted(nms["scores"][obj_ind])[::-1])
    mask = np.zeros_like(labels)
    # mask_soft = np.zeros_like(labels, float)

    # TODO: use prob/scores for weighting?
    # TODO: use mask that weights pixels on distance to poly boundary?
    for k, i in tqdm(
        zip(range(len(obj_ind), 0, -1), reversed(obj_ind)),
        total=len(obj_ind),
        disable=(not progress),
    ):
        polys_i = nms["coord"][i : i + 1]  # winner poly after nms
        polys_i_suppressed = nms["coord"][nms["suppressed"] == i]  # suppressed polys by winner
        # array of all polys (first winner, then all suppressed)
        polys_i = np.concatenate([polys_i, polys_i_suppressed], axis=0)
        # bounding slice around all polys wrt image
        ss = tuple(
            slice(max(int(np.floor(start)), 0), min(int(np.ceil(stop)), w))
            for start, stop, w in zip(
                np.min(polys_i, axis=(0, 2)), np.max(polys_i, axis=(0, 2)), labels.shape
            )
        )
        # shape of image crop/region that contains all polys
        shape_i = tuple(s.stop - s.start for s in ss)
        # offset of image region
        offset = np.array([s.start for s in ss]).reshape(2, 1)
        # voting weights for polys
        n_i = len(polys_i)
        # vote weight of winning poly (1 = same vote as each suppressed poly)
        weight_winner = w_winner
        # define and normalize weights for all polys
        polys_i_weights = np.ones(n_i)
        polys_i_weights[0] = weight_winner
        # polys_i_weights = np.array([weight_winner if j==0 else max(0,n_i-weight_winner)/(n_i-1) for j in range(n_i)])
        polys_i_weights = polys_i_weights / np.sum(polys_i_weights)
        # display(polys_i_weights)
        assert np.allclose(np.sum(polys_i_weights), 1)
        # merge by summing weighted poly masks
        mask_i = np.zeros(shape_i, float)
        for p, w in zip(polys_i, polys_i_weights):
            ind = polygon(*(p - offset), shape=shape_i)
            mask_i[ind] += w
        # write refined shape for instance i back to new label image
        # refined shape are all pixels with accumulated votes >= threshold
        mask[ss][mask_i > thr] = k
        # mask_soft[ss][mask_i>0] += mask_i[mask_i>0]

    return mask  # , mask_soft


def rot90(x, k=1, roll=True):
    """Rotate stardist cnn predictions by multiples of 90 degrees."""
    from stardist import ray_angles

    k = (k + 4) % 4
    # print(k)
    assert x.ndim in (2, 3)
    if x.ndim == 2 or roll == False:
        # rotate 2D image or 2D+channel
        return np.rot90(x, k)
    # dist image has radial distances as 3rd dimension
    # -> need to roll values
    deg_roll = (-90 * k) % 360
    rad_roll = np.deg2rad(deg_roll)
    # print(deg_roll)
    n_rays = x.shape[2]
    rays = ray_angles(n_rays)
    n_roll = [i for i, v in enumerate(rays) if np.isclose(v, rad_roll)]
    assert len(n_roll) == 1, (rays, rad_roll)
    n_roll = n_roll[0]
    z = np.rot90(x, k, axes=(0, 1))  # rotate spatial axes
    z = np.roll(z, n_roll, axis=2)  # roll polar axis
    return z


def flip(x, doit=True, reverse=True):
    """Flip stardist cnn predictions."""
    assert x.ndim in (2, 3)
    if not doit:
        return x
    if x.ndim == 2 or reverse == False:
        return np.flipud(x)
    # dist image has radial distances as 3rd dimension
    # -> need to reverse values
    z = np.flipud(x)
    z = np.concatenate((z[..., 0:1], z[..., :0:-1]), axis=-1)
    return z


def crop_center(x, crop_shape):
    """Crop an array at the centre with specified dimensions."""
    orig_shape = x.shape
    h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
    w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
    x = x[h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    return x


def count_classes(y, classes=range(1, 7), crop=(224, 224)):
    assert y.ndim == 3 and y.shape[-1] == 2
    if crop is not None:
        y = crop_center(y, crop)
    return tuple(len(np.unique(y[..., 0] * (y[..., 1] == i))) - 1 for i in classes)


def predict(
    model,
    x,
    normalize=False,
    aggregate_prob_class=True,
    no_background_class=True,
    crop_counts=True,
    test_time_augment=False,
    refine_shapes=False,
    tta_merge=dict(prob=np.median, dist=np.mean, prob_class=np.mean),
    return_details=False,
    kwargs_instances=None,
    **kwargs,
):
    """ main prediction function to be used from several parts of the code

    Accepts a list of models for ensemble prediction

    model: StardDist2D model (or list of models)
    aggregate_prob_class: take the mean of prob_class per instance (instead of single point)
    no_background_class:  ignore background class in prob_class
    crop_counts:          crop labels to 224x224 before counting        
    test_time_augment:    True or maximal number of tta per model (=8 if True)
    refine_shapes:        False or dict of refine kwargs (empty dict for default values)
    Returns 
    -------
    u, counts 

    u:        class map arrays of shape (256,256,2) 
    counts:   cell counts,  tuple of length 6

    """
    from skimage.morphology import remove_small_objects

    if kwargs_instances is None:
        kwargs_instances = {}

    # model should be a list of models
    if not isinstance(model, Iterable):
        model = (model,)

    # if more than one model is given, ensure all have same outputs
    if not all([m.keras_model.output_shape == model[0].keras_model.output_shape for m in model]):
        raise ValueError(
            "Cannot combine model output shapes (models have different number of rays?)"
        )

    # use average thresholds of the models
    kwargs_instances.setdefault("nms_thresh", np.mean([m.thresholds.nms for m in model]))
    kwargs_instances.setdefault("prob_thresh", np.mean([m.thresholds.prob for m in model]))

    warnings.warn(f"{kwargs_instances}")

    if test_time_augment == False:
        # print('no tta')
        # no augmentation
        augs = ((0, False),)
        model_augs = tuple(product(model, augs))
    elif test_time_augment in (True, -1):
        # print('full tta')
        # 8 augmentations (4 rotations x 2 flips)
        augs = tuple(product((0, 1, 2, 3), (False, True)))
        model_augs = tuple(product(model, augs))
    else:
        # print(f'partial tta {test_time_augment}')
        augs = tuple(product((0, 1, 2, 3), (False, True)))

        # augs = augs[:test_time_augment]
        # model_augs  = tuple(product(model, augs ))

        # picking random flip/roations
        # rng = np.random.RandomState(42)
        # aug_idx = tuple(rng.choice(np.arange(len(augs)), min(test_time_augment, len(augs)), replace=False) for _ in model)

        aug_idx = (([1, 5, 0]), ([3, 7, 0]), ([0, 6, 3]), ([5, 2, 7]))
        model_augs = tuple((m, augs[i]) for m, idx in zip(model, aug_idx) for i in idx)

    warnings.warn(f"combining {len(model)} models -> total {len(model_augs)} predictions")

    def _preprocess(x):
        if normalize:
            x = x.astype(np.float32) / 255
        return x

    prob, dist, prob_class = zip(
        *[
            m.predict(flip(rot90(_preprocess(x), k, False), f, False), **kwargs)
            for m, (k, f) in model_augs
        ]
    )

    # undo augmentations for predictions
    prob = [rot90(flip(v, f), -k) for (m, (k, f)), v in zip(model_augs, prob)]
    dist = [rot90(flip(v, f), -k) for (m, (k, f)), v in zip(model_augs, dist)]
    prob_class = [
        rot90(flip(v, f, False), -k, False) for (m, (k, f)), v in zip(model_augs, prob_class)
    ]

    # merge predictions
    prob = tta_merge["prob"](np.stack(prob), axis=0)
    dist = tta_merge["dist"](np.stack(dist), axis=0)
    prob_class = tta_merge["prob_class"](np.stack(prob_class), axis=0)
    prob_class /= np.sum(prob_class, axis=-1, keepdims=True)

    u, res = model[0]._instances_from_prediction(
        x.shape[:2], prob, dist, prob_class=prob_class, **kwargs_instances
    )

    if refine_shapes is not False:
        assert "nms" in res
        u = refine(u, res, **refine_shapes)

    u_cls = np.zeros(u.shape, np.uint16)

    n_objects = len(res["prob"])

    cls = dict(zip(range(1, n_objects + 1), res["class_id"]))

    # u = remove_small_objects(u,10)

    if any(g > 1 for g in model[0].config.grid):
        prob = resize(prob, u.shape, order=1)
        prob_class = resize(prob_class, u.shape + prob_class.shape[-1:], order=1)

    if aggregate_prob_class:
        # take the sum of class probabilities
        pc_weighted = np.expand_dims(prob, -1) * prob_class
        if no_background_class:
            pc_weighted[..., 0] = -1

        for r in regionprops(u):
            m = u[r.slice] == r.label
            class_id = np.argmax(np.sum(pc_weighted[r.slice][m], 0))
            u_cls[r.slice][m] = class_id
            cls[r.label] = class_id
    else:
        # only take center class prob
        for r in regionprops(u):
            m = u[r.slice] == r.label
            u_cls[r.slice][m] = cls[r.label]

    out = np.stack([u, u_cls], axis=-1)

    class_count = count_classes(out, classes=range(1, 7), crop=(224, 224) if crop_counts else None)

    if return_details:
        return out, class_count, cls, prob_class
    else:
        return out, class_count
