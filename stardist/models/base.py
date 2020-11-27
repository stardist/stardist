from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
import sys
import warnings
import math
from tqdm import tqdm
from collections import namedtuple
from pathlib import Path

from csbdeep.models.base_model import BaseModel
from csbdeep.utils.tf import export_SavedModel, keras_import, IS_TF_1, CARETensorBoard

import tensorflow as tf
K = keras_import('backend')
Sequence = keras_import('utils', 'Sequence')
Adam = keras_import('optimizers', 'Adam')
ReduceLROnPlateau, TensorBoard = keras_import('callbacks', 'ReduceLROnPlateau', 'TensorBoard')

from csbdeep.utils import _raise, backend_channels_last, axes_check_and_normalize, axes_dict, load_json, save_json
from csbdeep.internals.predict import tile_iterator, total_n_tiles
from csbdeep.internals.train import RollingSequence
from csbdeep.data import Resizer

from ..sample_patches import get_valid_inds
from ..utils import _is_power_of_2,  _is_floatarray, optimize_threshold


# TODO: support (optional) classification of objects?
# TODO: helper function to check if receptive field of cnn is sufficient for object sizes in GT

def generic_masked_loss(mask, loss, weights=1, norm_by_mask=True, reg_weight=0, reg_penalty=K.abs):
    def _loss(y_true, y_pred):
        actual_loss = K.mean(mask * weights * loss(y_true, y_pred), axis=-1)
        norm_mask = (K.mean(mask) + K.epsilon()) if norm_by_mask else 1
        if reg_weight > 0:
            reg_loss = K.mean((1-mask) * reg_penalty(y_pred), axis=-1)
            return actual_loss / norm_mask + reg_weight * reg_loss
        else:
            return actual_loss / norm_mask
    return _loss

def masked_loss(mask, penalty, reg_weight, norm_by_mask):
    loss = lambda y_true, y_pred: penalty(y_true - y_pred)
    return generic_masked_loss(mask, loss, reg_weight=reg_weight, norm_by_mask=norm_by_mask)

# TODO: should we use norm_by_mask=True in the loss or only in a metric?
#       previous 2D behavior was norm_by_mask=False
#       same question for reg_weight? use 1e-4 (as in 3D) or 0 (as in 2D)?

def masked_loss_mae(mask, reg_weight=0, norm_by_mask=True):
    return masked_loss(mask, K.abs, reg_weight=reg_weight, norm_by_mask=norm_by_mask)

def masked_loss_mse(mask, reg_weight=0, norm_by_mask=True):
    return masked_loss(mask, K.square, reg_weight=reg_weight, norm_by_mask=norm_by_mask)

def masked_metric_mae(mask):
    def relevant_mae(y_true, y_pred):
        return masked_loss(mask, K.abs, reg_weight=0, norm_by_mask=True)(y_true, y_pred)
    return relevant_mae

def masked_metric_mse(mask):
    def relevant_mse(y_true, y_pred):
        return masked_loss(mask, K.square, reg_weight=0, norm_by_mask=True)(y_true, y_pred)
    return relevant_mse

def kld(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.mean(K.binary_crossentropy(y_true, y_pred) - K.binary_crossentropy(y_true, y_true), axis=-1)



class StarDistDataBase(RollingSequence):

    def __init__(self, X, Y, n_rays, grid, batch_size, patch_size, length, use_gpu=False, sample_ind_cache=True, maxfilter_patch_size=None, augmenter=None, foreground_prob=0):

        super().__init__(data_size=len(X), batch_size=batch_size, length=length, shuffle=True)

        if isinstance(X, (np.ndarray, tuple, list)):
            X = [x.astype(np.float32, copy=False) for x in X]

        # Y = [y.astype(np.uint16,  copy=False) for y in Y]

        # sanity checks
        assert len(X)==len(Y) and len(X)>0
        nD = len(patch_size)
        assert nD in (2,3)
        x_ndim = X[0].ndim
        assert x_ndim in (nD,nD+1)

        if isinstance(X, (np.ndarray, tuple, list)) and \
           isinstance(Y, (np.ndarray, tuple, list)):
            all(y.ndim==nD and x.ndim==x_ndim and x.shape[:nD]==y.shape for x,y in zip(X,Y)) or _raise(ValueError("images and masks should have corresponding shapes/dimensions"))
            all(x.shape[:nD]>=tuple(patch_size) for x in X) or _raise(ValueError("Some images are too small for given patch_size {patch_size}".format(patch_size=patch_size)))

        if x_ndim == nD:
            self.n_channel = None
        else:
            self.n_channel = X[0].shape[-1]
            assert all(x.shape[-1]==self.n_channel for x in X)
        assert 0 <= foreground_prob <= 1

        self.X, self.Y = X, Y
        # self.batch_size = batch_size
        self.n_rays = n_rays
        self.patch_size = patch_size
        self.ss_grid = (slice(None),) + tuple(slice(0, None, g) for g in grid)
        self.use_gpu = bool(use_gpu)
        if augmenter is None:
            augmenter = lambda *args: args
        callable(augmenter) or _raise(ValueError("augmenter must be None or callable"))
        self.augmenter = augmenter
        self.foreground_prob = foreground_prob

        if self.use_gpu:
            from gputools import max_filter
            self.max_filter = lambda y, patch_size: max_filter(y.astype(np.float32), patch_size)
        else:
            from scipy.ndimage.filters import maximum_filter
            self.max_filter = lambda y, patch_size: maximum_filter(y, patch_size, mode='constant')

        self.maxfilter_patch_size = maxfilter_patch_size if maxfilter_patch_size is not None else self.patch_size

        self.sample_ind_cache = sample_ind_cache
        self._ind_cache_fg  = {}
        self._ind_cache_all = {}


    def get_valid_inds(self, k, foreground_prob=None):
        if foreground_prob is None:
            foreground_prob = self.foreground_prob
        foreground_only = np.random.uniform() < foreground_prob
        _ind_cache = self._ind_cache_fg if foreground_only else self._ind_cache_all
        if k in _ind_cache:
            inds = _ind_cache[k]
        else:
            patch_filter = (lambda y,p: self.max_filter(y, self.maxfilter_patch_size) > 0) if foreground_only else None
            inds = get_valid_inds((self.Y[k],)+self.channels_as_tuple(self.X[k]), self.patch_size, patch_filter=patch_filter)
            if self.sample_ind_cache:
                _ind_cache[k] = inds
        if foreground_only and len(inds[0])==0:
            # no foreground pixels available
            return self.get_valid_inds(k, foreground_prob=0)
        return inds


    def channels_as_tuple(self, x):
        if self.n_channel is None:
            return (x,)
        else:
            return tuple(x[...,i] for i in range(self.n_channel))



class StarDistBase(BaseModel):

    def __init__(self, config, name=None, basedir='.'):
        super().__init__(config=config, name=name, basedir=basedir)
        threshs = dict(prob=None, nms=None)
        if basedir is not None:
            try:
                threshs = load_json(str(self.logdir / 'thresholds.json'))
                print("Loading thresholds from 'thresholds.json'.")
                if threshs.get('prob') is None or not (0 < threshs.get('prob') < 1):
                    print("- Invalid 'prob' threshold (%s), using default value." % str(threshs.get('prob')))
                    threshs['prob'] = None
                if threshs.get('nms') is None or not (0 < threshs.get('nms') < 1):
                    print("- Invalid 'nms' threshold (%s), using default value." % str(threshs.get('nms')))
                    threshs['nms'] = None
            except FileNotFoundError:
                if config is None and len(tuple(self.logdir.glob('*.h5'))) > 0:
                    print("Couldn't load thresholds from 'thresholds.json', using default values. "
                          "(Call 'optimize_thresholds' to change that.)")

        self.thresholds = dict (
            prob = 0.5 if threshs['prob'] is None else threshs['prob'],
            nms  = 0.4 if threshs['nms']  is None else threshs['nms'],
        )
        print("Using default values: prob_thresh={prob:g}, nms_thresh={nms:g}.".format(prob=self.thresholds.prob, nms=self.thresholds.nms))


    @property
    def thresholds(self):
        return self._thresholds


    @thresholds.setter
    def thresholds(self, d):
        self._thresholds = namedtuple('Thresholds',d.keys())(*d.values())


    def prepare_for_training(self, optimizer=None):
        """Prepare for neural network training.

        Compiles the model and creates
        `Keras Callbacks <https://keras.io/callbacks/>`_ to be used for training.

        Note that this method will be implicitly called once by :func:`train`
        (with default arguments) if not done so explicitly beforehand.

        Parameters
        ----------
        optimizer : obj or None
            Instance of a `Keras Optimizer <https://keras.io/optimizers/>`_ to be used for training.
            If ``None`` (default), uses ``Adam`` with the learning rate specified in ``config``.

        """
        if optimizer is None:
            optimizer = Adam(lr=self.config.train_learning_rate)

        masked_dist_loss = {'mse': masked_loss_mse, 'mae': masked_loss_mae}[self.config.train_dist_loss]
        prob_loss = 'binary_crossentropy'

        def split_dist_true_mask(dist_true_mask):
            return tf.split(dist_true_mask, num_or_size_splits=[self.config.n_rays,-1], axis=-1)

        def dist_loss(dist_true_mask, dist_pred):
            dist_true, dist_mask = split_dist_true_mask(dist_true_mask)
            return masked_dist_loss(dist_mask, reg_weight=self.config.train_background_reg)(dist_true, dist_pred)

        def relevant_mae(dist_true_mask, dist_pred):
            dist_true, dist_mask = split_dist_true_mask(dist_true_mask)
            return masked_metric_mae(dist_mask)(dist_true, dist_pred)

        def relevant_mse(dist_true_mask, dist_pred):
            dist_true, dist_mask = split_dist_true_mask(dist_true_mask)
            return masked_metric_mse(dist_mask)(dist_true, dist_pred)

        self.keras_model.compile(optimizer, loss=[prob_loss, dist_loss],
                                            loss_weights = list(self.config.train_loss_weights),
                                            metrics={'prob': kld, 'dist': [relevant_mae, relevant_mse]})

        self.callbacks = []
        if self.basedir is not None:
            self.callbacks += self._checkpoint_callbacks()

            if self.config.train_tensorboard:
                if IS_TF_1:
                    self.callbacks.append(CARETensorBoard(log_dir=str(self.logdir), prefix_with_timestamp=False, n_images=3, write_images=True, prob_out=False))
                else:
                    self.callbacks.append(TensorBoard(log_dir=str(self.logdir/'logs'), write_graph=False, profile_batch=0))

        if self.config.train_reduce_lr is not None:
            rlrop_params = self.config.train_reduce_lr
            if 'verbose' not in rlrop_params:
                rlrop_params['verbose'] = True
            # TF2: add as first callback to put 'lr' in the logs for TensorBoard
            self.callbacks.insert(0,ReduceLROnPlateau(**rlrop_params))

        self._model_prepared = True


    def predict(self, img, axes=None, normalizer=None, n_tiles=None, show_tile_progress=True, **predict_kwargs):
        """Predict.

        Parameters
        ----------
        img : :class:`numpy.ndarray`
            Input image
        axes : str or None
            Axes of the input ``img``.
            ``None`` denotes that axes of img are the same as denoted in the config.
        normalizer : :class:`csbdeep.data.Normalizer` or None
            (Optional) normalization of input image before prediction.
            Note that the default (``None``) assumes ``img`` to be already normalized.
        n_tiles : iterable or None
            Out of memory (OOM) errors can occur if the input image is too large.
            To avoid this problem, the input image is broken up into (overlapping) tiles
            that are processed independently and re-assembled.
            This parameter denotes a tuple of the number of tiles for every image axis (see ``axes``).
            ``None`` denotes that no tiling should be used.
        show_tile_progress: bool
            Whether to show progress during tiled prediction.
        predict_kwargs: dict
            Keyword arguments for ``predict`` function of Keras model.

        Returns
        -------
        (:class:`numpy.ndarray`,:class:`numpy.ndarray`)
            Returns the tuple (`prob`, `dist`) of per-pixel object probabilities and star-convex polygon/polyhedra distances.

        """
        if n_tiles is None:
            n_tiles = [1]*img.ndim
        try:
            n_tiles = tuple(n_tiles)
            img.ndim == len(n_tiles) or _raise(TypeError())
        except TypeError:
            raise ValueError("n_tiles must be an iterable of length %d" % img.ndim)
        all(np.isscalar(t) and 1<=t and int(t)==t for t in n_tiles) or _raise(
            ValueError("all values of n_tiles must be integer values >= 1"))
        if not _is_floatarray(img):
            warnings.warn("Predicting on non-float image ( forgot to normalize? )")

        n_tiles = tuple(map(int,n_tiles))

        axes     = self._normalize_axes(img, axes)
        axes_net = self.config.axes

        _permute_axes = self._make_permute_axes(axes, axes_net)
        x = _permute_axes(img) # x has axes_net semantics

        channel = axes_dict(axes_net)['C']
        self.config.n_channel_in == x.shape[channel] or _raise(ValueError())
        axes_net_div_by = self._axes_div_by(axes_net)

        grid = tuple(self.config.grid)
        len(grid) == len(axes_net)-1 or _raise(ValueError())
        grid_dict = dict(zip(axes_net.replace('C',''),grid))

        normalizer = self._check_normalizer_resizer(normalizer, None)[0]
        resizer = StarDistPadAndCropResizer(grid=grid_dict)

        x = normalizer.before(x, axes_net)
        x = resizer.before(x, axes_net, axes_net_div_by)

        def predict_direct(tile):
            prob, dist = self.keras_model.predict(tile[np.newaxis], **predict_kwargs)
            return prob[0], dist[0]

        if np.prod(n_tiles) > 1:
            tiling_axes   = axes_net.replace('C','') # axes eligible for tiling
            x_tiling_axis = tuple(axes_dict(axes_net)[a] for a in tiling_axes) # numerical axis ids for x
            axes_net_tile_overlaps = self._axes_tile_overlap(axes_net)
            # hack: permute tiling axis in the same way as img -> x was permuted
            n_tiles = _permute_axes(np.empty(n_tiles,np.bool)).shape
            (all(n_tiles[i] == 1 for i in range(x.ndim) if i not in x_tiling_axis) or
                _raise(ValueError("entry of n_tiles > 1 only allowed for axes '%s'" % tiling_axes)))

            sh = [s//grid_dict.get(a,1) for a,s in zip(axes_net,x.shape)]
            sh[channel] = 1;                  prob = np.empty(sh,np.float32)
            sh[channel] = self.config.n_rays; dist = np.empty(sh,np.float32)

            n_block_overlaps = [int(np.ceil(overlap/blocksize)) for overlap, blocksize
                                in zip(axes_net_tile_overlaps, axes_net_div_by)]

            num_tiles_used = total_n_tiles(x, n_tiles, block_sizes=axes_net_div_by, n_block_overlaps=n_block_overlaps)

            for tile, s_src, s_dst in tqdm(tile_iterator(x, n_tiles, block_sizes=axes_net_div_by, n_block_overlaps=n_block_overlaps),
                                           disable=(not show_tile_progress), total=num_tiles_used):
                prob_tile, dist_tile = predict_direct(tile)
                # account for grid
                s_src = [slice(s.start//grid_dict.get(a,1),s.stop//grid_dict.get(a,1)) for s,a in zip(s_src,axes_net)]
                s_dst = [slice(s.start//grid_dict.get(a,1),s.stop//grid_dict.get(a,1)) for s,a in zip(s_dst,axes_net)]
                # prob and dist have different channel dimensionality than image x
                s_src[channel] = slice(None)
                s_dst[channel] = slice(None)
                s_src, s_dst = tuple(s_src), tuple(s_dst)
                # print(s_src,s_dst)
                prob[s_dst] = prob_tile[s_src]
                dist[s_dst] = dist_tile[s_src]

        else:
            prob, dist = predict_direct(x)

        prob = resizer.after(prob, axes_net)
        dist = resizer.after(dist, axes_net)
        dist = np.maximum(1e-3, dist) # avoid small/negative dist values to prevent problems with Qhull

        prob = np.take(prob,0,axis=channel)
        dist = np.moveaxis(dist,channel,-1)

        return prob, dist


    def predict_instances(self, img, axes=None, normalizer=None,
                          prob_thresh=None, nms_thresh=None,
                          n_tiles=None, show_tile_progress=True,
                          verbose = False,
                          predict_kwargs=None, nms_kwargs=None, overlap_label=None):
        """Predict instance segmentation from input image.

        Parameters
        ----------
        img : :class:`numpy.ndarray`
            Input image
        axes : str or None
            Axes of the input ``img``.
            ``None`` denotes that axes of img are the same as denoted in the config.
        normalizer : :class:`csbdeep.data.Normalizer` or None
            (Optional) normalization of input image before prediction.
            Note that the default (``None``) assumes ``img`` to be already normalized.
        prob_thresh : float or None
            Consider only object candidates from pixels with predicted object probability
            above this threshold (also see `optimize_thresholds`).
        nms_thresh : float or None
            Perform non-maximum suppression that considers two objects to be the same
            when their area/surface overlap exceeds this threshold (also see `optimize_thresholds`).
        n_tiles : iterable or None
            Out of memory (OOM) errors can occur if the input image is too large.
            To avoid this problem, the input image is broken up into (overlapping) tiles
            that are processed independently and re-assembled.
            This parameter denotes a tuple of the number of tiles for every image axis (see ``axes``).
            ``None`` denotes that no tiling should be used.
        show_tile_progress: bool
            Whether to show progress during tiled prediction.
        predict_kwargs: dict
            Keyword arguments for ``predict`` function of Keras model.
        nms_kwargs: dict
            Keyword arguments for non-maximum suppression.
        overlap_label: scalar or None
            if not None, label the regions where polygons overlap with that value

        Returns
        -------
        (:class:`numpy.ndarray`, dict)
            Returns a tuple of the label instances image and also
            a dictionary with the details (coordinates, etc.) of all remaining polygons/polyhedra.

        """
        if predict_kwargs is None:
            predict_kwargs = {}
        if nms_kwargs is None:
            nms_kwargs = {}

        nms_kwargs.setdefault("verbose", verbose)

        _axes         = self._normalize_axes(img, axes)
        _axes_net     = self.config.axes
        _permute_axes = self._make_permute_axes(_axes, _axes_net)
        _shape_inst   = tuple(s for s,a in zip(_permute_axes(img).shape, _axes_net) if a != 'C')

        prob, dist = self.predict(img, axes=axes, normalizer=normalizer, n_tiles=n_tiles, show_tile_progress=show_tile_progress, **predict_kwargs)
        return self._instances_from_prediction(_shape_inst, prob, dist, prob_thresh=prob_thresh, nms_thresh=nms_thresh, overlap_label=overlap_label, **nms_kwargs)


    def predict_instances_big(self, img, axes, block_size, min_overlap, context=None,
                              labels_out=None, labels_out_dtype=np.int32, show_progress=True, **kwargs):
        """Predict instance segmentation from very large input images.

        Intended to be used when `predict_instances` cannot be used due to memory limitations.
        This function will break the input image into blocks and process them individually
        via `predict_instances` and assemble all the partial results. If used as intended, the result
        should be the same as if `predict_instances` was used directly on the whole image.

        **Important**: The crucial assumption is that all predicted object instances are smaller than
                       the provided `min_overlap`. Also, it must hold that: min_overlap + 2*context < block_size.

        Example
        -------
        >>> img.shape
        (20000, 20000)
        >>> labels, polys = model.predict_instances_big(img, axes='YX', block_size=4096,
                                                        min_overlap=128, context=128, n_tiles=(4,4))

        Parameters
        ----------
        img: :class:`numpy.ndarray` or similar
            Input image
        axes: str
            Axes of the input ``img`` (such as 'YX', 'ZYX', 'YXC', etc.)
        block_size: int or iterable of int
            Process input image in blocks of the provided shape.
            (If a scalar value is given, it is used for all spatial image dimensions.)
        min_overlap: int or iterable of int
            Amount of guaranteed overlap between blocks.
            (If a scalar value is given, it is used for all spatial image dimensions.)
        context: int or iterable of int, or None
            Amount of image context on all sides of a block, which is discarded.
            If None, uses an automatic estimate that should work in many cases.
            (If a scalar value is given, it is used for all spatial image dimensions.)
        labels_out: :class:`numpy.ndarray` or similar, or None, or False
            numpy array or similar (must be of correct shape) to which the label image is written.
            If None, will allocate a numpy array of the correct shape and data type ``labels_out_dtype``.
            If False, will not write the label image (useful if only the dictionary is needed).
        labels_out_dtype: str or dtype
            Data type of returned label image if ``labels_out=None`` (has no effect otherwise).
        show_progress: bool
            Show progress bar for block processing.
        kwargs: dict
            Keyword arguments for ``predict_instances``.

        Returns
        -------
        (:class:`numpy.ndarray` or False, dict)
            Returns the label image and a dictionary with the details (coordinates, etc.) of the polygons/polyhedra.

        """
        from ..big import _grid_divisible, BlockND, OBJECT_KEYS#, repaint_labels
        from ..matching import relabel_sequential

        n = img.ndim
        axes = axes_check_and_normalize(axes, length=n)
        grid = self._axes_div_by(axes)
        axes_out = self._axes_out.replace('C','')
        shape_dict = dict(zip(axes,img.shape))
        shape_out = tuple(shape_dict[a] for a in axes_out)

        if context is None:
            context = self._axes_tile_overlap(axes)

        if np.isscalar(block_size):  block_size  = n*[block_size]
        if np.isscalar(min_overlap): min_overlap = n*[min_overlap]
        if np.isscalar(context):     context     = n*[context]
        block_size, min_overlap, context = list(block_size), list(min_overlap), list(context)
        assert n == len(block_size) == len(min_overlap) == len(context)

        if 'C' in axes:
            # single block for channel axis
            i = axes_dict(axes)['C']
            # if (block_size[i], min_overlap[i], context[i]) != (None, None, None):
            #     print("Ignoring values of 'block_size', 'min_overlap', and 'context' for channel axis " +
            #           "(set to 'None' to avoid this warning).", file=sys.stderr, flush=True)
            block_size[i] = img.shape[i]
            min_overlap[i] = context[i] = 0

        block_size  = tuple(_grid_divisible(g, v, name='block_size',  verbose=False) for v,g,a in zip(block_size, grid,axes))
        min_overlap = tuple(_grid_divisible(g, v, name='min_overlap', verbose=False) for v,g,a in zip(min_overlap,grid,axes))
        context     = tuple(_grid_divisible(g, v, name='context',     verbose=False) for v,g,a in zip(context,    grid,axes))

        # print(f"input: shape {img.shape} with axes {axes}")
        print(f'effective: block_size={block_size}, min_overlap={min_overlap}, context={context}', flush=True)

        for a,c,o in zip(axes,context,self._axes_tile_overlap(axes)):
            if c < o:
                print(f"{a}: context of {c} is small, recommended to use at least {o}", flush=True)

        # create block cover
        blocks = BlockND.cover(img.shape, axes, block_size, min_overlap, context, grid)

        if np.isscalar(labels_out) and bool(labels_out) is False:
            labels_out = None
        else:
            if labels_out is None:
                labels_out = np.zeros(shape_out, dtype=labels_out_dtype)
            else:
                labels_out.shape == shape_out or _raise(ValueError(f"'labels_out' must have shape {shape_out} (axes {axes_out})."))

        polys_all = {}
        # problem_ids = []
        label_offset = 1

        kwargs_override = dict(axes=axes, overlap_label=None)
        if show_progress:
            kwargs_override['show_tile_progress'] = False # disable progress for predict_instances
        for k,v in kwargs_override.items():
            if k in kwargs: print(f"changing '{k}' from {kwargs[k]} to {v}", flush=True)
            kwargs[k] = v

        blocks = tqdm(blocks, disable=(not show_progress))
        # actual computation
        for block in blocks:
            labels, polys = self.predict_instances(block.read(img, axes=axes), **kwargs)
            labels = block.crop_context(labels, axes=axes_out)
            labels, polys = block.filter_objects(labels, polys, axes=axes_out)
            # TODO: relabel_sequential is not very memory-efficient (will allocate memory proportional to label_offset)
            labels = relabel_sequential(labels, label_offset)[0]
            # labels, fwd_map, _ = relabel_sequential(labels, label_offset)
            # if len(incomplete) > 0:
            #     problem_ids.extend([fwd_map[i] for i in incomplete])
            #     if show_progress:
            #         blocks.set_postfix_str(f"found {len(problem_ids)} problematic {'object' if len(problem_ids)==1 else 'objects'}")
            if labels_out is not None:
                block.write(labels_out, labels, axes=axes_out)
            for k,v in polys.items():
                polys_all.setdefault(k,[]).append(v)
            label_offset += len(polys['prob'])

        polys_all = {k: (np.concatenate(v) if k in OBJECT_KEYS else v[0]) for k,v in polys_all.items()}

        # if labels_out is not None and len(problem_ids) > 0:
        #     # if show_progress:
        #     #     blocks.write('')
        #     # print(f"Found {len(problem_ids)} objects that violate the 'min_overlap' assumption.", file=sys.stderr, flush=True)
        #     repaint_labels(labels_out, problem_ids, polys_all, show_progress=False)

        return labels_out, polys_all#, tuple(problem_ids)


    def optimize_thresholds(self, X_val, Y_val, nms_threshs=[0.3,0.4,0.5], iou_threshs=[0.3,0.5,0.7], predict_kwargs=None, optimize_kwargs=None, save_to_json=True):
        """Optimize two thresholds (probability, NMS overlap) necessary for predicting object instances.

        Note that the default thresholds yield good results in many cases, but optimizing
        the thresholds for a particular dataset can further improve performance.

        The optimized thresholds are automatically used for all further predictions
        and also written to the model directory.

        See ``utils.optimize_threshold`` for details and possible choices for ``optimize_kwargs``.

        Parameters
        ----------
        X_val : list of ndarray
            (Validation) input images (must be normalized) to use for threshold tuning.
        Y_val : list of ndarray
            (Validation) label images to use for threshold tuning.
        nms_threshs : list of float
            List of overlap thresholds to be considered for NMS.
            For each value in this list, optimization is run to find a corresponding prob_thresh value.
        iou_threshs : list of float
            List of intersection over union (IOU) thresholds for which
            the (average) matching performance is considered to tune the thresholds.
        predict_kwargs: dict
            Keyword arguments for ``predict`` function of this class.
            (If not provided, will guess value for `n_tiles` to prevent out of memory errors.)
        optimize_kwargs: dict
            Keyword arguments for ``utils.optimize_threshold`` function.

        """
        if predict_kwargs is None:
            predict_kwargs = {}
        if optimize_kwargs is None:
            optimize_kwargs = {}

        def _predict_kwargs(x):
            if 'n_tiles' in predict_kwargs:
                return predict_kwargs
            else:
                return {**predict_kwargs, 'n_tiles': self._guess_n_tiles(x), 'show_tile_progress': False}

        Yhat_val = [self.predict(x, **_predict_kwargs(x)) for x in X_val]

        opt_prob_thresh, opt_measure, opt_nms_thresh = None, -np.inf, None
        for _opt_nms_thresh in nms_threshs:
            _opt_prob_thresh, _opt_measure = optimize_threshold(Y_val, Yhat_val, model=self, nms_thresh=_opt_nms_thresh, iou_threshs=iou_threshs, **optimize_kwargs)
            if _opt_measure > opt_measure:
                opt_prob_thresh, opt_measure, opt_nms_thresh = _opt_prob_thresh, _opt_measure, _opt_nms_thresh
        opt_threshs = dict(prob=opt_prob_thresh, nms=opt_nms_thresh)

        self.thresholds = opt_threshs
        print(end='', file=sys.stderr, flush=True)
        print("Using optimized values: prob_thresh={prob:g}, nms_thresh={nms:g}.".format(prob=self.thresholds.prob, nms=self.thresholds.nms))
        if save_to_json and self.basedir is not None:
            print("Saving to 'thresholds.json'.")
            save_json(opt_threshs, str(self.logdir / 'thresholds.json'))
        return opt_threshs


    def _guess_n_tiles(self, img):
        axes = self._normalize_axes(img, axes=None)
        shape = list(img.shape)
        if 'C' in axes:
            del shape[axes_dict(axes)['C']]
        b = self.config.train_batch_size**(1.0/self.config.n_dim)
        n_tiles = [int(np.ceil(s/(p*b))) for s,p in zip(shape,self.config.train_patch_size)]
        if 'C' in axes:
            n_tiles.insert(axes_dict(axes)['C'],1)
        return tuple(n_tiles)


    def _normalize_axes(self, img, axes):
        if axes is None:
            axes = self.config.axes
            assert 'C' in axes
            if img.ndim == len(axes)-1 and self.config.n_channel_in == 1:
                # img has no dedicated channel axis, but 'C' always part of config axes
                axes = axes.replace('C','')
        return axes_check_and_normalize(axes, img.ndim)


    def _compute_receptive_field(self, img_size=None):
        # TODO: good enough?
        from scipy.ndimage import zoom
        if img_size is None:
            img_size = tuple(g*(128 if self.config.n_dim==2 else 64) for g in self.config.grid)
        if np.isscalar(img_size):
            img_size = (img_size,) * self.config.n_dim
        img_size = tuple(img_size)
        # print(img_size)
        assert all(_is_power_of_2(s) for s in img_size)
        mid = tuple(s//2 for s in img_size)
        x = np.zeros((1,)+img_size+(self.config.n_channel_in,), dtype=np.float32)
        z = np.zeros_like(x)
        x[(0,)+mid+(slice(None),)] = 1
        y  = self.keras_model.predict(x)[0][0,...,0]
        y0 = self.keras_model.predict(z)[0][0,...,0]
        grid = tuple((np.array(x.shape[1:-1])/np.array(y.shape)).astype(int))
        assert grid == self.config.grid
        y  = zoom(y, grid,order=0)
        y0 = zoom(y0,grid,order=0)
        ind = np.where(np.abs(y-y0)>0)
        return [(m-np.min(i), np.max(i)-m) for (m,i) in zip(mid,ind)]


    def _axes_tile_overlap(self, query_axes):
        query_axes = axes_check_and_normalize(query_axes)
        try:
            self._tile_overlap
        except AttributeError:
            self._tile_overlap = self._compute_receptive_field()
        overlap = dict(zip(
            self.config.axes.replace('C',''),
            tuple(max(rf) for rf in self._tile_overlap)
        ))
        return tuple(overlap.get(a,0) for a in query_axes)


    def export_TF(self, fname=None, single_output=True, upsample_grid=True):
        """Export model to TensorFlow's SavedModel format that can be used e.g. in the Fiji plugin

        Parameters
        ----------
        fname : str
            Path of the zip file to store the model
            If None, the default path "<modeldir>/TF_SavedModel.zip" is used
        single_output: bool
            If set, concatenates the two model outputs into a single output (note: this is currently mandatory for further use in Fiji)
        upsample_grid: bool
            If set, upsamples the output to the input shape (note: this is currently mandatory for further use in Fiji)
        """
        Concatenate, UpSampling2D, UpSampling3D, Conv2DTranspose, Conv3DTranspose = keras_import('layers', 'Concatenate', 'UpSampling2D', 'UpSampling3D', 'Conv2DTranspose', 'Conv3DTranspose')
        Model = keras_import('models', 'Model')

        if self.basedir is None and fname is None:
            raise ValueError("Need explicit 'fname', since model directory not available (basedir=None).")

        grid = self.config.grid
        prob = self.keras_model.outputs[0]
        dist = self.keras_model.outputs[1]
        assert self.config.n_dim in (2,3)

        if upsample_grid and any(g>1 for g in grid):
            # CSBDeep Fiji plugin needs same size input/output
            # -> we need to upsample the outputs if grid > (1,1)
            # note: upsampling prob with a transposed convolution creates sparse
            #       prob output with less candidates than with standard upsampling
            conv_transpose = Conv2DTranspose if self.config.n_dim==2 else Conv3DTranspose
            upsampling     = UpSampling2D    if self.config.n_dim==2 else UpSampling3D
            prob = conv_transpose(1, (1,)*self.config.n_dim,
                                  strides=grid, padding='same',
                                  kernel_initializer='ones', use_bias=False)(prob)
            dist = upsampling(grid)(dist)

        inputs  = self.keras_model.inputs[0]
        outputs = Concatenate()([prob,dist]) if single_output else [prob,dist]
        csbdeep_model = Model(inputs, outputs)

        fname = (self.logdir / 'TF_SavedModel.zip') if fname is None else Path(fname)
        export_SavedModel(csbdeep_model, str(fname))
        return csbdeep_model



class StarDistPadAndCropResizer(Resizer):

    # TODO: check correctness
    def __init__(self, grid, mode='reflect', **kwargs):
        assert isinstance(grid, dict)
        self.mode = mode
        self.grid = grid
        self.kwargs = kwargs


    def before(self, x, axes, axes_div_by):
        assert all(a%g==0 for g,a in zip((self.grid.get(a,1) for a in axes), axes_div_by))
        axes = axes_check_and_normalize(axes,x.ndim)
        def _split(v):
            return 0, v # only pad at the end
        self.pad = {
            a : _split((div_n-s%div_n)%div_n)
            for a, div_n, s in zip(axes, axes_div_by, x.shape)
        }
        x_pad = np.pad(x, tuple(self.pad[a] for a in axes), mode=self.mode, **self.kwargs)
        self.padded_shape = dict(zip(axes,x_pad.shape))
        if 'C' in self.padded_shape: del self.padded_shape['C']
        return x_pad


    def after(self, x, axes):
        # axes can include 'C', which may not have been present in before()
        axes = axes_check_and_normalize(axes,x.ndim)
        assert all(s_pad == s * g for s,s_pad,g in zip(x.shape,
                                                       (self.padded_shape.get(a,_s) for a,_s in zip(axes,x.shape)),
                                                       (self.grid.get(a,1) for a in axes)))
        # print(self.padded_shape)
        # print(self.pad)
        # print(self.grid)
        crop = tuple (
            slice(0, -(math.floor(p[1]/g)) if p[1]>=g else None)
            for p,g in zip((self.pad.get(a,(0,0)) for a in axes),(self.grid.get(a,1) for a in axes))
        )
        # print(crop)
        return x[crop]
