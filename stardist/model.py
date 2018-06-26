from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter
from six import string_types

import numpy as np
import argparse
import warnings
import datetime

import keras.backend as K
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.layers import Input, Conv2D
from keras.models import Model
from keras.utils import Sequence
from keras.optimizers import Adam

from csbdeep.internals.blocks import unet_block
from csbdeep.utils import _raise, Path, load_json, save_json, backend_channels_last
from csbdeep.data import Resizer, NoResizer, PadAndCropResizer

from .utils import star_dist, edt_prob
from skimage.segmentation import clear_border



if not backend_channels_last():
    raise NotImplementedError(
        "Keras is configured to use the '%s' image data format, which is currently not supported. "
        "Please change it to use 'channels_last' instead: "
        "https://keras.io/getting-started/faq/#where-is-the-keras-configuration-file-stored" % K.image_data_format()
    )



def masked_loss(mask, penalty):
    def _loss(d_true, d_pred):
        return K.mean(mask * penalty(d_pred - d_true), axis=-1)
    return _loss

def masked_loss_mae(mask):
    return masked_loss(mask, K.abs)

def masked_loss_mse(mask):
    return masked_loss(mask, K.square)



class StarDistData(Sequence):

    def __init__(self, X, Y, batch_size, n_rays, patch_size=(256,256), b=32, shape_completion=False, same_patches=False):
        """
        Parameters
        ----------
        same_patches : bool
            Set to true for validation data to always get the same patch for each image
        """

        # TODO: simple augmentations (rotation & flips)
        self.X, self.Y = X, Y
        self.batch_size = batch_size
        self.n_rays = n_rays
        self.patch_size = patch_size
        self.perm = np.random.permutation(len(self.X))
        self.shape_completion = bool(shape_completion)
        self.same_patches = bool(same_patches)

        if self.shape_completion and b > 0:
            self.b = slice(b,-b),slice(b,-b)
        else:
            self.b = slice(None),slice(None)

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def on_epoch_end(self):
        self.perm = np.random.permutation(len(self.X))

    def _random_patches(self, shapes, idx):
        def _single_patch(shape,i):
            all(s>=p for s,p in zip(shape, self.patch_size)) or _raise(ValueError('patch size > image size'))
            rng = np.random.RandomState(i) if self.same_patches else np.random
            start = (rng.randint(0,1+s-p) for s,p in zip(shape, self.patch_size))
            return tuple(slice(st,st+p) for st,p in zip(start, self.patch_size))
        return tuple(_single_patch(s,i) for s,i in zip(shapes,idx))

    def __getitem__(self, i):
        idx = slice(i*self.batch_size,(i+1)*self.batch_size)
        idx = list(self.perm[idx])
        patches = self._random_patches([self.X[k].shape for k in idx], idx)
        X = [self.X[k][sl][self.b] for k,sl in zip(idx,patches)]
        Y = [self.Y[k][sl]         for k,sl in zip(idx,patches)]

        prob = np.stack([edt_prob(lbl[self.b]) for lbl in Y])

        if self.shape_completion:
            Y_cleared = [clear_border(lbl) for lbl in Y]
            dist      = np.stack([star_dist(lbl,self.n_rays)[self.b+(slice(None),)] for lbl in Y_cleared])
            dist_mask = np.stack([edt_prob(lbl[self.b]) for lbl in Y_cleared])
        else:
            dist      = np.stack([star_dist(lbl,self.n_rays) for lbl in Y])
            dist_mask = prob

        X = np.expand_dims(np.stack(X),-1)
        prob = np.expand_dims(prob,-1)
        dist_mask = np.expand_dims(dist_mask,-1)

        return [X,dist_mask], [prob,dist]



class Config(argparse.Namespace):
    """Configuration for a :class:`StarDist` model.

    Parameters
    ----------
    n_rays : int
        Number of radial directions for the star-convex polygon.
        Recommended to use a power of 2 (default: 32).
    n_channel_in : int
        Number of channels of given input image (default: 1).
    kwargs : dict
        Overwrite (or add) configuration attributes (see below).


    Attributes
    ----------
    unet_n_depth : int
        Number of U-Net resolution levels (down/up-sampling layers).
    unet_kernel_size : (int,int)
        Convolution kernel size for all (U-Net) convolution layers.
    unet_n_filter_base : int
        Number of convolution kernels (feature channels) for first U-Net layer.
        Doubled after each down-sampling layer.
    net_conv_after_unet : int
        Number of extra convolution layers after U-Net (0 to disable).
    train_shape_completion : bool
        Train model to predict complete shapes for partially visible objects at image boundary.
    train_completion_crop : int
        If 'train_shape_completion' is set to True, specify number of pixels to crop at boundary of training patches.
        Should be chosen based on (largest) object sizes.
    train_patch_size : (int,int)
        Size of patches to be cropped from provided training images.
    train_dist_loss : str
        Training loss for star-convex polygon distances ('mse' or 'mae').
    train_epochs : int
        Number of training epochs.
    train_steps_per_epoch : int
        Number of parameter update steps per epoch.
    train_learning_rate : float
        Learning rate for training.
    train_batch_size : int
        Batch size for training.
    train_tensorboard : bool
        Enable TensorBoard for monitoring training progress.
    train_checkpoint : str
        Name of checkpoint file for model weights (only best are saved); set to ``None`` to disable.
    train_reduce_lr : dict
        Parameter :class:`dict` of ReduceLROnPlateau_ callback; set to ``None`` to disable.

        .. _ReduceLROnPlateau: https://keras.io/callbacks/#reducelronplateau
    """

    def __init__(self, n_rays=32, n_channel_in=1, **kwargs):
        """See class docstring."""

        # directly set by parameters
        self.n_rays                 = n_rays
        self.n_channel_in           = int(n_channel_in)

        # default config (can be overwritten by kwargs below)
        self.unet_n_depth           = 3
        self.unet_kernel_size       = (3,3)
        self.unet_n_filter_base     = 32
        self.net_conv_after_unet    = 128
        if backend_channels_last():
            self.net_input_shape    = (None, None, self.n_channel_in)
        else:
            self.net_input_shape    = (self.n_channel_in, None, None)

        self.train_shape_completion = False
        self.train_completion_crop  = 32
        self.train_patch_size       = (256,256)

        self.train_dist_loss        = 'mae'
        self.train_epochs           = 100
        self.train_steps_per_epoch  = 400
        self.train_learning_rate    = 0.0003
        self.train_batch_size       = 4
        self.train_tensorboard      = True
        self.train_checkpoint       = 'weights_best.h5'
        self.train_reduce_lr        = {'factor': 0.5, 'patience': 10}

        for k in kwargs:
            setattr(self, k, kwargs[k])


    def is_valid(self, return_invalid=False):
        # TODO: check if configuration is valid
        return True



class StarDist(object):
    """StarDist model.

    Parameters
    ----------
    config : :class:`Config` or None
        Will be saved to disk as JSON (``config.json``).
        If set to ``None``, will be loaded from disk (must exist).
    name : str or None
        Model name. Uses a timestamp if set to ``None`` (default).
    basedir : str
        Directory that contains (or will contain) a folder with the given model name.

    Raises
    ------
    FileNotFoundError
        If ``config=None`` and config cannot be loaded from disk.
    ValueError
        Illegal arguments, including invalid configuration.

    Attributes
    ----------
    config : :class:`Config`
        Configuration, as provided during instantiation.
    keras_model : `Keras model <https://keras.io/getting-started/functional-api-guide/>`_
        Keras neural network model.
    name : str
        Model name.
    logdir : :class:`pathlib.Path`
        Path to model folder (which stores configuration, weights, etc.)
    """

    def __init__(self, config=Config(), name=None, basedir='.'):
        """See class docstring."""

        config is None or isinstance(config,Config) or _raise(ValueError('Invalid configuration: %s' % str(config)))
        # if config is not None and not config.is_valid():
        #     invalid_attr = config.is_valid(True)[1]
        #     raise ValueError('Invalid configuration attributes: ' + ', '.join(invalid_attr))

        name is None or isinstance(name,string_types) or _raise(ValueError())
        isinstance(basedir,(string_types,Path)) or _raise(ValueError())
        self.config = config
        self.basedir = Path(basedir)
        self.name = name
        self._set_logdir()
        self._model_prepared = False
        self.keras_model = self._build()
        if config is None:
            self._find_and_load_weights()


    def _set_logdir(self):
        if self.name is None:
            self.name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
        self.logdir = self.basedir / self.name

        config_file =  self.logdir / 'config.json'
        if self.config is None:
            if config_file.exists():
                config_dict = load_json(str(config_file))
                self.config = Config(**config_dict)
                if not self.config.is_valid():
                    invalid_attr = self.config.is_valid(True)[1]
                    raise ValueError('Invalid attributes in loaded config: ' + ', '.join(invalid_attr))
            else:
                raise FileNotFoundError("config file doesn't exist: %s" % str(config_file.resolve()))
        else:
            if self.logdir.exists():
                warnings.warn('output path for model already exists, files may be overwritten: %s' % str(self.logdir.resolve()))
            self.logdir.mkdir(parents=True, exist_ok=True)
            save_json(vars(self.config), str(config_file))


    def _find_and_load_weights(self,prefer='best'):
        from itertools import chain
        # get all weight files and sort by modification time descending (newest first)
        weights_ext   = ('*.h5','*.hdf5')
        weights_files = chain(*(self.logdir.glob(ext) for ext in weights_ext))
        weights_files = reversed(sorted(weights_files, key=lambda f: f.stat().st_mtime))
        weights_files = list(weights_files)
        if len(weights_files) == 0:
            warnings.warn("Couldn't find any network weights (%s) to load." % ', '.join(weights_ext))
            return
        weights_preferred = list(filter(lambda f: prefer in f.name, weights_files))
        weights_chosen = weights_preferred[0] if len(weights_preferred)>0 else weights_files[0]
        print("Loading network weights from '%s'." % weights_chosen.name)
        self.load_weights(weights_chosen.name)


    def _build(self):
        input_img  = Input(self.config.net_input_shape,name='input')
        input_mask = Input(self.config.net_input_shape,name='dist_mask')

        unet_kwargs = {k[5:]:v for (k,v) in vars(self.config).items() if k.startswith('unet_')}
        unet        = unet_block(**unet_kwargs)(input_img)
        if self.config.net_conv_after_unet > 0:
            unet    = Conv2D(self.config.net_conv_after_unet,self.config.unet_kernel_size,
                             name='features',padding='same',activation='relu')(unet)

        output_prob  = Conv2D(1,                 (1,1),name='prob',padding='same',activation='sigmoid')(unet)
        output_dist  = Conv2D(self.config.n_rays,(1,1),name='dist',padding='same',activation='linear')(unet)
        return Model([input_img,input_mask],[output_prob,output_dist])


    def load_weights(self, name='weights_best.h5'):
        """Load neural network weights from model folder.

        Parameters
        ----------
        name : str
            Name of HDF5 weight file (as saved during or after training).
        """
        self.keras_model.load_weights(str(self.logdir/name))


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

        dist_loss = {'mse': masked_loss_mse, 'mae': masked_loss_mae}[self.config.train_dist_loss]
        input_mask = self.keras_model.inputs[1] # second input layer is mask for dist loss
        self.keras_model.compile(optimizer, loss=['binary_crossentropy',dist_loss(input_mask)])

        self.callbacks = []
        if self.config.train_checkpoint is not None:
            self.callbacks.append(ModelCheckpoint(str(self.logdir / self.config.train_checkpoint), save_best_only=True, save_weights_only=True))

        if self.config.train_tensorboard:
            self.callbacks.append(TensorBoard(log_dir=str(self.logdir), write_graph=False))

        if self.config.train_reduce_lr is not None:
            rlrop_params = self.config.train_reduce_lr
            if 'verbose' not in rlrop_params:
                rlrop_params['verbose'] = True
            self.callbacks.append(ReduceLROnPlateau(**rlrop_params))

        self._model_prepared = True


    def train(self, X, Y, validation_data, epochs=None, steps_per_epoch=None):
        """Train the neural network with the given data.

        Parameters
        ----------
        X : :class:`numpy.ndarray`
            Array of input images.
        Y : :class:`numpy.ndarray`
            Array of label masks.
        validation_data : tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`)
            Tuple of X,Y validation arrays.
        epochs : int
            Optional argument to use instead of the value from ``config``.
        steps_per_epoch : int
            Optional argument to use instead of the value from ``config``.

        Returns
        -------
        ``History`` object
            See `Keras training history <https://keras.io/models/model/#fit>`_.

        """

        validation_data is not None or _raise(ValueError())
        ((isinstance(validation_data,(list,tuple)) and len(validation_data)==2)
            or _raise(ValueError('validation_data must be a pair of numpy arrays')))

        patch_size = self.config.train_patch_size
        b = self.config.train_completion_crop if self.config.train_shape_completion else 0
        div_by = 2**self.config.unet_n_depth
        if any((p-2*b)%div_by!=0 for p in patch_size):
            if self.config.train_shape_completion:
                raise ValueError("every value of 'train_patch_size' - 2*'train_completion_crop' must be divisible by 2**'unet_n_depth'")
            else:
                raise ValueError("every value of 'train_patch_size' must be divisible by 2**'unet_n_depth'")

        if epochs is None:
            epochs = self.config.train_epochs
        if steps_per_epoch is None:
            steps_per_epoch = self.config.train_steps_per_epoch

        if not self._model_prepared:
            self.prepare_for_training()

        data_kwargs = {
            'n_rays':           self.config.n_rays,
            'batch_size':       self.config.train_batch_size,
            'patch_size':       self.config.train_patch_size,
            'shape_completion': self.config.train_shape_completion,
            'b':                self.config.train_completion_crop,
        }

        X_val, Y_val = validation_data
        data_train = StarDistData(X,     Y,     same_patches=False, **data_kwargs)
        data_val   = StarDistData(X_val, Y_val, same_patches=True,  **data_kwargs)

        history = self.keras_model.fit_generator(generator=data_train, validation_data=data_val,
                                                 epochs=epochs, steps_per_epoch=steps_per_epoch,
                                                 callbacks=self.callbacks, verbose=1)

        self.keras_model.save_weights(str(self.logdir / 'weights_last.h5'))

        if self.config.train_checkpoint is not None:
            self.load_weights(self.config.train_checkpoint)

        return history


    def predict(self, img, resizer=PadAndCropResizer(), **predict_kwargs):
        """Predict.

        Parameters
        ----------
        img : :class:`numpy.ndarray`
            Input image
        resizer : :class:`csbdeep.data.Resizer` or None
            If necessary, input image is resized to enable neural network prediction and result is (possibly)
            resized to yield original image size.

        Returns
        -------
        (:class:`numpy.ndarray`,:class:`numpy.ndarray`)
            Returns the tuple (`prob`, `dist`) of per-pixel object probabilities and star-convex polygon distances.

        """
        if resizer is None:
            resizer = NoResizer()
        isinstance(resizer,Resizer) or _raise(ValueError())

        img.ndim in (2,3) or _raise(ValueError())

        channel = img.ndim if backend_channels_last() else 0
        if img.ndim == 2:
            x = np.expand_dims(img,channel)
        self.config.n_channel_in == x.shape[channel] or _raise(ValueError())

        # resize: make divisible by power of 2 to allow downsampling steps in unet
        div_n = 2 ** self.config.unet_n_depth
        x = resizer.before(x,div_n,exclude=channel)

        if backend_channels_last():
            sh = x.shape[:-1] + (1,)
        else:
            sh = (1,) + x.shape[1:]
        dummy = np.empty((1,)+sh,np.float32)

        prob, dist = self.keras_model.predict([np.expand_dims(x,0),dummy],**predict_kwargs)
        prob, dist = prob[0], dist[0]

        prob = resizer.after(prob,exclude=channel)
        dist = resizer.after(dist,exclude=channel)

        prob = np.take(prob,0,channel)
        dist = np.moveaxis(dist,channel,-1)

        return prob, dist
