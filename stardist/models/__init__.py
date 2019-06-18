from __future__ import absolute_import, print_function

from .model2d import Config2D, StarDist2D, StarDistData2D
from .model3d import Config3D, StarDist3D, StarDistData3D

from csbdeep.utils import backend_channels_last
import keras.backend as K
if not backend_channels_last():
    raise NotImplementedError(
        "Keras is configured to use the '%s' image data format, which is currently not supported. "
        "Please change it to use 'channels_last' instead: "
        "https://keras.io/getting-started/faq/#where-is-the-keras-configuration-file-stored" % K.image_data_format()
    )
del backend_channels_last, K