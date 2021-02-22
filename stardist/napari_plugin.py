"""
TODO:
- cache selected models?
- option to use CPU or GPU, limit GPU memory ('allow_growth'?)
- execute in different thread, ability to cancel?
- normalize per channel or jointly
- other output types (labels, shapes, surface). possible to choose dynamically?
- support custom models from file system (also url?)
- advanced options (n_tiles, boundary exclusion, show cnn output)
- restore defaults button
- make ui pretty
- how to deal with errors? catch and show to user?
- let user choose output type (labels, polys, both)
"""

from napari_plugin_engine import napari_hook_implementation
from magicgui import magicgui
# from magicgui.widgets import ProgressBar

import numpy as np
from csbdeep.utils import normalize, axes_check_and_normalize
from csbdeep.models.pretrained import get_registered_models
from .models import StarDist2D, StarDist3D

import napari
from typing import List

# TODO: inelegant (wouldn't work if a pretrained model is called the same as _MODEL_FOLDER)
_MODEL_FOLDER = 'Custom Model'

# get available models
_models2d, _aliases2d = get_registered_models(StarDist2D)
_models3d, _aliases3d = get_registered_models(StarDist3D)
# use first alias for model selection (if alias exists)
models2d = [(_aliases2d[m][0] if len(_aliases2d[m]) > 0 else m) for m in _models2d] + [_MODEL_FOLDER]
models3d = [(_aliases3d[m][0] if len(_aliases3d[m]) > 0 else m) for m in _models3d] + [_MODEL_FOLDER]


def _is3D(image):
    # TODO: possible to know if an image is multi-channel/timelapse 2D or single-channel 3D?
    # TODO: best would be to know image axes...
    return image.data.ndim == 3 and image.rgb == False


def widget_wrapper():

    @magicgui (
        image        = dict(label='Input Image'),
        axes         = dict(widget_type='LineEdit', label='Input Axes'),
        label_nn     = dict(widget_type='Label', label='<br>Neural Network Prediction:'),
        model2d      = dict(widget_type='ComboBox', label='2D Model', choices=models2d),
        model3d      = dict(widget_type='ComboBox', label='3D Model', choices=models3d),
        model_folder = dict(widget_type='FileEdit', label=' ', mode='d'),
        norm_image   = dict(widget_type='CheckBox', text='Normalize Image', value=True),
        label_nms    = dict(widget_type='Label', label='<br>NMS Postprocessing:'),
        perc_low     = dict(widget_type='FloatSpinBox', label='Percentile low',  min=0.0, max=100.0, step=0.1, value=1.0),
        perc_high    = dict(widget_type='FloatSpinBox', label='Percentile high', min=0.0, max=100.0, step=0.1, value=99.8),
        prob_thresh  = dict(widget_type='FloatSpinBox', label='Probability/Score Threshold', min=0.0, max=1.0, step=0.05, value=0.5),
        nms_thresh   = dict(widget_type='FloatSpinBox', label='Overlap Threshold',           min=0.0, max=1.0, step=0.05, value=0.4),
        label_adv    = dict(widget_type='Label', label='<br>Advanced Options:'),
        reset_button = dict(widget_type='PushButton', text='Restore Defaults'),
        layout       = 'vertical',
        call_button  = True,
    )
    def widget (
        image: 'napari.layers.Image',
        axes,
        label_nn,
        model2d,
        model3d,
        model_folder,
        norm_image: bool,
        perc_low: float,
        perc_high: float,
        label_nms,
        prob_thresh: float,
        nms_thresh: float,
        label_adv,
        reset_button,
        # pbar: ProgressBar,
    ) -> List[napari.types.LayerDataTuple]:

        # for i in range(100):
        #     pbar.increment()
        # return (normalize(image.data,1,99.8) > 0.6).astype(int)


        if _is3D(image):
            model = StarDist3D.from_pretrained(model3d)
        else:
            model = StarDist2D.from_pretrained(model2d)

        x = image.data
        axes = axes_check_and_normalize(axes, length=x.ndim)
        if norm_image:
            # TODO: address joint vs. separate normalization
            if image.rgb == True:
                x = normalize(x, perc_low,perc_high, axis=(0,1,2))
            else:
                x = normalize(x, perc_low,perc_high)

        labels, polys = model.predict_instances(x, axes=axes, prob_thresh=prob_thresh, nms_thresh=nms_thresh)

        results = [(labels, dict(name='StarDist labels'), 'labels')]
        if _is3D(image):
            # TODO: 3D polyhedra or surfaces
            pass
        else:
            # TODO: coordinates correct or need offset (0.5 or so)?
            shapes = np.moveaxis(polys['coord'], 2,1)
            results.append (
                (shapes, dict(name='StarDist polygons', shape_type='polygon', edge_width=0.5, edge_color='coral', face_color=[0,0,0,0]), 'shapes')
            )

        # TODO: option to also return 'prob' and 'dist'
        # prob, dist = model.predict(x, axes=axes)
        # results.append( (prob,dict(name='prob'),'image') )
        # results.append( (dist,dict(name='dist'),'image') )

        return results

    # print(widget)

    def _reset_button_change(event):
        print("TODO: restore defaults")
    widget.reset_button.changed.connect(_reset_button_change)

    # ensure that percentile low <= percentile high
    def _perc_low_change(event):
        widget.perc_high.value = max(widget.perc_low.value, widget.perc_high.value)
    def _perc_high_change(event):
        widget.perc_low.value  = min(widget.perc_low.value, widget.perc_high.value)
    widget.perc_low.changed.connect(_perc_low_change)
    widget.perc_high.changed.connect(_perc_high_change)

    # hide percentile selection if normalization turned off
    def _norm_image_change(event):
        widget.perc_low.visible = widget.norm_image.value
        widget.perc_high.visible = widget.norm_image.value
    widget.norm_image.changed.connect(_norm_image_change)

    # show/hide model folder picker
    def _model_change(event):
        if (widget.model2d.visible and widget.model2d.value == _MODEL_FOLDER) or \
           (widget.model3d.visible and widget.model3d.value == _MODEL_FOLDER):
            widget.model_folder.show()
        else:
            widget.model_folder.hide()
    widget.model2d.changed.connect(_model_change)
    widget.model3d.changed.connect(_model_change)

    # show 2d or 3d models (based on guessed image dimensionality)
    def _image_changed(event):
        image = widget.image.get_value()
        if _is3D(image):
            widget.model2d.hide()
            widget.model3d.show()
            widget.axes.value = 'ZYX'
        else:
            widget.model3d.hide()
            widget.model2d.show()
            widget.axes.value = 'YXC' if image.rgb else 'YX'
        _model_change(None)
    widget.image.changed.connect(_image_changed)

    # def _axes_change(event):
    #     print(f'axes = {widget.axes.value}')
    # widget.axes.changed.connect(_axes_change)

    # allow to shrink model selector
    widget.model2d.native.setMinimumWidth(240)
    widget.model3d.native.setMinimumWidth(240)

    # make reset button smaller
    widget.reset_button.native.setMaximumWidth(150)

    # push 'call_button' to bottom
    layout = widget.native.layout()
    layout.insertStretch(layout.count()-1)

    return widget


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return widget_wrapper, {'name': 'StarDist'}
