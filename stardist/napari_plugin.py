"""
TODO:
- apply only to field of view to apply to huge images? (cf https://forum.image.sc/t/how-could-i-get-the-viewed-coordinates/49709)
- make ui pretty
- option to use CPU or GPU, limit GPU memory ('allow_growth'?)
- execute in different thread, ability to cancel?
- support timelapse or channel-wise processing?
- normalize per channel or jointly
- cache selected model instances (when re-running the plugin, e.g. for different images)
- show info messages in tooltip? or use a label to show info messages?
- how to deal with errors? catch and show to user? cf. napari issues #2205 and #2290
- show progress for tiled prediction and/or timelapse processing?
o load prob and nms thresholds from model
o load model axes and check if compatible with chosen image/axes
- errors when running the plugin multiple times (without deleting output layers first)
- separate 'stardist-napari' package?
"""

from napari_plugin_engine import napari_hook_implementation
from magicgui import magicgui, magic_factory
from magicgui.events import Event

import functools
import numpy as np
from csbdeep.utils import _raise, normalize, axes_check_and_normalize, axes_dict
from csbdeep.models.pretrained import get_registered_models, get_model_folder
from csbdeep.utils import load_json
from .models import StarDist2D, StarDist3D
from .utils import abspath
from pathlib import Path

import napari
from napari.qt.threading import thread_worker, create_worker
from napari.utils.colormaps import label_colormap
from typing import List
from enum import Enum
import time

# TODO: inelegant (wouldn't work if a pretrained model is called the same as CUSTOM_MODEL)
CUSTOM_MODEL = 'CUSTOM_MODEL'

# get available models
_models2d, _aliases2d = get_registered_models(StarDist2D)
_models3d, _aliases3d = get_registered_models(StarDist3D)
# use first alias for model selection (if alias exists)
models2d = [(_aliases2d[m][0] if len(_aliases2d[m]) > 0 else m) for m in _models2d]
models3d = [(_aliases3d[m][0] if len(_aliases3d[m]) > 0 else m) for m in _models3d]

model_configs = dict()
model_threshs = dict()
model_selected = None

class Output(Enum):
    Labels = 'Label Image'
    Polys  = 'Polygons / Polyhedra'
    Both   = 'Both'
output_choices = [Output.Labels.value, Output.Polys.value, Output.Both.value]

class ModelType(Enum):
    TwoD   = '2D'
    ThreeD = '3D'
    Custom = 'Custom 2D/3D'
model_type_choices = [ModelType.TwoD.value, ModelType.ThreeD.value, ModelType.Custom.value]


DEFAULTS = dict (
    model_type   = ModelType.TwoD.value,
    model2d      = models2d[0],
    model3d      = models3d[0],
    norm_image   = True,
    perc_low     =  1.0,
    perc_high    = 99.8,
    prob_thresh  = 0.5,
    nms_thresh   = 0.4,
    output_type  = Output.Both.value,
    n_tiles      = 'None',
    cnn_output   = False,
)



def surface_from_polys(polys):
    from stardist.geometry import dist_to_coord3D
    dist = polys['dist']
    points = polys['points']
    rays_vertices = polys['rays_vertices']
    rays_faces = polys['rays_faces'].copy()
    coord = dist_to_coord3D(dist, points, rays_vertices)

    if not all((coord.ndim==3, coord.shape[-1]==3, rays_faces.shape[-1]==3)):
        raise ValueError(f"Wrong shapes! coord -> (m,n,3) rays_faces -> (k,3)")

    vertices, faces, values = [], [], []
    for i, xs in enumerate(coord, start=1):
        # values.extend(np.random.uniform(0.3,1)+np.random.uniform(-.1,.1,len(xs)))
        values.extend(i+np.zeros(len(xs)))
        vertices.extend(xs)
        faces.extend(rays_faces.copy())
        rays_faces += len(xs)

    return [np.array(vertices), np.array(faces), np.array(values)]


def change_handler(*widgets, init=True, debug=True):
    def decorator_change_handler(handler):
        @functools.wraps(handler)
        def wrapper(event):
            if debug:
                if isinstance(event, Event):
                    print(f"{event.type}{' (blocked)' if event.blocked else ''}: {event.source.name} = {event.value}")
                    # for a in ['blocked','handled','native','source','sources','type','value']:
                    #     try: print(f'- event.{a} = {getattr(event,a)}')
                    #     except: pass
                else:
                    print(f"{handler.__name__}({str(event)})")
            return handler(event)
        for widget in widgets:
            widget.changed.connect(wrapper)
            if init:
                widget.changed(value=widget.value)
        return wrapper
    return decorator_change_handler

# TODO: replace with @magic_factory(..., widget_init=...)
def widget_wrapper():
    logo = abspath( __file__, 'resources/stardist_logo_napari.png')
    @magicgui (
        label_head      = dict(widget_type='Label', label=f'<h1><img src="{logo}">StarDist</h1>'),
        image           = dict(label='Input Image'),
        axes            = dict(widget_type='LineEdit', label='Image Axes'),
        label_nn        = dict(widget_type='Label', label='<br><b>Neural Network Prediction:</b>'),
        model_type      = dict(widget_type='RadioButtons', label='Model Type', orientation='horizontal', choices=model_type_choices, value=DEFAULTS['model_type']),
        model2d         = dict(widget_type='ComboBox', visible=False, label='Pre-trained Model', choices=models2d, value=DEFAULTS['model2d']),
        model3d         = dict(widget_type='ComboBox', visible=False, label='Pre-trained Model', choices=models3d, value=DEFAULTS['model3d']),
        model_folder    = dict(widget_type='FileEdit', visible=False, label='Custom Model', mode='d'),
        model_axes      = dict(widget_type='LineEdit', label='Model Axes', value=''),
        norm_image      = dict(widget_type='CheckBox', text='Normalize Image', value=DEFAULTS['norm_image']),
        label_nms       = dict(widget_type='Label', label='<br><b>NMS Postprocessing:</b>'),
        perc_low        = dict(widget_type='FloatSpinBox', label='Percentile low',              min=0.0, max=100.0, step=0.1,  value=DEFAULTS['perc_low']),
        perc_high       = dict(widget_type='FloatSpinBox', label='Percentile high',             min=0.0, max=100.0, step=0.1,  value=DEFAULTS['perc_high']),
        prob_thresh     = dict(widget_type='FloatSpinBox', label='Probability/Score Threshold', min=0.0, max=  1.0, step=0.05, value=DEFAULTS['prob_thresh']),
        nms_thresh      = dict(widget_type='FloatSpinBox', label='Overlap Threshold',           min=0.0, max=  1.0, step=0.05, value=DEFAULTS['nms_thresh']),
        output_type     = dict(widget_type='ComboBox', label='Output Type', choices=output_choices, value=DEFAULTS['output_type']),
        label_adv       = dict(widget_type='Label', label='<br><b>Advanced Options:</b>'),
        n_tiles         = dict(widget_type='LineEdit', label='Number of Tiles', value=DEFAULTS['n_tiles']),
        cnn_output      = dict(widget_type='CheckBox', text='Show CNN Output', value=DEFAULTS['cnn_output']),
        set_thresholds  = dict(widget_type='PushButton', text='Set optimized postprocessing thresholds (for selected model)'),
        defaults_button = dict(widget_type='PushButton', text='Restore Defaults'),
        progress_bar    = dict(widget_type='ProgressBar', label=' ', min=0, max=0, visible=False),
        layout          = 'vertical',
        persist         = True,
        call_button     = True,
    )
    def widget (
        viewer: napari.Viewer,
        label_head,
        image: napari.layers.Image,
        axes,
        label_nn,
        model_type,
        model2d,
        model3d,
        model_folder,
        model_axes,
        norm_image,
        perc_low,
        perc_high,
        label_nms,
        prob_thresh,
        nms_thresh,
        output_type,
        label_adv,
        n_tiles,
        cnn_output,
        set_thresholds,
        defaults_button,
        progress_bar,
    ) -> List[napari.types.LayerDataTuple]:

        key = {ModelType.TwoD.value:   (StarDist2D,   model2d),
               ModelType.ThreeD.value: (StarDist3D,   model3d),
               ModelType.Custom.value: (CUSTOM_MODEL, model_folder)}[model_type]
        assert key == model_selected
        config = model_configs[key]

        s = lambda _2d,_3d: _2d if config['n_dim'] == 2 else _3d
        if key[0] == CUSTOM_MODEL:
            path = Path(model_folder)
            path.exists() or _raise(FileNotFoundError(f"{path} doesn't exist."))
            model = s(StarDist2D,StarDist3D)(None, name=path.name, basedir=str(path.parent))
        else:
            model = s(StarDist2D,StarDist3D).from_pretrained(s(model2d,model3d))

        x = image.data
        axes = axes_check_and_normalize(axes, length=x.ndim)
        if norm_image:
            # TODO: address joint vs. separate normalization
            if image.rgb == True:
                x = normalize(x, perc_low,perc_high, axis=(0,1,2))
            else:
                x = normalize(x, perc_low,perc_high)

        (labels,polys), (prob,dist) = model.predict_instances(x, axes=axes, prob_thresh=prob_thresh, nms_thresh=nms_thresh, return_predict=True)
        layers = []
        if cnn_output:
            scale = tuple(model.config.grid)
            dist = np.moveaxis(dist, -1,0)
            # TODO: check why scale can be the same tuple for prob and dist (which has a dimension more)
            layers.append((dist, dict(name='StarDist distances', scale=(1,)+scale),   'image'))
            layers.append((prob, dict(name='StarDist probability', scale=scale), 'image'))

        if output_type in (Output.Labels.value,Output.Both.value):
            layers.append((labels, dict(name='StarDist labels'), 'labels'))
        if output_type in (Output.Polys.value,Output.Both.value):
            n_objects = len(polys['points'])
            if isinstance(model, StarDist3D):
                surface = surface_from_polys(polys)
                layers.append((surface, dict(name='StarDist polyhedra',
                                             contrast_limits=(0,surface[-1].max()),
                                             colormap=label_colormap(n_objects)), 'surface'))
            else:
                # TODO: coordinates correct or need offset (0.5 or so)?
                shapes = np.moveaxis(polys['coord'], 2,1)
                layers.append((shapes, dict(name='StarDist polygons', shape_type='polygon',
                                            edge_width=0.5, edge_color='coral', face_color=[0,0,0,0]), 'shapes'))
        return layers

    # -------------------------------------------------------------------------

    def widgets_valid(*widgets, valid):
        for widget in widgets:
            widget.native.setStyleSheet("" if valid else "background-color: lightcoral")

    # https://doc.qt.io/qt-5/qsizepolicy.html#Policy-enum
    for w in (widget.label_head, widget.label_nn, widget.label_nms, widget.label_adv):
        w.native.setSizePolicy(1|2, 0)
    widget.label_head.value = '<small>Star-convex object detection for 2D and 3D images.<br>If you are using this in your research please <a href="https://github.com/stardist/stardist#how-to-cite" style="color:gray;">cite us</a>.</small><br><br><tt><a href="https://stardist.net" style="color:gray;">https://stardist.net</a></tt>'


    def help(msg):
        # it may take a little while until ready
        while widget.viewer.value is None:
            time.sleep(0.01)
        widget.viewer.value.help = msg


    class Updater:
        def __init__(self, debug=True):
            from types import SimpleNamespace
            self.debug = debug
            self.valid = SimpleNamespace(**{k:False for k in ('image_axes', 'model')})
            self.args  = SimpleNamespace()

        def __call__(self, k, valid, args=None):
            assert k in vars(self.valid)
            setattr(self.valid, k, bool(valid))
            setattr(self.args,  k, args)
            self._update()

        def _update(self):

            def _model(valid):
                widgets_valid(widget.model2d, widget.model3d, widget.model_folder.line_edit, valid=valid)
                if valid:
                    config = self.args.model
                    axes = config.get('axes', 'ZYXC'[-len(config['net_input_shape']):])
                    widget.model_axes.value = axes.replace("C", f"C[{config['n_channel_in']}]")
                    widget.model_folder.line_edit.tooltip = ''
                    return axes, config
                else:
                    widget.model_axes.value = ''
                    widget.model_folder.line_edit.tooltip = 'Invalid model directory'

            def _image_axes(valid):
                widgets_valid(widget.axes, valid=valid)
                axes, image, err = getattr(self.args, 'image_axes', (None,None,None))
                if valid:
                    widget.axes.tooltip = '\n'.join([f'{a} = {s}' for a,s in zip(axes,image.data.shape)])
                    return axes, image
                else:
                    if err is not None:
                        err = str(err)
                        widget.axes.tooltip = err[:-1] if err.endswith('.') else err
                    else:
                        widget.axes.tooltip = ''

            all_valid = False
            help_msg = ''
            if self.valid.image_axes and self.valid.model:
                # check if image and model are compatible
                axes_image, image  = _image_axes(True)
                axes_model, config = _model(True)
                ch_model = config['n_channel_in']
                ch_image = image.data.shape[axes_dict(axes_image)['C']] if 'C' in axes_image else 1
                all_valid = set(axes_model.replace('C','')) == set(axes_image.replace('C','')) and ch_model == ch_image

                widgets_valid(widget.image, widget.model2d, widget.model3d, widget.model_folder.line_edit, valid=all_valid)
                if all_valid:
                    help_msg = ''
                else:
                    help_msg = f'Model with axes {axes_model.replace("C", f"C[{ch_model}]")} and image with axes {axes_image.replace("C", f"C[{ch_image}]")} not compatible'
            else:
                # either model or image_axes ist invalid
                _model(self.valid.model)
                _image_axes(self.valid.image_axes)
                widgets_valid(widget.image, valid=True)

            help(help_msg)
            widget.call_button.enabled = all_valid
            # widgets_valid(widget.call_button, valid=all_valid)
            if self.debug:
                print(f"valid ({all_valid}):", ', '.join([f'{k}={v}' for k,v in vars(self.valid).items()]))

    update = Updater()


    def select_model(key):
        # print(f"select_model: {key}")
        global model_selected
        model_selected = key
        config = model_configs.get(key)
        update('model', config is not None, config)

    # -------------------------------------------------------------------------

    # hide percentile selection if normalization turned off
    @change_handler(widget.norm_image)
    def _norm_image_change(event):
        widget.perc_low.visible = event.value
        widget.perc_high.visible = event.value

    # ensure that percentile low < percentile high
    @change_handler(widget.perc_low)
    def _perc_low_change(event):
        widget.perc_high.value = max(widget.perc_low.value+0.01, widget.perc_high.value)
    @change_handler(widget.perc_high)
    def _perc_high_change(event):
        widget.perc_low.value  = min(widget.perc_low.value, widget.perc_high.value-0.01)

    # -------------------------------------------------------------------------

    # RadioButtons widget triggers a change event initially (either when 'value' is set in constructor, or via 'persist')
    @change_handler(widget.model_type, init=False)
    def _model_type_change(event):
        selected = {ModelType.TwoD.value:   widget.model2d,
                    ModelType.ThreeD.value: widget.model3d,
                    ModelType.Custom.value: widget.model_folder}[event.value]
        for w in set((widget.model2d, widget.model3d, widget.model_folder))-{selected}:
            w.hide()
        selected.show()
        # trigger _model_change
        selected.changed(value=selected.value)


    # show/hide model folder picker
    # load config/thresholds for selected pretrained model
    # -> triggered by _model_type_change
    @change_handler(widget.model2d, widget.model3d, init=False)
    def _model_change(event):
        model_class = StarDist2D if event.source == widget.model2d else StarDist3D
        model_name  = event.value
        key = model_class, model_name

        if key not in model_configs:
            @thread_worker
            def _get_model_folder():
                return get_model_folder(*key)

            def _process_model_folder(path):
                try:
                    model_configs[key] = load_json(str(path/'config.json'))
                    try:
                        # not all models have associated thresholds
                        model_threshs[key] = load_json(str(path/'thresholds.json'))
                    except FileNotFoundError:
                        pass
                finally:
                    select_model(key)
                    widget.progress_bar.hide()

            worker = _get_model_folder()
            worker.returned.connect(_process_model_folder)
            worker.start()

            # delay showing progress bar -> won't show up if model already downloaded
            # TODO: hacky -> better way to do this?
            time.sleep(0.1)
            widget.call_button.enabled = False
            widget.progress_bar.label = 'Downloading model...'
            widget.progress_bar.show()

        else:
            select_model(key)


    # load config/thresholds from custom model path
    # -> triggered by _model_type_change
    @change_handler(widget.model_folder, init=False)
    def _model_folder_change(event):
        # path = event.value # bug, does (sometimes?) return the FileEdit widget instead of its value
        path = Path(widget.model_folder.value)
        # note: will be triggered at every keystroke (when typing the path)
        key = CUSTOM_MODEL, path
        try:
            if not path.is_dir(): return
            model_configs[key] = load_json(str(path/'config.json'))
            model_threshs[key] = load_json(str(path/'thresholds.json'))
        except FileNotFoundError:
            pass
        finally:
            select_model(key)

    # -------------------------------------------------------------------------

    # -> triggered by napari
    @change_handler(widget.image, init=False)
    def _image_change(event):
        image = event.value
        ndim = image.data.ndim
        widget.image.tooltip = f"Shape: {image.data.shape}"

        # TODO: guess images axes better...
        axes = None
        if ndim == 3:
            axes = 'YXC' if image.rgb else 'ZYX'
        elif ndim == 2:
            axes = 'YX'
        else:
            raise NotImplementedError()

        if (axes == widget.axes.value):
            # make sure to trigger a change event, even if value didn't actually change (when persisted value is loaded)
            widget.axes.changed(value=axes)
        else:
            widget.axes.value = axes


    # -> triggered by _image_change
    @change_handler(widget.axes, init=False)
    def _axes_change(event):
        value = str(event.value)
        # https://github.com/napari/magicgui/issues/187
        # if value != value.upper():
        #     with widget.axes.changed.blocker():
        #         widget.axes.value = value.upper()
        image = widget.image.value
        try:
            axes = axes_check_and_normalize(value, length=image.data.ndim, disallowed='S')
            update('image_axes', True, (axes, image, None))
        except ValueError as err:
            update('image_axes', False, (value, image, err))

    # -------------------------------------------------------------------------

    # set thresholds to optimized values for chosen model
    @change_handler(widget.set_thresholds, init=False)
    def _set_thresholds(event):
        key = {ModelType.TwoD.value:   (StarDist2D,   widget.model2d.value),
               ModelType.ThreeD.value: (StarDist3D,   widget.model3d.value),
               ModelType.Custom.value: (CUSTOM_MODEL, widget.model_folder.value)}[widget.model_type.value]
        assert model_selected == key
        if key in model_threshs:
            thresholds = model_threshs[key]
            widget.nms_thresh.value = thresholds['nms']
            widget.prob_thresh.value = thresholds['prob']


    # restore defaults
    @change_handler(widget.defaults_button, init=False)
    def restore_defaults(event=None):
        for k,v in DEFAULTS.items():
            getattr(widget,k).value = v

    # -------------------------------------------------------------------------

    # allow to shrink model selector
    widget.model2d.native.setMinimumWidth(240)
    widget.model3d.native.setMinimumWidth(240)

    widget.label_head.native.setOpenExternalLinks(True)
    # make reset button smaller
    # widget.defaults_button.native.setMaximumWidth(150)

    # widget.model_axes.native.setReadOnly(True)
    widget.model_axes.enabled = False

    # push 'call_button' to bottom
    layout = widget.native.layout()
    layout.insertStretch(layout.count()-2)


    return widget


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return widget_wrapper, {'name': 'StarDist'}
