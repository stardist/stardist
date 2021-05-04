"""
TODO:
- error when running the plugin multiple times with shape/surface output (https://github.com/napari/napari/issues/2354)
- support timelapse or channel-wise processing (incl. progress display)
- ability to cancel running stardist computation
- run only on field of view
  - https://forum.image.sc/t/how-could-i-get-the-viewed-coordinates/49709
  - https://napari.zulipchat.com/#narrow/stream/212875-general/topic/corner.20pixels.20and.20dims.20displayed
- normalize image separately per channel or jointly
- add general tooltip help/info messages
- option to use CPU or GPU, limit tensorflow GPU memory ('allow_growth'?)
- move to stardist/stardist-napari repo
- sample data hook
"""

from napari_plugin_engine import napari_hook_implementation
from magicgui import magicgui
from magicgui import widgets as mw
from magicgui.events import Event
from magicgui.application import use_app

import functools
import time
import numpy as np

from pathlib import Path
from warnings import warn

import napari
from napari.qt.threading import thread_worker, create_worker
from napari.utils.colormaps import label_colormap
from typing import List
from enum import Enum


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
        values.extend(i+np.zeros(len(xs)))
        vertices.extend(xs)
        faces.extend(rays_faces.copy())
        rays_faces += len(xs)

    return [np.array(vertices), np.array(faces), np.array(values)]



def plugin_wrapper():
    # delay imports until plugin is requested by user
    # -> especially those importing tensorflow (csbdeep.models.*, stardist.models.*)
    from csbdeep.utils import _raise, normalize, axes_check_and_normalize, axes_dict
    from csbdeep.models.pretrained import get_registered_models, get_model_folder
    from csbdeep.utils import load_json
    from .models import StarDist2D, StarDist3D
    from .utils import abspath

    DEBUG = False

    def get_data(image):
        return image.data[0] if image.multiscale else image.data

    def change_handler(*widgets, init=True, debug=DEBUG):
        def decorator_change_handler(handler):
            @functools.wraps(handler)
            def wrapper(event):
                if debug:
                    if isinstance(event, Event):
                        print(f"{event.type}: {event.source.name} = {event.value}")
                    else:
                        print(f"{handler.__name__}({str(event)})")
                return handler(event)
            for widget in widgets:
                widget.changed.connect(wrapper)
                if init:
                    widget.changed(value=widget.value)
            return wrapper
        return decorator_change_handler

    # -------------------------------------------------------------------------

    # get available models
    _models2d, _aliases2d = get_registered_models(StarDist2D)
    _models3d, _aliases3d = get_registered_models(StarDist3D)
    # use first alias for model selection (if alias exists)
    models2d = [((_aliases2d[m][0] if len(_aliases2d[m]) > 0 else m),m) for m in _models2d]
    models3d = [((_aliases3d[m][0] if len(_aliases3d[m]) > 0 else m),m) for m in _models3d]

    model_configs = dict()
    model_threshs = dict()
    model_selected = None

    CUSTOM_MODEL = 'CUSTOM_MODEL'
    model_type_choices = [('2D', StarDist2D), ('3D', StarDist3D), ('Custom 2D/3D', CUSTOM_MODEL)]

    @functools.lru_cache(maxsize=None)
    def get_model(model_type, model):
        if model_type == CUSTOM_MODEL:
            path = Path(model)
            path.is_dir() or _raise(FileNotFoundError(f"{path} is not a directory"))
            config = model_configs[(model_type,model)]
            model_class = StarDist2D if config['n_dim'] == 2 else StarDist3D
            return model_class(None, name=path.name, basedir=str(path.parent))
        else:
            return model_type.from_pretrained(model)

    # -------------------------------------------------------------------------

    class Output(Enum):
        Labels = 'Label Image'
        Polys  = 'Polygons / Polyhedra'
        Both   = 'Both'
    output_choices = [Output.Labels.value, Output.Polys.value, Output.Both.value]

    # -------------------------------------------------------------------------

    DEFAULTS = dict (
        model_type   = StarDist2D,
        model2d      = models2d[0][1],
        model3d      = models3d[0][1],
        norm_image   = True,
        perc_low     =  1.0,
        perc_high    = 99.8,
        prob_thresh  = 0.5,
        nms_thresh   = 0.4,
        output_type  = Output.Both.value,
        n_tiles      = 'None',
        cnn_output   = False,
    )

    # -------------------------------------------------------------------------

    logo = abspath(__file__, 'resources/stardist_logo_napari.png')

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
        n_tiles         = dict(widget_type='LiteralEvalLineEdit', label='Number of Tiles', value=DEFAULTS['n_tiles']),
        cnn_output      = dict(widget_type='CheckBox', text='Show CNN Output', value=DEFAULTS['cnn_output']),
        set_thresholds  = dict(widget_type='PushButton', text='Set optimized postprocessing thresholds (for selected model)'),
        defaults_button = dict(widget_type='PushButton', text='Restore Defaults'),
        progress_bar    = dict(label=' ', min=0, max=0, visible=False),
        layout          = 'vertical',
        persist         = True,
        call_button     = True,
    )
    def plugin (
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
        progress_bar: mw.ProgressBar,
    ) -> List[napari.types.LayerDataTuple]:

        model = get_model(*model_selected)
        lkwargs = {}
        x = get_data(image)
        axes = axes_check_and_normalize(axes, length=x.ndim)
        if norm_image:
            # TODO: address joint vs. channel-separate normalization properly (let user choose)
            if 'C' not in axes or image.rgb == True:
                 # normalize channels jointly
                _axis = None
            else:
                # normalize channels independently
                _axis = tuple(i for i in range(x.ndim) if i != axes_dict(axes)['C']) if 'C' in axes else None
            x = normalize(x, perc_low,perc_high, axis=_axis)

        if n_tiles is not None and np.prod(n_tiles) > 1:
            n_tiles = tuple(n_tiles)
            app = use_app()
            def progress(it, **kwargs):
                progress_bar.label = 'Neural Network Prediction'
                progress_bar.range = (0, kwargs.get('total',0))
                progress_bar.value = 0
                progress_bar.show()
                app.process_events()
                for item in it:
                    yield item
                    progress_bar.increment()
                    app.process_events()
                #
                progress_bar.label = 'NMS Postprocessing'
                progress_bar.range = (0, 0)
                app.process_events()
                # TODO: progress bar doesn't update during NMS since events not processed?
        else:
            progress = False
            progress_bar.label = 'Neural Network Prediction'
            progress_bar.range = (0, 0)
            progress_bar.show()
            use_app().process_events()

        # TODO: possible to run this in a way that it can be canceled?
        (labels,polys), (prob,dist) = model.predict_instances(x, axes=axes, prob_thresh=prob_thresh, nms_thresh=nms_thresh,
                                                              n_tiles=n_tiles, show_tile_progress=progress, return_predict=True)
        progress_bar.hide()

        layers = []
        if cnn_output:
            scale = tuple(model.config.grid)
            dist = np.moveaxis(dist, -1,0)
            layers.append((dist, dict(name='StarDist distances',   scale=(1,)+scale, **lkwargs), 'image'))
            layers.append((prob, dict(name='StarDist probability', scale=     scale, **lkwargs), 'image'))

        if output_type in (Output.Labels.value,Output.Both.value):
            layers.append((labels, dict(name='StarDist labels', **lkwargs), 'labels'))
        if output_type in (Output.Polys.value,Output.Both.value):
            n_objects = len(polys['points'])
            if isinstance(model, StarDist3D):
                surface = surface_from_polys(polys)
                layers.append((surface, dict(name='StarDist polyhedra',
                                             contrast_limits=(0,surface[-1].max()),
                                             colormap=label_colormap(n_objects), **lkwargs), 'surface'))
            else:
                # TODO: sometimes hangs for long time (indefinitely?) when returning many polygons (?)
                # TODO: coordinates correct or need offset (0.5 or so)?
                shapes = np.moveaxis(polys['coord'], 2,1)
                layers.append((shapes, dict(name='StarDist polygons', shape_type='polygon',
                                            edge_width=0.5, edge_color='yellow', face_color=[0,0,0,0], **lkwargs), 'shapes'))
        return layers

    # -------------------------------------------------------------------------

    # don't want to load persisted values for these widgets
    plugin.axes.value = ''
    plugin.n_tiles.value = DEFAULTS['n_tiles']
    plugin.label_head.value = '<small>Star-convex object detection for 2D and 3D images.<br>If you are using this in your research please <a href="https://github.com/stardist/stardist#how-to-cite" style="color:gray;">cite us</a>.</small><br><br><tt><a href="https://stardist.net" style="color:gray;">https://stardist.net</a></tt>'

    # make labels prettier (https://doc.qt.io/qt-5/qsizepolicy.html#Policy-enum)
    for w in (plugin.label_head, plugin.label_nn, plugin.label_nms, plugin.label_adv):
        w.native.setSizePolicy(1|2, 0)

    # -------------------------------------------------------------------------

    widget_for_modeltype = {
        StarDist2D:   plugin.model2d,
        StarDist3D:   plugin.model3d,
        CUSTOM_MODEL: plugin.model_folder,
    }

    def widgets_inactive(*widgets, active):
        for widget in widgets:
            widget.visible = active
            # widget.native.setStyleSheet("" if active else "text-decoration: line-through")

    def widgets_valid(*widgets, valid):
        for widget in widgets:
            widget.native.setStyleSheet("" if valid else "background-color: lightcoral")


    class Updater:
        def __init__(self, debug=DEBUG):
            from types import SimpleNamespace
            self.debug = debug
            self.valid = SimpleNamespace(**{k:False for k in ('image_axes', 'model', 'n_tiles')})
            self.args  = SimpleNamespace()
            self.viewer = None

        def __call__(self, k, valid, args=None):
            assert k in vars(self.valid)
            setattr(self.valid, k, bool(valid))
            setattr(self.args,  k, args)
            self._update()

        def help(self, msg):
            self.viewer.help = msg

        def _update(self):

            if self.viewer is None:
                # TODO: when is this not safe to do and will hang forever?
                while plugin.viewer.value is None:
                    time.sleep(0.01)
                self.viewer = plugin.viewer.value

                @self.viewer.layers.events.removed.connect
                def _layer_removed(event):
                    layers_remaining = event.source
                    if len(layers_remaining) == 0:
                        plugin.image.tooltip = ''
                        plugin.axes.value = ''
                        plugin.n_tiles.value = 'None'


            def _model(valid):
                widgets_valid(plugin.model2d, plugin.model3d, plugin.model_folder.line_edit, valid=valid)
                if valid:
                    config = self.args.model
                    axes = config.get('axes', 'ZYXC'[-len(config['net_input_shape']):])
                    plugin.model_axes.value = axes.replace("C", f"C[{config['n_channel_in']}]")
                    plugin.model_folder.line_edit.tooltip = ''
                    return axes, config
                else:
                    plugin.model_axes.value = ''
                    plugin.model_folder.line_edit.tooltip = 'Invalid model directory'

            def _image_axes(valid):
                axes, image, err = getattr(self.args, 'image_axes', (None,None,None))
                widgets_valid(plugin.axes, valid=(valid or (image is None and (axes is None or len(axes) == 0))))
                if valid:
                    plugin.axes.tooltip = '\n'.join([f'{a} = {s}' for a,s in zip(axes,get_data(image).shape)])
                    return axes, image
                else:
                    if err is not None:
                        err = str(err)
                        err = err[:-1] if err.endswith('.') else err
                        plugin.axes.tooltip = err
                        # warn(err) # alternative to tooltip (gui doesn't show up in ipython)
                    else:
                        plugin.axes.tooltip = ''

            def _n_tiles(valid):
                n_tiles, image, err = getattr(self.args, 'n_tiles', (None,None,None))
                widgets_valid(plugin.n_tiles, valid=(valid or image is None))
                if valid:
                    plugin.n_tiles.tooltip = 'no tiling' if n_tiles is None else '\n'.join([f'{t}: {s}' for t,s in zip(n_tiles,get_data(image).shape)])
                    return n_tiles
                else:
                    msg = str(err) if err is not None else ''
                    plugin.n_tiles.tooltip = msg

            def _valid_tiles_for_channel(axes_image, n_tiles):
                if n_tiles is not None and 'C' in axes_image:
                    return n_tiles[axes_dict(axes_image)['C']] == 1
                return True

            def _restore():
                widgets_valid(plugin.image, valid=plugin.image.value is not None)


            all_valid = False
            help_msg = ''

            if self.valid.image_axes and self.valid.n_tiles and self.valid.model:
                axes_image, image  = _image_axes(True)
                axes_model, config = _model(True)
                n_tiles = _n_tiles(True)
                if not _valid_tiles_for_channel(axes_image, n_tiles):
                    # check if image axes and n_tiles are compatible
                    widgets_valid(plugin.n_tiles, valid=False)
                    err = 'number of tiles must be 1 for C axis'
                    plugin.n_tiles.tooltip = err
                    _restore()
                else:
                    # check if image and model are compatible
                    ch_model = config['n_channel_in']
                    ch_image = get_data(image).shape[axes_dict(axes_image)['C']] if 'C' in axes_image else 1
                    all_valid = set(axes_model.replace('C','')) == set(axes_image.replace('C','')) and ch_model == ch_image

                    widgets_valid(plugin.image, plugin.model2d, plugin.model3d, plugin.model_folder.line_edit, valid=all_valid)
                    if all_valid:
                        help_msg = ''
                    else:
                        help_msg = f'Model with axes {axes_model.replace("C", f"C[{ch_model}]")} and image with axes {axes_image.replace("C", f"C[{ch_image}]")} not compatible'
            else:
                _image_axes(self.valid.image_axes)
                _n_tiles(self.valid.n_tiles)
                _model(self.valid.model)
                _restore()

            self.help(help_msg)
            plugin.call_button.enabled = all_valid
            # widgets_valid(plugin.call_button, valid=all_valid)
            if self.debug:
                print(f"valid ({all_valid}):", ', '.join([f'{k}={v}' for k,v in vars(self.valid).items()]))

    update = Updater()


    def select_model(key):
        nonlocal model_selected
        model_selected = key
        config = model_configs.get(key)
        update('model', config is not None, config)

    # -------------------------------------------------------------------------

    # hide percentile selection if normalization turned off
    @change_handler(plugin.norm_image)
    def _norm_image_change(event):
        widgets_inactive(plugin.perc_low, plugin.perc_high, active=event.value)

    # ensure that percentile low < percentile high
    @change_handler(plugin.perc_low)
    def _perc_low_change(event):
        plugin.perc_high.value = max(plugin.perc_low.value+0.01, plugin.perc_high.value)
    @change_handler(plugin.perc_high)
    def _perc_high_change(event):
        plugin.perc_low.value  = min(plugin.perc_low.value, plugin.perc_high.value-0.01)

    # -------------------------------------------------------------------------

    # RadioButtons widget triggers a change event initially (either when 'value' is set in constructor, or via 'persist')
    # TODO: seems to be triggered too when a layer is added or removed (why?)
    @change_handler(plugin.model_type, init=False)
    def _model_type_change(event):
        selected = widget_for_modeltype[event.value]
        for w in set((plugin.model2d, plugin.model3d, plugin.model_folder)) - {selected}:
            w.hide()
        selected.show()
        # trigger _model_change
        selected.changed(value=selected.value)


    # show/hide model folder picker
    # load config/thresholds for selected pretrained model
    # -> triggered by _model_type_change
    @change_handler(plugin.model2d, plugin.model3d, init=False)
    def _model_change(event):
        model_class = StarDist2D if event.source == plugin.model2d else StarDist3D
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
                    plugin.progress_bar.hide()

            worker = _get_model_folder()
            worker.returned.connect(_process_model_folder)
            worker.start()

            # delay showing progress bar -> won't show up if model already downloaded
            # TODO: hacky -> better way to do this?
            time.sleep(0.1)
            plugin.call_button.enabled = False
            plugin.progress_bar.label = 'Downloading model'
            plugin.progress_bar.show()

        else:
            select_model(key)


    # load config/thresholds from custom model path
    # -> triggered by _model_type_change
    # note: will be triggered at every keystroke (when typing the path)
    @change_handler(plugin.model_folder, init=False)
    def _model_folder_change(event):
        path = Path(event.value)
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

    # -> triggered by napari (if there are any open images on plugin launch)
    @change_handler(plugin.image, init=False)
    def _image_change(event):
        image = event.value
        ndim = get_data(image).ndim
        plugin.image.tooltip = f"Shape: {get_data(image).shape}"

        # TODO: guess images axes better...
        axes = None
        if ndim == 3:
            axes = 'YXC' if image.rgb else 'ZYX'
        elif ndim == 2:
            axes = 'YX'
        else:
            raise NotImplementedError()

        if (axes == plugin.axes.value):
            # make sure to trigger a changed event, even if value didn't actually change
            plugin.axes.changed(value=axes)
        else:
            plugin.axes.value = axes
        plugin.n_tiles.changed(value=plugin.n_tiles.value)


    # -> triggered by _image_change
    @change_handler(plugin.axes, init=False)
    def _axes_change(event):
        value = str(event.value)
        if value != value.upper():
            with plugin.axes.changed.blocker():
                plugin.axes.value = value.upper()
        image = plugin.image.value
        try:
            image is not None or _raise(ValueError("no image selected"))
            axes = axes_check_and_normalize(value, length=get_data(image).ndim, disallowed='S')
            update('image_axes', True, (axes, image, None))
        except ValueError as err:
            update('image_axes', False, (value, image, err))


    # -> triggered by _image_change
    @change_handler(plugin.n_tiles, init=False)
    def _n_tiles_change(event):
        image = plugin.image.value
        try:
            image is not None or _raise(ValueError("no image selected"))
            value = plugin.n_tiles.get_value()
            if value is None:
                update('n_tiles', True, (None, image, None))
                return
            shape = get_data(image).shape
            try:
                value = tuple(value)
                len(value) == len(shape) or _raise(TypeError())
            except TypeError:
                raise ValueError(f'must be a tuple/list of length {len(shape)}')
            if not all(isinstance(t,int) and t >= 1 for t in value):
                raise ValueError(f'each value must be an integer >= 1')
            update('n_tiles', True, (value, image, None))
        except (ValueError, SyntaxError) as err:
            update('n_tiles', False, (None, image, err))


    # -------------------------------------------------------------------------

    # set thresholds to optimized values for chosen model
    @change_handler(plugin.set_thresholds, init=False)
    def _set_thresholds(event):
        model_type = plugin.model_type.value
        if model_selected in model_threshs:
            thresholds = model_threshs[model_selected]
            plugin.nms_thresh.value = thresholds['nms']
            plugin.prob_thresh.value = thresholds['prob']


    # restore defaults
    @change_handler(plugin.defaults_button, init=False)
    def restore_defaults(event=None):
        for k,v in DEFAULTS.items():
            getattr(plugin,k).value = v

    # -------------------------------------------------------------------------

    # allow some widgets to shrink because their size depends on user input
    plugin.image.native.setMinimumWidth(240)
    plugin.model2d.native.setMinimumWidth(240)
    plugin.model3d.native.setMinimumWidth(240)

    plugin.label_head.native.setOpenExternalLinks(True)
    # make reset button smaller
    # plugin.defaults_button.native.setMaximumWidth(150)

    # plugin.model_axes.native.setReadOnly(True)
    plugin.model_axes.enabled = False

    # push 'call_button' and 'progress_bar' to bottom
    layout = plugin.native.layout()
    layout.insertStretch(layout.count()-2)

    return plugin



@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return plugin_wrapper, {'name': 'StarDist'}
