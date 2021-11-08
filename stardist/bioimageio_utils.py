from pathlib import Path
from pkg_resources import get_distribution
from importlib_metadata import metadata
from itertools import chain

import numpy as np
from csbdeep.utils import axes_check_and_normalize, normalize

try:
    from bioimageio.core.build_spec import build_model
except ImportError:
    build_model = None


def _create_stardist_dependencies(outdir):
    pkg_info = get_distribution("stardist")
    reqs = ("tensorflow", ) + tuple(map(str, pkg_info.requires()))
    path = outdir / "requirements.txt"
    with open(path, "w") as f:
        f.write("\n".join(reqs))
    return f"pip:{path}"


def _create_stardist_doc(outdir):
    doc_path = outdir / "README.md"
    text = (
        "# StarDist Model\n"
        "This is a model for instance segmentation of starconvex objects with stardist.\n"
        "For details please check out the [stardist repo](https://github.com/stardist/stardist)."
    )
    with open(doc_path, "w") as f:
        f.write(text)
    return doc_path


def _get_stardist_metadata(outdir):
    package_data = metadata("stardist")
    doi_2d = "https://doi.org/10.1007/978-3-030-00934-2_30"
    doi_3d = "https://doi.org/10.1109/WACV45572.2020.9093435"
    data = dict(
        description=package_data["Summary"],
        authors=list(dict(name=name.strip()) for name in package_data["Author"].split(",")),
        git_repo=package_data["Home-Page"],
        license=package_data["License"],
        dependencies=_create_stardist_dependencies(outdir),
        cite={"Cell Detection with Star-Convex Polygons": doi_2d,
              "Star-convex Polyhedra for 3D Object Detection and Segmentation in Microscopy": doi_3d},
        tags=["stardist", "segmentation", "instance segmentation", "tensorflow"],
        covers=["https://raw.githubusercontent.com/stardist/stardist/master/images/stardist_logo.jpg"],
        documentation=_create_stardist_doc(outdir)
    )
    return data


# TODO factor that out (its the same as csbdeep.base_model)
def _get_weights_name(model, prefer="best"):
    # get all weight files and sort by modification time descending (newest first)
    weights_ext = ("*.h5", "*.hdf5")
    weights_files = chain(*(model.logdir.glob(ext) for ext in weights_ext))
    weights_files = reversed(sorted(weights_files, key=lambda f: f.stat().st_mtime))
    weights_files = list(weights_files)
    if len(weights_files) == 0:
        raise ValueError("Couldn't find any network weights (%s) to load." % ', '.join(weights_ext))
    weights_preferred = list(filter(lambda f: prefer in f.name, weights_files))
    weights_chosen = weights_preferred[0] if len(weights_preferred) > 0 else weights_files[0]
    return weights_chosen.name


# TODO we may need to permute axes for images with channel as well
def _expand_dims(x, axes):
    n_expand = len(axes) - x.ndim
    assert n_expand in (0, 1, 2)
    if n_expand == 0:
        return x

    # batch should always be first
    assert axes[0] == "b"
    if n_expand == 1:
        return x[None]

    # channel first or channel last
    assert axes[1] == "c" or axes[-1] == "c"
    if axes[1] == "c":
        expander = np.s_[None, None]
    else:
        expander = np.s_[None, ..., None]
    expanded = x[expander]
    assert expanded.ndim == len(axes)
    return expanded


def _get_weights_and_model_metadata(outdir, model, test_input, mode, prefer_weights):

    # get the path to the weights
    weights_name = _get_weights_name(model, prefer_weights)
    if mode == "keras_hdf5":
        raise NotImplementedError("Export to keras format is not supported yet")
        weight_uri = model.logdir / weights_name
    elif mode == "tensorflow_saved_model_bundle":
        weight_uri = model.logdir / "TF_SavedModel.zip"
        model.load_weights(weights_name)
        model_csbdeep = model.export_TF(weight_uri, single_output=True)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # TODO: this needs more attention, e.g. how axes are treated in a general way
    axes = model.config.axes.lower()
    # img_axes_in = axes_check_and_normalize(axes, model.config.n_dim+1)
    net_axes_in = axes
    net_axes_out = axes_check_and_normalize(model._axes_out).lower()
    # net_axes_lost = set(net_axes_in).difference(set(net_axes_out))
    # img_axes_out = ''.join(a for a in img_axes_in if a not in net_axes_lost)

    ndim_tensor = model.config.n_dim + 2

    # input shape including batch size
    div_by = list(model._axes_div_by(net_axes_in))

    if mode == "keras_hdf5":
        output_names = ("prob", "dist") + (("class_prob",) if model._is_multiclass() else ())
        output_n_channels = (1, model.config.n_rays,) + ((1,) if model._is_multiclass() else ())
        output_scale = [1]+list(1/g for g in model.config.grid) + [0]

    elif mode == "tensorflow_saved_model_bundle":
        if model._is_multiclass():
            raise NotImplementedError("Tensorflow SaveModel not supported for multiclass models yet")
        # output_names = ("outputall",)
        # output_n_channels = (1 + model.config.n_rays,)
        # output_scale = [1]*(ndim_tensor-1) + [0]

        # output_names = ("prob",)
        # output_n_channels = (1,)
        # output_scale = [1]*(ndim_tensor-1) + [0]

        # TODO what are the correct input/output names?
        # input_names = [inp.name for inp in model_csbdeep.inputs]
        # output_names = [out.name for out in model_csbdeep.outputs]
        input_names = model_csbdeep.input_names
        output_names = model_csbdeep.output_names

        output_n_channels = (1 + model.config.n_rays,)
        output_scale = [1]*(ndim_tensor-1) + [0]
        # print(output_names)

    # TODO need config format that is compatible with deepimagej; discuss with Esti
    package_data = metadata("stardist")
    config = dict(
        stardist=dict(
            stardist_version=package_data["Version"],
            thresholds=dict(nms=model.thresholds.nms, prob=model.thresholds.prob)
        )
    )

    n_inputs = len(input_names)
    assert n_inputs == 1
    input_axes = "b" + net_axes_in.lower()
    input_config = dict(
        input_name=input_names,
        input_step=[[0]+div_by] * n_inputs,
        input_min_shape=[[1] + div_by] * n_inputs,
        input_axes=[input_axes] * n_inputs,
        input_data_range=[["-inf", "inf"]] * n_inputs,
        preprocessing=[dict(scale_range=dict(
            mode="per_sample",
            # TODO mighe make it an option to normalize across channels ...
            axes=net_axes_in.lower().replace("c", ""),
            min_percentile=1.0,
            max_percentile=99.8,
        ))] * n_inputs
    )

    n_outputs = len(output_names)
    assert len(output_n_channels) == n_outputs
    output_axes = "b" + net_axes_out.lower()
    output_config = dict(
        output_name=output_names,
        output_data_range=[["-inf", "inf"]] * n_outputs,
        output_axes=[output_axes] * n_outputs,
        output_reference=[input_names[0]] * n_outputs,
        output_scale=[output_scale] * n_outputs,
        output_offset=[[1] * (ndim_tensor-1) + [n_channel] for n_channel in output_n_channels]
    )

    in_path = outdir / "test_input.npy"
    np.save(in_path, _expand_dims(test_input, input_axes))

    test_outputs = model.predict(normalize(test_input))
    # tensorflow model provides a merged output tensor
    if mode == "tensorflow_saved_model_bundle":
        # hard-coded to channel last
        test_outputs = np.concatenate([_expand_dims(out, output_axes) for out in test_outputs], axis=-1)
        test_outputs = [test_outputs]

    out_paths = []
    for i, out in enumerate(test_outputs):
        p = outdir / f"test_output{i}.npy"
        np.save(p, _expand_dims(out, output_axes))
        out_paths.append(p)

    data = dict(weight_uri=weight_uri, test_inputs=[in_path], test_outputs=out_paths, config=config)
    data.update(input_config)
    data.update(output_config)
    return data


def export_bioimageio(
    model,
    outpath,
    test_input,
    name=None,
    mode="tensorflow_saved_model_bundle",
    prefer_weights="best",
    overwrite_spec_kwargs={}
):
    """Export stardist model into bioimageio format, https://github.com/bioimage-io/spec-bioimage-io.

    Parameters
    ----------
    model: StarDist2D, StarDist3d
        the model to convert
    outpath: str, Path
        where to save the model
    test_input: np.ndarray
        input image for generating test data
    name: str
        the name of this model (default: None)
    mode: str
        (default: "tensorflow_saved_model_bundle")
    prefer_weights: str
        (default: "best")
    overwrite_spec_kwargs: dict
        (default: {})
    """
    if build_model is None:
        raise RuntimeError(
            "bioimageio.core is required for modelzoo export."
            "Install it via 'pip install bioimageio.core' or 'conda install -c conda-forge bioimageio.core'."
        )
    name = "StarDist Model" if name is None else name

    outpath = Path(outpath)
    if outpath.suffix == "":
        outdir = outpath
        zip_path = outdir / f"{name}.zip"
    elif outpath.suffix == ".zip":
        outdir = outpath.parent
        zip_path = outpath
    else:
        raise ValueError(f"outpath has to be a folder or zip file, got {outpath.suffix}")
    outdir.mkdir(exist_ok=True, parents=True)

    kwargs = _get_stardist_metadata(outdir)
    model_kwargs = _get_weights_and_model_metadata(outdir, model, test_input, mode, prefer_weights)
    kwargs.update(model_kwargs)
    kwargs.update(overwrite_spec_kwargs)

    build_model(name=name, output_path=zip_path, **kwargs)
