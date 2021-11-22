from pathlib import Path
from pkg_resources import get_distribution
from itertools import chain
from zipfile import ZipFile
import numpy as np
from csbdeep.utils import axes_check_and_normalize, normalize, _raise


def _import(error=True):
    try:
        from importlib_metadata import metadata
        from bioimageio.core.build_spec import build_model
    except ImportError:
        if error:
            raise RuntimeError(
                "Required libraries are missing for bioimage.io model export.\n"
                "Please install StarDist as follows: pip install 'stardist[bioimageio]'\n"
                "(You do not need to uninstall StarDist first.)"
            )
        else:
            return None
    return metadata, build_model


def _create_stardist_dependencies(outdir):
    pkg_info = get_distribution("stardist")
    reqs = ("tensorflow",) + tuple(map(str, pkg_info.requires()))
    path = outdir / "requirements.txt"
    with open(path, "w") as f:
        f.write("\n".join(reqs))
    return f"pip:{path}"


def _create_stardist_doc(outdir):
    doc_path = outdir / "README.md"
    text = (
        "# StarDist Model\n"
        "This is a model for object detection with star-convex shapes.\n"
        "Please see the [StarDist repository](https://github.com/stardist/stardist) for details."
    )
    with open(doc_path, "w") as f:
        f.write(text)
    return doc_path


def _get_stardist_metadata(outdir):
    metadata, _ = _import()
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
        tags=["stardist", "segmentation", "instance segmentation", "object detection", "tensorflow"],
        covers=["https://raw.githubusercontent.com/stardist/stardist/master/images/stardist_logo.jpg"],
        documentation=_create_stardist_doc(outdir),
    )
    return data


def _predict_tf(model_path, test_input):
    import tensorflow as tf
    from csbdeep.utils.tf import IS_TF_1
    # need to unzip the model assets
    model_assets = model_path.parent / "tf_model"
    with ZipFile(model_path, "r") as f:
        f.extractall(model_assets)
    if IS_TF_1:
        # make a new graph, i.e. don't use the global default graph
        with tf.Graph().as_default():
            with tf.Session() as sess:
                tf_model = tf.saved_model.load_v2(str(model_assets))
                x = tf.convert_to_tensor(test_input, dtype=tf.float32)
                model = tf_model.signatures["serving_default"]
                y = model(x)
                sess.run(tf.global_variables_initializer())
                output = sess.run(y["output"])
    else:
        tf_model = tf.saved_model.load(str(model_assets))
        x = tf.convert_to_tensor(test_input, dtype=tf.float32)
        model = tf_model.signatures["serving_default"]
        y = model(x)
        output = y["output"].numpy()
    return output


def _get_weights_and_model_metadata(outdir, model, test_input, test_input_axes, test_input_norm_axes, mode, min_percentile, max_percentile):

    # get the path to the exported model assets (saved in outdir)
    if mode == "keras_hdf5":
        raise NotImplementedError("Export to keras format is not supported yet")
    elif mode == "tensorflow_saved_model_bundle":
        assets_uri = outdir / "TF_SavedModel.zip"
        model_csbdeep = model.export_TF(assets_uri, single_output=True, upsample_grid=True)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # to force "inputs.data_type: float32" in the spec (bonus: disables normalization warning in model._predict_setup)
    test_input = test_input.astype(np.float32)

    # convert test_input to axes_net semantics and shape, also resize if necessary (to adhere to axes_net_div_by)
    test_input, axes_img, axes_net, axes_net_div_by, *_ = model._predict_setup(
        img = test_input,
        axes = test_input_axes,
        normalizer = None,
        n_tiles = None,
        show_tile_progress = False,
        predict_kwargs = {},
    )

    # normalization axes string and numeric indices
    axes_norm = set(axes_net).intersection(set(axes_check_and_normalize(test_input_norm_axes, disallowed='S')))
    axes_norm = ''.join(a for a in axes_net if a in axes_norm) # preserve order of axes_net
    axes_norm_num = tuple(axes_net.index(a) for a in axes_norm)

    # normalize input image
    test_input_norm = normalize(test_input, pmin=min_percentile, pmax=max_percentile, axis=axes_norm_num)

    net_axes_in = axes_net.lower()
    net_axes_out = axes_check_and_normalize(model._axes_out).lower()
    ndim_tensor = len(net_axes_out) + 1

    input_min_shape = list(axes_net_div_by)
    input_min_shape[axes_net.index('C')] = model.config.n_channel_in
    input_step = list(axes_net_div_by)
    input_step[axes_net.index('C')] = 0

    if mode == "keras_hdf5":
        output_names = ("prob", "dist") + (("class_prob",) if model._is_multiclass() else ())
        output_n_channels = (1, model.config.n_rays,) + ((1,) if model._is_multiclass() else ())
        output_scale = [1]+list(1/g for g in model.config.grid) + [0]

    elif mode == "tensorflow_saved_model_bundle":
        if model._is_multiclass():
            raise NotImplementedError("Tensorflow SavedModel not supported for multiclass models yet")
        # regarding input/output names: https://github.com/CSBDeep/CSBDeep/blob/b0d2f5f344ebe65a9b4c3007f4567fe74268c813/csbdeep/utils/tf.py#L193-L194
        input_names  = ["input"]
        output_names = ["output"]
        output_n_channels = (1 + model.config.n_rays,)
        output_scale = [1]*(ndim_tensor-1) + [0]

    # TODO need config format that is compatible with deepimagej; discuss with Esti
    # TODO do we need parameters for down/upsampling here?
    metadata, _ = _import()
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
        input_min_shape=[[1]+input_min_shape],
        input_step=[[0]+input_step],
        input_axes=[input_axes],
        input_data_range=[["-inf", "inf"]],
        preprocessing=[dict(scale_range=dict(
            mode="per_sample",
            axes=axes_norm.lower(),
            min_percentile=min_percentile,
            max_percentile=max_percentile,
        ))]
    )

    n_outputs = len(output_names)
    output_axes = "b" + net_axes_out.lower()
    output_config = dict(
        output_name=output_names,
        output_data_range=[["-inf", "inf"]] * n_outputs,
        output_axes=[output_axes] * n_outputs,
        output_reference=[input_names[0]] * n_outputs,
        output_scale=[output_scale] * n_outputs,
        output_offset=[[0]*(ndim_tensor)] * n_outputs,
    )

    in_path = outdir / "test_input.npy"
    np.save(in_path, test_input[np.newaxis])

    if mode == "tensorflow_saved_model_bundle":
        test_outputs = _predict_tf(assets_uri, test_input_norm[np.newaxis])
    else:
        test_outputs = model.predict(test_input_norm)

    # out_paths = []
    # for i, out in enumerate(test_outputs):
    #     p = outdir / f"test_output{i}.npy"
    #     np.save(p, out)
    #     out_paths.append(p)
    assert n_outputs == 1
    out_paths = [outdir / f"test_output.npy"]
    np.save(out_paths[0], test_outputs)

    data = dict(weight_uri=assets_uri, test_inputs=[in_path], test_outputs=out_paths, config=config)
    data.update(input_config)
    data.update(output_config)
    return data


def export_bioimageio(
    model,
    outpath,
    test_input,
    test_input_axes=None,
    test_input_norm_axes='ZYX',
    name="bioimageio_model",
    mode="tensorflow_saved_model_bundle",
    min_percentile=1.0,
    max_percentile=99.8,
    overwrite_spec_kwargs={}
):
    """Export stardist model into bioimageio format, https://github.com/bioimage-io/spec-bioimage-io.

    Parameters
    ----------
    model: StarDist2D, StarDist3D
        the model to convert
    outpath: str, Path
        where to save the model
    test_input: np.ndarray
        input image for generating test data
    test_input_axes: str or None
         the axes of the test input, for example 'YX' for a 2d image or 'ZYX' for a 3d volume
         using None assumes that axes of test_input are the same as those of model
    test_input_norm_axes: str
         the axes of the test input which will be jointly normalized, for example 'ZYX' for all spatial dimensions ('Z' ignored for 2D input)
         use 'ZYXC' to also jointly normalize channels (e.g. for RGB input images)
    name: str
        the name of this model (default: "StarDist Model")
    mode: str
        the export type for this model (default: "tensorflow_saved_model_bundle")
    overwrite_spec_kwargs: dict
        spec keywords that should be overloaded (default: {})
    """
    _, build_model = _import()
    from stardist.models import StarDist2D, StarDist3D
    isinstance(model, (StarDist2D, StarDist3D)) or _raise(ValueError("not a valid model"))
    0 <= min_percentile < max_percentile <= 100 or _raise(ValueError("invalid percentile values"))

    outpath = Path(outpath)
    if outpath.suffix == "":
        outdir = outpath
        zip_path = outdir / f"{name}.zip"
    elif outpath.suffix == ".zip":
        outdir = outpath.parent
        zip_path = outpath
    else:
        raise ValueError(f"outpath has to be a folder or zip file, got {outpath}")
    outdir.mkdir(exist_ok=True, parents=True)

    kwargs = _get_stardist_metadata(outdir)
    model_kwargs = _get_weights_and_model_metadata(outdir, model, test_input, test_input_axes, test_input_norm_axes, mode,
                                                   min_percentile=min_percentile, max_percentile=max_percentile)
    kwargs.update(model_kwargs)
    kwargs.update(overwrite_spec_kwargs)

    build_model(name=name, output_path=zip_path, **kwargs)
