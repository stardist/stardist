import tempfile
from pathlib import Path
from typing import Union
from zipfile import ZipFile, is_zipfile

import numpy as np
from csbdeep.utils import _raise, axes_check_and_normalize, normalize
from packaging.version import Version
from pkg_resources import get_distribution
from typing_extensions import assert_never

import shutil
from csbdeep.utils import save_json
from .models.model2d import Config2D, StarDist2D
from .models.model3d import Config3D, StarDist3D

DEEPIMAGEJ_MACRO = """
//*******************************************************************
// Date: July-2021
// Credits: StarDist, DeepImageJ
// URL:
//      https://github.com/stardist/stardist
//      https://deepimagej.github.io/deepimagej
// This macro was adapted from
// https://github.com/deepimagej/imagej-macros/blob/648caa867f6ccb459649d4d3799efa1e2e0c5204/StarDist2D_Post-processing.ijm
// Please cite the respective contributions when using this code.
//*******************************************************************
//  Macro to run StarDist postprocessing on 2D images.
//  StarDist and deepImageJ plugins need to be installed.
//  The macro assumes that the image to process is a stack in which
//  the first channel corresponds to the object probability map
//  and the remaining channels are the radial distances from each
//  pixel to the object boundary.
//*******************************************************************

// Get the name of the image to call it
getDimensions(width, height, channels, slices, frames);
name=getTitle();

probThresh={probThresh};
nmsThresh={nmsThresh};

// Isolate the detection probability scores
run("Make Substack...", "channels=1");
rename("scores");

// Isolate the oriented distances
run("Fire");
selectWindow(name);
run("Delete Slice", "delete=channel");
selectWindow(name);
run("Properties...", "channels=" + maxOf(channels, slices) - 1 + " slices=1 frames=1 pixel_width=1.0000 pixel_height=1.0000 voxel_depth=1.0000");
rename("distances");
run("royal");

// Run StarDist plugin
run("Command From Macro", "command=[de.csbdresden.stardist.StarDist2DNMS], args=['prob':'scores', 'dist':'distances', 'probThresh':'" + probThresh + "', 'nmsThresh':'" + nmsThresh + "', 'outputType':'Both', 'excludeBoundary':'2', 'roiPosition':'Stack', 'verbose':'false'], process=[false]");
"""

_BIOIMAGEIO_LIBRARIES_ARE_MISSING = (
    "Required libraries are missing for bioimage.io model export.\n"
    "Please install StarDist as follows: pip install 'stardist[bioimageio]'\n"
    "(You do not need to uninstall StarDist first.)"
)


def _import(error=True):
    try:
        import bioimageio.core
        import xarray as xr
        from bioimageio.spec.model.v0_5 import ModelDescr
        from importlib_metadata import metadata
    except ImportError:
        if error:
            raise RuntimeError(_BIOIMAGEIO_LIBRARIES_ARE_MISSING)
        else:
            return None
    return metadata, ModelDescr, bioimageio.core, xr


def _create_stardist_dependencies(outdir):
    from ruamel.yaml import YAML
    from tensorflow import __version__ as tf_version

    from . import __version__ as stardist_version

    pkg_info = get_distribution("stardist")
    # dependencies that start with the name "bioimageio" will be added as conda dependencies
    reqs_conda = [
        f"{req.project_name}{req.specifier}"
        for req in pkg_info.requires(extras=["bioimageio"])
        if req.key.startswith("bioimageio")
    ]
    # only stardist and tensorflow as pip dependencies
    v_tf = Version(tf_version)
    reqs_pip = (
        f"stardist>={stardist_version}",
        f"tensorflow>={v_tf.major}.{v_tf.minor},<{v_tf.major+1}",
    )
    # conda environment
    env = dict(
        name="stardist",
        channels=["defaults", "conda-forge"],
        dependencies=[
            ("python>=3.7,<3.8" if v_tf.major == 1 else "python>=3.7"),
            *reqs_conda,
            "pip",
            {"pip": reqs_pip},
        ],
    )
    yaml = YAML(typ="safe")
    path = outdir / "environment.yaml"
    with open(path, "w") as f:
        yaml.dump(env, f)
    return f"conda:{path}"


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


def _get_stardist_metadata_legacy(outdir, model, generate_default_deps):
    metadata, *_ = _import()
    package_data = metadata("stardist")
    # doi_2d = "https://doi.org/10.1007/978-3-030-00934-2_30"
    # doi_3d = "https://doi.org/10.1109/WACV45572.2020.9093435"
    doi_2d = "10.1007/978-3-030-00934-2_30"
    doi_3d = "10.1109/WACV45572.2020.9093435"
    authors = {
        "Martin Weigert": dict(name="Martin Weigert", github_user="maweigert"),
        "Uwe Schmidt": dict(name="Uwe Schmidt", github_user="uschmidt83"),
    }
    data = dict(
        description=package_data["Summary"],
        authors=list(
            authors.get(name.strip(), dict(name=name.strip()))
            for name in package_data["Author"].split(",")
        ),
        git_repo=package_data["Home-Page"],
        license=package_data["License"],
        cite=[
            {"text": "Cell Detection with Star-Convex Polygons", "doi": doi_2d},
            {
                "text": "Star-convex Polyhedra for 3D Object Detection and Segmentation in Microscopy",
                "doi": doi_3d,
            },
        ],
        tags=[
            "fluorescence-light-microscopy",
            "whole-slide-imaging",
            "other",  # modality
            f"{model.config.n_dim}d",  # dims
            "cells",
            "nuclei",  # content
            "tensorflow",  # framework
            "fiji",  # software
            "unet",  # network
            "instance-segmentation",
            "object-detection",  # task
            "stardist",
        ],
        covers=[
            "https://raw.githubusercontent.com/stardist/stardist/main/images/stardist_logo.jpg"
        ],
        documentation=_create_stardist_doc(outdir),
    )
    if generate_default_deps:  # only if requested, as not required for bioimage.io
        data["dependencies"] = _create_stardist_dependencies(outdir)

    return data

def _get_stardist_metadata(outdir, model, generate_default_deps):
    from bioimageio.spec.model.v0_5 import Author, CiteEntry, Doi, HttpUrl, LicenseId
    from importlib_metadata import metadata

    package_data = metadata("stardist")
    
    data = dict(
        authors = [Author(name="Martin Weigert", github_user="maweigert"),
                Author(name="Uwe Schmidt", github_user="uschmidt83")],
        git_repo = HttpUrl(package_data["Home-Page"]),
        license=LicenseId(package_data["License"]),
        cite = [
            CiteEntry(text="Cell Detection with Star-Convex Polygons", doi=Doi("10.1007/978-3-030-00934-2_30")),
            CiteEntry(text="Star-convex Polyhedra for 3D Object Detection and Segmentation in Microscopy", doi=Doi("10.1109/WACV45572.2020.9093435"))
        ],
        tags = [
                "fluorescence-light-microscopy",
                "whole-slide-imaging",
                "other",  # modality
                f"{model.config.n_dim}d",  # dims
                "cells",
                "nuclei",  # content
                "tensorflow",  # framework
                "fiji",  # software
                "unet",  # network
                "instance-segmentation",
                "object-detection",  # task
                "stardist",
            ],
        covers = HttpUrl("https://raw.githubusercontent.com/stardist/stardist/main/images/stardist_logo.jpg"),
        documentation = _create_stardist_doc(outdir)
    )
    if generate_default_deps:  # only if requested, as not required for bioimage.io
        data["dependencies"] = _create_stardist_dependencies(outdir)

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


def _get_weights_and_model_metadata(
    outdir,
    model,
    test_input,
    test_input_axes,
    test_input_norm_axes,
    mode,
    min_percentile,
    max_percentile,
    upsample_grid=True,
):

    # get the path to the exported model assets (saved in outdir)
    if mode == "keras_hdf5":
        raise NotImplementedError("Export to keras format is not supported yet")
    elif mode == "keras_v3":
        assets_uri = outdir / "model.keras"
        model_keras = model.keras_model
        model_keras.save(assets_uri)
    elif mode == "tensorflow_saved_model_bundle":
        assets_uri = outdir / "TF_SavedModel.zip"
        model_csbdeep = model.export_TF(
            assets_uri, single_output=True, upsample_grid=upsample_grid
            # assets_uri, single_output=True, upsample_grid=True
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # to force "inputs.data_type: float32" in the spec (bonus: disables normalization warning in model._predict_setup)
    test_input = test_input.astype(np.float32)

    # convert test_input to axes_net semantics and shape, also resize if necessary (to adhere to axes_net_div_by)
    test_input, axes_img, axes_net, axes_net_div_by, *_ = model._predict_setup(
        img=test_input,
        axes=test_input_axes,
        normalizer=None,
        n_tiles=None,
        show_tile_progress=False,
        predict_kwargs={},
    )

    # normalization axes string and numeric indices
    axes_norm = set(axes_net).intersection(
        set(axes_check_and_normalize(test_input_norm_axes, disallowed="S"))
    )
    axes_norm = "".join(
        a for a in axes_net if a in axes_norm
    )  # preserve order of axes_net
    axes_norm_num = tuple(axes_net.index(a) for a in axes_norm)

    # normalize input image
    test_input_norm = normalize(
        test_input, pmin=min_percentile, pmax=max_percentile, axis=axes_norm_num
    )

    net_axes_in = axes_net.lower()
    net_axes_out = axes_check_and_normalize(model._axes_out).lower()
    ndim_tensor = len(net_axes_out) + 1

    input_min_shape = list(axes_net_div_by)
    input_min_shape[axes_net.index("C")] = model.config.n_channel_in
    input_step = list(axes_net_div_by)
    input_step[axes_net.index("C")] = 0

    # add the batch axis to shape and step
    input_min_shape = [1] + input_min_shape
    input_step = [0] + input_step

    # the axes strings in bioimageio convention
    input_axes = "b" + net_axes_in.lower()
    output_axes = "b" + net_axes_out.lower()

    if mode == "keras_hdf5":
        output_names = ("prob", "dist") + (
            ("class_prob",) if model._is_multiclass() else ()
        )
        output_n_channels = (
            1,
            model.config.n_rays,
        ) + ((1,) if model._is_multiclass() else ())
        # the output shape is computed from the input shape using
        # output_shape[i] = output_scale[i] * input_shape[i] + 2 * output_offset[i]
        output_scale = [1] + list(1 / g for g in model.config.grid) + [0]
        output_offset = [0] * (ndim_tensor)

    elif mode == "keras_v3":
        import keras
        if not hasattr(keras, "__version__") or Version(keras.__version__) < Version("3.0.0"):
            raise NotImplementedError(
                "Keras v3 export requires Keras 3.0.0 or higher"
            )
    
        input_names = ["input"]
        output_names = ["output"]
        output_n_channels = (1 + model.config.n_rays,)

        output_scale = [1] * (ndim_tensor)
        output_scale[output_axes.index("c")] = 0
        
        output_offset = [0.0] * (ndim_tensor)
        output_offset[output_axes.index("c")] = output_n_channels[0] / 2.0

    elif mode == "tensorflow_saved_model_bundle":
        if model._is_multiclass():
            raise NotImplementedError(
                "Tensorflow SavedModel not supported for multiclass models yet"
            )
        # regarding input/output names: https://github.com/CSBDeep/CSBDeep/blob/b0d2f5f344ebe65a9b4c3007f4567fe74268c813/csbdeep/utils/tf.py#L193-L194
        input_names = ["input"]
        output_names = ["output"]
        output_n_channels = (1 + model.config.n_rays,)
        # the output shape is computed from the input shape using
        # output_shape[i] = output_scale[i] * input_shape[i] + 2 * output_offset[i]
        # same shape as input except for the channel dimension
        output_scale = [1] * (ndim_tensor)
        output_scale[output_axes.index("c")] = 0
        # no offset, except for the input axes, where it is output channel / 2
        output_offset = [0.0] * (ndim_tensor)
        output_offset[output_axes.index("c")] = output_n_channels[0] / 2.0

    assert all(
        s in (0, 1) for s in output_scale
    ), "halo computation assumption violated"
    halo = model._axes_tile_overlap(output_axes.replace("b", "s"))
    halo = [
        int(np.ceil(v / 8) * 8) for v in halo
    ]  # optional: round up to be divisible by 8

    # the output shape needs to be valid after cropping the halo, so we add the halo to the input min shape
    input_min_shape = [ms + 2 * ha for ms, ha in zip(input_min_shape, halo)]

    # make sure the input min shape is still divisible by the min axis divisor
    input_min_shape = input_min_shape[:1] + [
        ms + (-ms % div_by) for ms, div_by in zip(input_min_shape[1:], axes_net_div_by)
    ]
    assert all(
        ms % div_by == 0 for ms, div_by in zip(input_min_shape[1:], axes_net_div_by)
    )

    metadata, *_ = _import()
    package_data = metadata("stardist")
    is_2D = model.config.n_dim == 2

    weights_file = outdir / "stardist_weights.h5"
    model.keras_model.save_weights(str(weights_file))

    config = dict(
        stardist=dict(
            python_version=package_data["Version"],
            thresholds=dict(model.thresholds._asdict()),
            weights=weights_file.name,
            config=vars(model.config),
        )
    )

    if is_2D:
        macro_file = outdir / "stardist_postprocessing.ijm"
        with open(str(macro_file), "w", encoding="utf-8") as f:
            f.write(
                DEEPIMAGEJ_MACRO.format(
                    probThresh=model.thresholds.prob, nmsThresh=model.thresholds.nms
                )
            )
        config["stardist"].update(postprocessing_macro=macro_file.name)

    n_inputs = len(input_names)
    assert n_inputs == 1
    input_config = dict(
        input_names=input_names,
        input_min_shape=[input_min_shape],
        input_step=[input_step],
        input_axes=[input_axes],
        input_data_range=[["-inf", "inf"]],
        preprocessing=[
            [
                dict(
                    name="scale_range",
                    kwargs=dict(
                        mode="per_sample",
                        axes=axes_norm.lower(),
                        min_percentile=min_percentile,
                        max_percentile=max_percentile,
                    ),
                )
            ]
        ],
    )

    n_outputs = len(output_names)
    output_config = dict(
        output_names=output_names,
        output_data_range=[["-inf", "inf"]] * n_outputs,
        output_axes=[output_axes] * n_outputs,
        output_reference=[input_names[0]] * n_outputs,
        output_scale=[output_scale] * n_outputs,
        output_offset=[output_offset] * n_outputs,
        halo=[halo] * n_outputs,
    )

    in_path = outdir / "test_input.npy"
    np.save(in_path, test_input[np.newaxis])

    if mode == "tensorflow_saved_model_bundle":
        test_outputs = _predict_tf(assets_uri, test_input_norm[np.newaxis])
    else:
        test_outputs = model.predict(test_input_norm)
        if mode == "keras_v3":
            prob, dist = test_outputs
            test_outputs = np.concatenate([prob[..., np.newaxis], dist], axis=-1)

    # out_paths = []
    # for i, out in enumerate(test_outputs):
    #     p = outdir / f"test_output{i}.npy"
    #     np.save(p, out)
    #     out_paths.append(p)
    assert n_outputs == 1
    out_paths = [outdir / "test_output.npy"]
    np.save(out_paths[0], test_outputs)

    from tensorflow import __version__ as tf_version

    data = dict(
        weight_uri=assets_uri,
        test_inputs=[in_path],
        test_outputs=out_paths,
        config=config,
        tensorflow_version=tf_version,
    )
    data.update(input_config)
    data.update(output_config)
    _files = [str(weights_file)]
    if is_2D:
        _files.append(str(macro_file))
    data.update(attachments=dict(files=_files))

    return data


def _build_model(name: str, outpath: Path, datapath: Path, **kwargs):
    """Build a bioimage.io model using the ModelDescr specification.
    
    Parameters
    ----------
    name: str
        Name of the model
    output_path: Path
        Path where to save the model package
    root: Path
        Path to the model assets
    kwargs: dict
        Model metadata from _get_weights_and_model_metadata
    """

    from bioimageio.spec.model.v0_5 import (
        ModelDescr, InputTensorDescr, OutputTensorDescr,
        BatchAxis, ChannelAxis, SpaceInputAxis, SpaceOutputAxisWithHalo,
        AxisId, TensorId, Identifier, ParameterizedSize,
        IntervalOrRatioDataDescr, FileDescr, SizeReference, WeightsDescr,
        Version, TensorflowSavedModelBundleWeightsDescr,
    )
    
    # Extract config information
    stardist_config = kwargs.get('config', {}).get('stardist', {}).get('config', {})
    n_rays = stardist_config.get('n_rays', {})
    halo = kwargs.get('halo', {})[0] # list
    grid = stardist_config.get('grid', {}) #. tuple
    n_dim = stardist_config.get('n_dim', {}) # 2D or 3D

    upsample_grid = kwargs.get('upsample_grid', True)

    min_size_y = kwargs.get('input_min_shape', {})[0][1]
    min_size_x = kwargs.get('input_min_shape', {})[0][2]
    step_y = kwargs.get('input_step', {})[0][1]
    step_x = kwargs.get('input_step', {})[0][2]

    n_channels_in = stardist_config.get('n_channel_in', {})
    channel_names_in = [f"input_{i:02d}" for i in range(n_channels_in)] if n_channels_in > 1 else ["input"]
    # n_channels_out = stardist_config.get('n_channel_out', {})
    channel_names_out = ["prob"] + [f"dist_{i:02d}" for i in range(n_rays)]

    # check if StarDist2D or StarDist3D model
    is_2d = n_dim == 2

    # Build input tensor description
    axes = stardist_config.get('axes', {})
    spatial_axes = [axis.lower() for axis in axes if axis in 'ZYX']
    grid_indices = list(range(len(spatial_axes)))
    min_in_shape = kwargs.get('input_min_shape', {})[0] # list
    steps = kwargs.get('input_step', {})[0] # list

    spatial_input_axes = []
    for i, axis_name in enumerate(spatial_axes):
        spatial_input_axes.append(
            SpaceInputAxis(
                id=AxisId(axis_name),
                size=ParameterizedSize(min=min_in_shape[i+1], step=steps[i+1]),
                scale=grid[grid_indices[i]] if not upsample_grid else 1
            )
        )

    model_inputs = [
        InputTensorDescr(
            id=TensorId("raw"),
            axes=[
                BatchAxis(),
                *spatial_input_axes,
                ChannelAxis(channel_names=[Identifier(name) for name in channel_names_in]),
            ],
            data=IntervalOrRatioDataDescr(type="float32"),
            test_tensor=FileDescr(source=(datapath / "test_input.npy")),
            sample_tensor=FileDescr(source=(datapath / "sample_input_0.tif")) if (datapath / "sample_input_0.tif").exists() else None,
        )
    ]

    # Build output tensor description
    spatial_output_axes = []
    for i, axis_name in enumerate(spatial_axes):
        spatial_output_axes.append(
            SpaceOutputAxisWithHalo(
                id=AxisId(axis_name),
                halo=halo[i+1] if not upsample_grid else halo[i+1] / grid[grid_indices[i]],
                size=SizeReference(tensor_id=TensorId("raw"), axis_id=AxisId(axis_name)),
                scale=grid[grid_indices[i]] if not upsample_grid else 1
            )
        )

    model_outputs = [
        OutputTensorDescr(
            id=TensorId("predictions"),
            axes=[
                BatchAxis(),
                *spatial_output_axes,
                ChannelAxis(channel_names=[Identifier(name) for name in channel_names_out]),
            ],
            test_tensor=FileDescr(source=(datapath / "test_output.npy")),
            sample_tensor=FileDescr(source=(datapath / "sample_output_0.tif")) if (datapath / "sample_output_0.tif").exists() else None,
        )
    ]

    # Weights
    # if kwargs.get('mode') == "keras_v3":
    #     weights = WeightsDescr(
    #         keras_hdf5=dict(
    #             source=kwargs.get('weight_uri'),
    #             tensorflow_version=kwargs.get('tensorflow_version')
    #         )
    #     )
    # else:
    tensorflow_saved_model_bundle_weights = TensorflowSavedModelBundleWeightsDescr(
        source=str(kwargs.get('weight_uri')),
        tensorflow_version=Version(kwargs.get('tensorflow_version')),
    )

    weights = WeightsDescr(
        tensorflow_saved_model_bundle=tensorflow_saved_model_bundle_weights
    )

    # Attachments
    attachments = []
    if kwargs.get('attachments').get('files'):
        for file_path in kwargs['attachments']['files']:
            attachments.append(FileDescr(
                source=str(file_path)
            ))

    # Create model description
    model = ModelDescr(
        name=name,
        description=kwargs.get('description', 'StarDist model'),
        documentation=kwargs.get('documentation', None),
        authors=kwargs.get('authors', []),
        cite=kwargs.get('cite', []),
        license=kwargs.get('license', []),
        git_repo=kwargs.get('git_repo', 'https://github.com/stardist/stardist'),
        tags=kwargs.get('tags', ['instance-segmentation', 'stardist']),
        config=kwargs.get('config', {}),
        inputs=model_inputs,
        outputs=model_outputs,
        weights=weights,
        attachments=attachments if attachments else None,
    )

    return model


def export_bioimageio(
    model,
    outpath,
    datapath,
    test_input,
    test_input_axes=None,
    test_input_norm_axes="ZYX",
    name=None,
    mode="tensorflow_saved_model_bundle",
    min_percentile=1.0,
    max_percentile=99.8,
    overwrite_spec_kwargs=None,
    generate_default_deps=False,
    upsample_grid=True,
):
    """Export stardist model into bioimage.io format, https://github.com/bioimage-io/spec-bioimage-io.

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
        the name of this model (default: None)
        if None, uses the (folder) name of the model (i.e. `model.name`)
    mode: str
        the export type for this model (default: "keras_v3", legacy: "tensorflow_saved_model_bundle")
    min_percentile: float
        min percentile to be used for image normalization (default: 1.0)
    max_percentile: float
        max percentile to be used for image normalization (default: 99.8)
    overwrite_spec_kwargs: dict or None
        spec keywords that should be overloaded (default: None)
    generate_default_deps: bool
        not required for bioimage.io, i.e. StarDist models don't need a dependencies field in rdf.yaml (default: False)
        if True, generate an environment.yaml file recording the python, bioimageio.core, stardist and tensorflow requirements
        from which a conda environment can be recreated to run this export
    """
    from bioimageio.spec import save_bioimageio_package

    isinstance(model, (StarDist2D, StarDist3D)) or _raise(
        ValueError("not a valid model")
    )
    0 <= min_percentile < max_percentile <= 100 or _raise(
        ValueError("invalid percentile values")
    )

    if name is None:
        name = model.name
    name = str(name)

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

    with tempfile.TemporaryDirectory() as _tmp_dir:
        tmp_dir = Path(_tmp_dir)
        kwargs = _get_stardist_metadata(tmp_dir, model, generate_default_deps)
        model_kwargs = _get_weights_and_model_metadata(
            tmp_dir,
            model,
            test_input,
            test_input_axes,
            test_input_norm_axes,
            mode,
            min_percentile=min_percentile,
            max_percentile=max_percentile,
            upsample_grid=upsample_grid,
        )
        kwargs.update(model_kwargs)
        if overwrite_spec_kwargs is not None:
            kwargs.update(overwrite_spec_kwargs)

        # bioimageio.core < 0.6.0 (legacy models)
        try:
            from bioimageio.core import build_model as _build_model_legacy
            model = _build_model_legacy(
                name=name,
                outpath=zip_path,
                add_deepimagej_config=(model.config.n_dim == 2),
                root=tmp_dir,
                mode=mode,
                **kwargs,
            )
            print(f"\nbioimage.io model with name '{name}' exported to '{zip_path}'")
        # bioimageio.core >= 0.6.0
        except ImportError:
            model = _build_model(
                name=name,
                outpath=zip_path,
                datapath=datapath,
                mode=mode,
                **kwargs)
            print(f"\nbioimage.io model with name '{name}' exported to '{zip_path}'")

        save_bioimageio_package(model, output_path=outpath)


def import_bioimageio(source: Union[str, Path], outpath: Union[str, Path]):
    """Import stardist model from bioimage.io format, https://github.com/bioimage-io/spec-bioimage-io.

    Load a model in bioimage.io format from the given `source` (e.g. path to zip file, URL)
    and convert it to a regular stardist model, which will be saved in the folder `outpath`.

    Parameters
    ----------
    source: str, Path
        bioimage.io resource (e.g. path, URL, bioimageio ID)
    outpath: str, Path
        folder to save the stardist model (must not exist previously)

    Returns
    -------
    StarDist2D or StarDist3D
        stardist model loaded from `outpath`

    """

    from io import BytesIO
    try:
        from bioimageio.spec import load_model_description
        from bioimageio.spec.model import v0_4, v0_5
        from bioimageio.spec.utils import extract_file_name, get_reader
    except Exception as e:
        raise RuntimeError(_BIOIMAGEIO_LIBRARIES_ARE_MISSING) from e

    outpath = Path(outpath)
    not outpath.exists() or _raise(FileExistsError(f"'{outpath}' already exists"))

    biomodel = load_model_description(source)

    stardist_config = None
    if isinstance(biomodel, v0_4.ModelDescr):
        if isinstance(biomodel.config, dict) and "stardist" in biomodel.config:
            stardist_config = biomodel.config["stardist"]
    elif isinstance(biomodel, v0_5.ModelDescr):
        if hasattr(biomodel.config, "stardist"):
            stardist_config = biomodel.config.stardist
    else:
        assert_never(biomodel)
    
    if stardist_config is None:
        raise RuntimeError("bioimage.io model not compatible, no stardist config found")

    config = stardist_config["config"]
    thresholds = stardist_config["thresholds"]
    weights = stardist_config["weights"]

    # handle weights sourcing from attachments
    weights_source = None
    if isinstance(biomodel, v0_4.ModelDescr):
        if biomodel.attachments is not None:
            for f in biomodel.attachments.files:
                if extract_file_name(f) == weights:
                    weights_source = f
                    break
    elif isinstance(biomodel, v0_5.ModelDescr):
        if hasattr(biomodel, "attachments"):
            for f in biomodel.attachments:
                if extract_file_name(f.source) == weights:
                    weights_source = f
                    break
        # if weights_source is None and hasattr(biomodel, "weights"):
        #     if (hasattr(biomodel.weights, "keras_hdf5") and biomodel.weights.keras_hdf5 is not None):
        #         weights_source = biomodel.weights.keras_hdf5.source
        #     elif hasattr(biomodel.weights, "tensorflow_saved_model_bundle"):
        #         weights_source = biomodel.weights.tensorflow_saved_model_bundle.source
    else:
        assert_never(biomodel)

    if weights_source is None:
        raise FileNotFoundError(f"couldn't find weights file '{weights}'")

    outpath.mkdir(parents=True)
    save_json(config, str(outpath / "config.json"))
    save_json(thresholds, str(outpath / "thresholds.json"))

    _has_keras_hdf5 = isinstance(biomodel, v0_4.ModelDescr or v0_5.ModelDescr) and hasattr(biomodel.weights, "keras_hdf5") and biomodel.weights.keras_hdf5 is not None

    # if _has_keras_hdf5:
    #     # extract .keras file for keras_v3 models
    #     with ZipFile(source) as source_zip:
    #         source_zip.extract(str(weights_source), outpath)
    # else:
        # copy h5 weights for legacy models
    with BytesIO(get_reader(weights_source).read()) as f, (outpath / "weights_bioimageio.h5").open(mode="wb") as out_f:
        shutil.copyfileobj(f, out_f)
        # with download(weights_source).path.open(mode="rb") as f, (outpath / "weights_bioimageio.h5").open(mode="wb") as out_f:
        #     shutil.copyfileobj(f, out_f)

    model_config = Config2D(**config) if config["n_dim"] == 2 else Config3D(**config)
    model_class = StarDist2D if config["n_dim"] == 2 else StarDist3D
    model = model_class(name=outpath.name, basedir=str(outpath.parent), config=model_config)

    # automatically load weights
    # if _has_keras_hdf5:
    #     try:
    #         import keras
    #     except ImportError:
    #         raise ImportError("Keras v3 export requires Keras 3.0.0 or higher")
        
    #     keras_file = outpath / str(weights_source)
    #     if not keras_file.exists():
    #         raise FileNotFoundError(f"Keras model file '{weights_source}' not found'")
        
    #     _keras_model = keras.models.load_model(keras_file)
    #     model.keras_model.set_weights(_keras_model.get_weights())
    # else:
    model.load_weights("weights_bioimageio.h5")

    return model
