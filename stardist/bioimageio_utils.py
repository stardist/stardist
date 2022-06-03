from pathlib import Path
from pkg_resources import get_distribution
from zipfile import ZipFile
import numpy as np
import tempfile
from distutils.version import LooseVersion
from csbdeep.utils import axes_check_and_normalize, normalize, _raise


DEEPIMAGEJ_MACRO = \
"""
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


def _import(error=True):
    try:
        from importlib_metadata import metadata
        from bioimageio.core.build_spec import build_model # type: ignore
        import xarray as xr
        import bioimageio.core # type: ignore
    except ImportError:
        if error:
            raise RuntimeError(
                "Required libraries are missing for bioimage.io model export.\n"
                "Please install StarDist as follows: pip install 'stardist[bioimageio]'\n"
                "(You do not need to uninstall StarDist first.)"
            )
        else:
            return None
    return metadata, build_model, bioimageio.core, xr


def _create_stardist_dependencies(outdir):
    from ruamel.yaml import YAML
    from tensorflow import __version__ as tf_version
    from . import __version__ as stardist_version
    pkg_info = get_distribution("stardist")
    # dependencies that start with the name "bioimageio" will be added as conda dependencies
    reqs_conda = [str(req) for req in pkg_info.requires(extras=['bioimageio']) if str(req).startswith('bioimageio')]
    # only stardist and tensorflow as pip dependencies
    tf_major, tf_minor = LooseVersion(tf_version).version[:2]
    reqs_pip = (f"stardist>={stardist_version}", f"tensorflow>={tf_major}.{tf_minor},<{tf_major+1}")
    # conda environment
    env = dict(
        name = 'stardist',
        channels = ['defaults', 'conda-forge'],
        dependencies = [
            ('python>=3.7,<3.8' if tf_major == 1 else 'python>=3.7'),
            *reqs_conda,
            'pip', {'pip': reqs_pip},
        ],
    )
    yaml = YAML(typ='safe')
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


def _get_stardist_metadata(outdir, model):
    metadata, *_ = _import()
    package_data = metadata("stardist")
    doi_2d = "https://doi.org/10.1007/978-3-030-00934-2_30"
    doi_3d = "https://doi.org/10.1109/WACV45572.2020.9093435"
    authors = {
        'Martin Weigert': dict(name='Martin Weigert', github_user='maweigert'),
        'Uwe Schmidt': dict(name='Uwe Schmidt', github_user='uschmidt83'),
    }
    data = dict(
        description=package_data["Summary"],
        authors=list(authors.get(name.strip(),dict(name=name.strip())) for name in package_data["Author"].split(",")),
        git_repo=package_data["Home-Page"],
        license=package_data["License"],
        dependencies=_create_stardist_dependencies(outdir),
        cite=[{"text": "Cell Detection with Star-Convex Polygons", "doi": doi_2d},
              {"text": "Star-convex Polyhedra for 3D Object Detection and Segmentation in Microscopy", "doi": doi_3d}],
        tags=[
            'fluorescence-light-microscopy', 'whole-slide-imaging', 'other', # modality
            f'{model.config.n_dim}d', # dims
            'cells', 'nuclei', # content
            'tensorflow', # framework
            'fiji', # software
            'unet', # network
            'instance-segmentation', 'object-detection', # task
            'stardist',
        ],
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
        img=test_input,
        axes=test_input_axes,
        normalizer=None,
        n_tiles=None,
        show_tile_progress=False,
        predict_kwargs={},
    )

    # normalization axes string and numeric indices
    axes_norm = set(axes_net).intersection(set(axes_check_and_normalize(test_input_norm_axes, disallowed='S')))
    axes_norm = "".join(a for a in axes_net if a in axes_norm)  # preserve order of axes_net
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

    # add the batch axis to shape and step
    input_min_shape = [1] + input_min_shape
    input_step = [0] + input_step

    # the axes strings in bioimageio convention
    input_axes = "b" + net_axes_in.lower()
    output_axes = "b" + net_axes_out.lower()

    if mode == "keras_hdf5":
        output_names = ("prob", "dist") + (("class_prob",) if model._is_multiclass() else ())
        output_n_channels = (1, model.config.n_rays,) + ((1,) if model._is_multiclass() else ())
        # the output shape is computed from the input shape using
        # output_shape[i] = output_scale[i] * input_shape[i] + 2 * output_offset[i]
        output_scale = [1]+list(1/g for g in model.config.grid) + [0]
        output_offset = [0]*(ndim_tensor)

    elif mode == "tensorflow_saved_model_bundle":
        if model._is_multiclass():
            raise NotImplementedError("Tensorflow SavedModel not supported for multiclass models yet")
        # regarding input/output names: https://github.com/CSBDeep/CSBDeep/blob/b0d2f5f344ebe65a9b4c3007f4567fe74268c813/csbdeep/utils/tf.py#L193-L194
        input_names = ["input"]
        output_names = ["output"]
        output_n_channels = (1 + model.config.n_rays,)
        # the output shape is computed from the input shape using
        # output_shape[i] = output_scale[i] * input_shape[i] + 2 * output_offset[i]
        # same shape as input except for the channel dimension
        output_scale = [1]*(ndim_tensor)
        output_scale[output_axes.index("c")] = 0
        # no offset, except for the input axes, where it is output channel / 2
        output_offset = [0.0]*(ndim_tensor)
        output_offset[output_axes.index("c")] = output_n_channels[0] / 2.0

    assert all(s in (0, 1) for s in output_scale), "halo computation assumption violated"
    halo = model._axes_tile_overlap(output_axes.replace('b', 's'))
    halo = [int(np.ceil(v/8)*8) for v in halo]  # optional: round up to be divisible by 8

    # the output shape needs to be valid after cropping the halo, so we add the halo to the input min shape
    input_min_shape = [ms + 2 * ha for ms, ha in zip(input_min_shape, halo)]

    # make sure the input min shape is still divisible by the min axis divisor
    input_min_shape = input_min_shape[:1] + [ms + (-ms % div_by) for ms, div_by in zip(input_min_shape[1:], axes_net_div_by)]
    assert all(ms % div_by == 0 for ms, div_by in zip(input_min_shape[1:], axes_net_div_by))

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
        with open(str(macro_file), 'w', encoding='utf-8') as f:
            f.write(DEEPIMAGEJ_MACRO.format(probThresh=model.thresholds.prob, nmsThresh=model.thresholds.nms))
        config['stardist'].update(postprocessing_macro=macro_file.name)

    n_inputs = len(input_names)
    assert n_inputs == 1
    input_config = dict(
        input_names=input_names,
        input_min_shape=[input_min_shape],
        input_step=[input_step],
        input_axes=[input_axes],
        input_data_range=[["-inf", "inf"]],
        preprocessing=[[dict(
            name="scale_range",
            kwargs=dict(
                mode="per_sample",
                axes=axes_norm.lower(),
                min_percentile=min_percentile,
                max_percentile=max_percentile,
            ))]]
        )

    n_outputs = len(output_names)
    output_config = dict(
        output_names=output_names,
        output_data_range=[["-inf", "inf"]] * n_outputs,
        output_axes=[output_axes] * n_outputs,
        output_reference=[input_names[0]] * n_outputs,
        output_scale=[output_scale] * n_outputs,
        output_offset=[output_offset] * n_outputs,
        halo=[halo] * n_outputs
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
    out_paths = [outdir / "test_output.npy"]
    np.save(out_paths[0], test_outputs)

    from tensorflow import __version__ as tf_version
    data = dict(weight_uri=assets_uri, test_inputs=[in_path], test_outputs=out_paths,
                config=config, tensorflow_version=tf_version)
    data.update(input_config)
    data.update(output_config)
    _files = [str(weights_file)]
    if is_2D:
        _files.append(str(macro_file))
    data.update(attachments=dict(files=_files))

    return data


def export_bioimageio(
    model,
    outpath,
    test_input,
    test_input_axes=None,
    test_input_norm_axes='ZYX',
    name=None,
    mode="tensorflow_saved_model_bundle",
    min_percentile=1.0,
    max_percentile=99.8,
    overwrite_spec_kwargs=None,
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
        the export type for this model (default: "tensorflow_saved_model_bundle")
    min_percentile: float
        min percentile to be used for image normalization (default: 1.0)
    max_percentile: float
        max percentile to be used for image normalization (default: 99.8)
    overwrite_spec_kwargs: dict or None
        spec keywords that should be overloaded (default: None)
    """
    _, build_model, *_ = _import()
    from .models import StarDist2D, StarDist3D
    isinstance(model, (StarDist2D, StarDist3D)) or _raise(ValueError("not a valid model"))
    0 <= min_percentile < max_percentile <= 100 or _raise(ValueError("invalid percentile values"))

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
        kwargs = _get_stardist_metadata(tmp_dir, model)
        model_kwargs = _get_weights_and_model_metadata(tmp_dir, model, test_input, test_input_axes, test_input_norm_axes, mode,
                                                       min_percentile=min_percentile, max_percentile=max_percentile)
        kwargs.update(model_kwargs)
        if overwrite_spec_kwargs is not None:
            kwargs.update(overwrite_spec_kwargs)

        build_model(name=name, output_path=zip_path, add_deepimagej_config=(model.config.n_dim==2), root=tmp_dir, **kwargs)
        print(f"\nbioimage.io model with name '{name}' exported to '{zip_path}'")


def import_bioimageio(source, outpath):
    """Import stardist model from bioimage.io format, https://github.com/bioimage-io/spec-bioimage-io.

    Load a model in bioimage.io format from the given `source` (e.g. path to zip file, URL)
    and convert it to a regular stardist model, which will be saved in the folder `outpath`.

    Parameters
    ----------
    source: str, Path
        bioimage.io resource (e.g. path, URL)
    outpath: str, Path
        folder to save the stardist model (must not exist previously)

    Returns
    -------
    StarDist2D or StarDist3D
        stardist model loaded from `outpath`

    """
    import shutil, uuid
    from csbdeep.utils import save_json
    from .models import StarDist2D, StarDist3D
    *_, bioimageio_core, _ = _import()

    outpath = Path(outpath)
    not outpath.exists() or _raise(FileExistsError(f"'{outpath}' already exists"))

    with tempfile.TemporaryDirectory() as _tmp_dir:
        tmp_dir = Path(_tmp_dir)
        # download the full model content to a temporary folder
        zip_path = tmp_dir / f"{str(uuid.uuid4())}.zip"
        bioimageio_core.export_resource_package(source, output_path=zip_path)
        with ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)
        zip_path.unlink()
        rdf_path = tmp_dir / "rdf.yaml"
        biomodel = bioimageio_core.load_resource_description(rdf_path)

        # read the stardist specific content
        'stardist' in biomodel.config or _raise(RuntimeError("bioimage.io model not compatible"))
        config = biomodel.config['stardist']['config']
        thresholds = biomodel.config['stardist']['thresholds']
        weights = biomodel.config['stardist']['weights']

        # make sure that the keras weights are in the attachments
        weights_file = None
        for f in biomodel.attachments.files:
            if f.name == weights and f.exists():
                weights_file = f
                break
        weights_file is not None or _raise(FileNotFoundError(f"couldn't find weights file '{weights}'"))

        # save the config and threshold to json, and weights to hdf5 to enable loading as stardist model
        # copy bioimageio files to separate sub-folder
        outpath.mkdir(parents=True)
        save_json(config, str(outpath / 'config.json'))
        save_json(thresholds, str(outpath / 'thresholds.json'))
        shutil.copy(str(weights_file), str(outpath / "weights_bioimageio.h5"))
        shutil.copytree(str(tmp_dir), str(outpath / "bioimageio"))

    model_class = (StarDist2D if config['n_dim'] == 2 else StarDist3D)
    model = model_class(None, outpath.name, basedir=str(outpath.parent))

    return model
