import shutil
import tempfile
import warnings
from io import BytesIO
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from csbdeep.data import PercentileNormalizer
from csbdeep.utils import save_json
from csbdeep.utils.tf import IS_KERAS_3_PLUS, IS_TF_1, export_SavedModel

from .models.model2d import Config2D, StarDist2D
from .models.model3d import Config3D, StarDist3D

if TYPE_CHECKING:
    from typing import TypedDict

    import keras.models
    from bioimageio.core import Tensor
    from bioimageio.spec.model.v0_5 import Author, SpaceUnit
    from numpy.typing import NDArray
    from typing_extensions import Literal, NotRequired, TypeGuard

    class _AuthorDict(TypedDict):
        name: str
        github_user: NotRequired[Optional[str]]
        affiliation: NotRequired[Optional[str]]
        email: NotRequired[Optional[str]]


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
    "Required libraries are missing for bioimage.io model export/import.\n"
    "Please install StarDist as follows: pip install 'stardist[bioimageio]'\n"
    "(You do not need to uninstall StarDist first.)"
)


def export_bioimageio(
    model: Union[StarDist2D, StarDist3D],
    outpath: Union[str, Path],
    test_input: "NDArray[Any]",
    test_input_axes: str,
    *,
    jointly_normalize_channels: bool = False,
    name: Optional[str] = None,
    mode: 'Literal["keras_v3", "tensorflow_saved_model_bundle"]' = "keras_v3",
    min_percentile: float = 1.0,
    max_percentile: float = 99.8,
    upsample_grid: bool = False,
    authors: Sequence["_AuthorDict | Author"] = (),
    model_description: str = "A custom StarDist model",
    additional_tags: Sequence[str] = (),
    input_channel_names: Sequence[str] = ("intensity",),
    input_pixel_size: "Tuple[float, SpaceUnit] | Tuple[Literal[1], None]" = (
        1,
        None,
    ),
    overlap_label: Optional[int] = None,
) -> Path:  # TODO: update parameters in docstring
    """Export stardist model into bioimage.io format, https://github.com/bioimage-io/spec-bioimage-io.

    Parameters
    ----------
    model: StarDist2D, StarDist3D
        the model to convert
    outpath: str, Path
        where to save the model
    test_input: np.ndarray
        input image for generating test data
    test_input_axes: str
        axes string for `test_input`, e.g. "zyx" or "cyx"
    jointly_normalize_channels:
        if True, jointly normalize channels (e.g. for RGB input images) (default: False)
    name: str
        the name of this model (default: None)
        If None, uses the (folder) name of the model (i.e. `model.name`).
    mode: str
        the export type for this model (default: "keras_v3", legacy: "tensorflow_saved_model_bundle")
    min_percentile: float
        min percentile to be used for image normalization (default: 1.0)
    max_percentile: float
        max percentile to be used for image normalization (default: 99.8)
    upsample_grid: bool
        If True, upsamples the network output to the input shape.
        Note: this is currently mandatory for further use in Fiji.
    authors: Sequence[Union[_AuthorDict, "Author"]]
        List of authors to be included in the model description (default: empty tuple).
        Each author is desribed by a bioimageio.spec.model.v0_5.Author instance or a dict with a "name" key
        and optional keys "github_user", "affiliation", "email", and "orcid".
    model_description: str
        A short description of the model (default: "A custom StarDist model")
    additional_tags: Sequence[str]
        Additional tags to be included in the model description (default: empty tuple).
    input_channel_names: Sequence[str]
        Names of the input channels. (default: ("intensity",))
        Note: Default is only valid for single channel input.
    input_pixel_size: Tuple[float, SpaceUnit] | Tuple[Literal[1], None]
        Physical size of the input pixels (default: (1, None))
        If model.config.anisotropy is not None, `input_pixel_size` refers to the any axis with anisotropy 1.
        Other axes are scaled accordingly.
    overlap_label: int | None
        If not None, label the regions where polygons overlap with that value.
    """
    try:
        from bioimageio.spec import InvalidDescr, save_bioimageio_package
    except Exception as e:
        raise RuntimeError(_BIOIMAGEIO_LIBRARIES_ARE_MISSING) from e

    assert isinstance(model, (StarDist2D, StarDist3D))

    if not (0 <= min_percentile < max_percentile <= 100):
        raise ValueError("invalid percentile values")

    if name is None:
        name = str(model.name)

    outpath = Path(outpath)
    if outpath.suffix == "":
        zip_path = outpath / f"{name}.zip"
    elif outpath.suffix == ".zip":
        zip_path = outpath
    else:
        raise ValueError(f"outpath has to be a folder or zip file, got {outpath}")

    del outpath

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_descr = _create_model_descr(
            model_name=name,
            additional_tags=additional_tags,
            authors=[
                Author.model_validate(a) if isinstance(a, dict) else a for a in authors
            ],
            description=model_description,
            max_percentile=max_percentile,
            min_percentile=min_percentile,
            mode=mode,
            model=model,
            test_input=test_input,
            jointly_normalize_channels=jointly_normalize_channels,
            test_input_axes=test_input_axes,
            tmp_dir=Path(tmp_dir),
            upsample_grid=upsample_grid,
            input_channel_names=input_channel_names,
            input_space_unit=input_pixel_size[1],
            input_space_scale=input_pixel_size[0],
            overlap_label=overlap_label,
        )
        if isinstance(model_descr, InvalidDescr):
            model_descr.validation_summary.display()
            raise ValueError(
                f"Model description is invalid, cannot export model: {model_descr.get_reason()}"
            )

        _ = save_bioimageio_package(
            model_descr, output_path=zip_path, allow_invalid=True
        )

    print(f"\nbioimage.io model with name '{name}' exported to '{zip_path}'")
    return zip_path


def _prepare_test_input(
    test_input: "NDArray[Any]",
    test_input_axes: str,
    network_axes: str,
) -> "Tensor":
    """check and transpose test_input to match network_axes and save it"""
    try:
        from bioimageio.core import AxisId, Tensor
    except Exception as e:
        raise RuntimeError(_BIOIMAGEIO_LIBRARIES_ARE_MISSING) from e

    test_input_axes = test_input_axes.lower()
    network_axes = network_axes.lower()

    if len(test_input_axes) == len(test_input.squeeze().shape):
        test_input = test_input.squeeze()
    elif len(test_input_axes) != len(test_input.shape):
        raise ValueError(
            f"test_input_axes length {len(test_input_axes)} does not match test_input shape {test_input.shape}"
        )

    extra_input_axes = {a for a in test_input_axes if a not in network_axes}
    if extra_input_axes:
        raise ValueError(
            f"test_input_axes contains axes not in model.config.axes: {extra_input_axes}"
        )

    missing_input_axes = [a for a in network_axes if a not in test_input_axes]
    for a in missing_input_axes:
        test_input_axes = a + test_input_axes
        test_input = test_input[np.newaxis]

    test_input = test_input.transpose([test_input_axes.index(a) for a in network_axes])
    input_axes = [
        AxisId(a.replace("c", "channel").replace("b", "batch")) for a in network_axes
    ]
    if AxisId("batch") not in input_axes:
        input_axes.insert(0, AxisId("batch"))
        test_input = test_input[np.newaxis]

    return Tensor(test_input, input_axes)


def _create_model_descr(
    *,
    model_name: str,
    additional_tags: Sequence[str],
    authors: List["Author"],
    description: str,
    max_percentile: float,
    min_percentile: float,
    mode: 'Literal["keras_v3", "tensorflow_saved_model_bundle"]',
    model: Union[StarDist2D, StarDist3D],
    test_input: "NDArray[Any]",
    test_input_axes: str,
    input_channel_names: Sequence[str],
    input_space_unit: Optional["SpaceUnit"],
    input_space_scale: float,
    tmp_dir: Path,
    jointly_normalize_channels: bool,
    upsample_grid: bool,
    overlap_label: Optional[int],
):
    try:
        from bioimageio.core import MemberId, Tensor
        from bioimageio.core.io import save_tensor
        from bioimageio.spec import InvalidDescr
        from bioimageio.spec.model.v0_5 import (
            AxisId,
            BatchAxis,
            ChannelAxis,
            CiteEntry,
            Config,
            Doi,
            EnsureDtypeDescr,
            EnsureDtypeKwargs,
            FileDescr,
            HttpUrl,
            Identifier,
            InputTensorDescr,
            IntervalOrRatioDataDescr,
            KerasV3WeightsDescr,
            LicenseId,
            ModelDescr,
            OutputTensorDescr,
            ParameterizedSize,
            ScaleRangeDescr,
            ScaleRangeKwargs,
            SizeReference,
            SpaceInputAxis,
            SpaceOutputAxisWithHalo,
            StardistPostprocessingDescr,
            StardistPostprocessingKwargs2D,
            StardistPostprocessingKwargs3D,
            TensorflowSavedModelBundleWeightsDescr,
            TensorId,
            Version,
            WeightsDescr,
        )
        from importlib_metadata import metadata
        from tensorflow import __version__ as tf_version
        from typing_extensions import assert_never
    except Exception as e:
        raise RuntimeError(_BIOIMAGEIO_LIBRARIES_ARE_MISSING) from e

    # use stardist axes convenction for test_input_axes
    test_input_axes = test_input_axes.upper().replace("B", "S")

    # make sure test input has a singleton sample dimension
    if "S" in test_input_axes and test_input.shape[test_input_axes.index("S")] != 1:
        raise ValueError(
            "test_input may only have a singleton sample (batch) dimension (or none)."
        )
    # else:
    #     test_input_axes = "S" + test_input_axes
    #     test_input = test_input[np.newaxis]

    # make sure test input has an explicit channel axis
    if "C" not in test_input_axes and "C" in model.config.axes:
        test_input_axes = test_input_axes + "C"
        test_input = test_input[..., np.newaxis]

    assert isinstance(model.config, (Config2D, Config3D)), (
        "expected model.config to be an instance of Config2D or Config3D"
    )
    n_dim = model.config.n_dim
    assert n_dim in (2, 3), "expected model.config.n_dim to be 2 or 3"
    if n_dim == 2 and overlap_label is not None:
        raise NotImplementedError("Overlap label for 2D not yet implemented")

    backbone = model.config.backbone
    n_channel_in = model.config.n_channel_in
    # assert isinstance(n_channel_in, int) and n_channel_in > 0, (
    #     "expected model.config.n_channel_in to be a positive integer"
    # )
    if "C" not in test_input_axes:
        if n_channel_in != 1:
            raise ValueError(
                f"test_input has no channel axis, but model.config.n_channel_in is {n_channel_in}"
            )
    elif test_input.shape[test_input_axes.index("C")] != n_channel_in:
        raise ValueError(
            f"test_input has {test_input.shape[test_input_axes.index('C')]} channels, but model.config.n_channel_in is {n_channel_in}"
        )
    if len(input_channel_names) != n_channel_in:
        raise ValueError(
            f"input_channel_names has length {len(input_channel_names)}, but model.config.n_channel_in is {n_channel_in}"
        )

    grid = model.config.grid
    network_axes = model.config.axes
    assert isinstance(network_axes, str), "expected model.config.axes to be a string"

    stardist_data = metadata("stardist")
    assert stardist_data is not None, "Could not load stardist package metadata"

    original_weights_file = tmp_dir / "original_stardist_weights.h5"
    model.keras_model.save_weights(str(original_weights_file))  # pyright: ignore[reportAttributeAccessIssue]
    network_axes_bioimageio = [
        AxisId("batch" if a == "b" else "channel" if a == "c" else a)
        for a in network_axes.lower().replace("s", "b")
    ]
    spatial_axes_bioimageio = [
        a
        for a in network_axes_bioimageio
        if a not in (AxisId("batch"), AxisId("channel"))
    ]
    assert len(spatial_axes_bioimageio) == n_dim, (
        f"expected {n_dim} spatial axes, but got {spatial_axes_bioimageio}"
    )
    input_shape_min_wo_overlap = dict(
        zip(network_axes_bioimageio, map(int, model._axes_div_by(network_axes)))
    )
    input_overlap = dict(
        zip(
            network_axes_bioimageio,
            map(int, model._axes_tile_overlap(network_axes)),
        )
    )
    input_shape_min = {
        a: input_shape_min_wo_overlap[a] + 2 * input_overlap[a]
        for a in network_axes_bioimageio
    }
    # round up to be divisible by div_by
    input_shape_min = {
        a: input_shape_min[a] + (-input_shape_min[a] % input_shape_min_wo_overlap[a])
        for a in network_axes_bioimageio
    }
    input_shape_step = input_shape_min_wo_overlap

    # use stardist instance prediction to generate test output independent of bioimageio
    instances = model.predict_instances(
        test_input,  # pyright: ignore
        axes=test_input_axes,
        normalizer=PercentileNormalizer(
            pmin=min_percentile,  # pyright: ignore[reportArgumentType]
            pmax=max_percentile,
            do_after=False,
            dtype=np.float32,
        ),
        return_predict=False,
    )[0]
    assert isinstance(instances, np.ndarray), (
        "expected model.predict_instances to return a tuple with a numpy array"
    )
    # add explicit batch and channel axis
    assert instances.ndim == n_dim, (
        f"Expected {n_dim} output dimensions, got {instances.ndim}"
    )
    instances = instances[np.newaxis, ..., np.newaxis]
    test_output_instance_labels_path = tmp_dir / "test_output_instances.npy"
    sample_output_instance_labels_path = test_output_instance_labels_path.with_suffix(
        ".tiff"
    )
    np.save(test_output_instance_labels_path, instances, allow_pickle=False)
    save_tensor(
        sample_output_instance_labels_path,
        Tensor(instances, dims=["batch", *spatial_axes_bioimageio, "channel"]),
    )
    stardist_config = dict(
        python_version=stardist_data["Version"],
        thresholds=dict(model.thresholds._asdict()),
        weights=original_weights_file.name,
        config=vars(model.config),
    )

    if n_dim == 2:
        macro_file = tmp_dir / "stardist_postprocessing.ijm"
        _ = macro_file.write_text(
            DEEPIMAGEJ_MACRO.format(
                probThresh=model.thresholds.prob,
                nmsThresh=model.thresholds.nms,
            ),
            encoding="utf-8",
        )
        stardist_config["postprocessing_macro"] = macro_file.name

    config_descr = Config.model_validate({"stardist": stardist_config})

    # patched_model = copy.copy(model)
    # del model
    # patched_model.keras_model = patched_keras_model

    input_tensor = _prepare_test_input(test_input, test_input_axes, network_axes)
    test_input_path = tmp_dir / "test_input.npy"
    np.save(test_input_path, input_tensor.to_numpy(), allow_pickle=False)
    sample_input_path = test_input_path.with_suffix(".tiff")
    save_tensor(sample_input_path, input_tensor)

    preprocessing = [
        ScaleRangeDescr(
            kwargs=ScaleRangeKwargs(
                axes=[
                    a
                    for a in input_tensor.dims
                    if (jointly_normalize_channels or a != AxisId("channel"))
                    and a != AxisId("batch")
                ],
                min_percentile=min_percentile,
                max_percentile=max_percentile,
                eps=1e-20,
            )
        ),
        EnsureDtypeDescr(kwargs=EnsureDtypeKwargs(dtype="float32")),
    ]

    patched_keras_model = _get_patched_keras_model(model, upsample_grid=upsample_grid)

    if mode == "keras_v3":
        import keras

        if not hasattr(keras, "__version__") or Version(keras.__version__) < Version(
            "3.0.0"
        ):
            raise NotImplementedError("Keras v3 export requires Keras 3.0.0 or higher")

        weights_path = tmp_dir / "model.keras"
        patched_keras_model.save(weights_path)
        weights = WeightsDescr(
            keras_v3=KerasV3WeightsDescr(
                source=weights_path,
                backend=("tensorflow", Version(tf_version)),
                keras_version=Version(keras.__version__),
            )
        )

    elif mode == "tensorflow_saved_model_bundle":
        weights_path = tmp_dir / "TF_SavedModel.zip"
        export_SavedModel(patched_keras_model, str(weights_path))
        weights = WeightsDescr(
            tensorflow_saved_model_bundle=TensorflowSavedModelBundleWeightsDescr(
                source=weights_path,
                tensorflow_version=Version(tf_version),
            )
        )

    else:
        assert_never(mode)

    # if mode == "keras_v3":
    #     output_scale = [1] * (ndim_tensor)
    #     output_scale[output_axes.index("c")] = 0

    #     output_offset = [0.0] * (ndim_tensor)
    #     output_offset[output_axes.index("c")] = output_n_channels / 2.0

    # elif mode == "tensorflow_saved_model_bundle":
    #     # regarding input/output names: https://github.com/CSBDeep/CSBDeep/blob/b0d2f5f344ebe65a9b4c3007f4567fe74268c813/csbdeep/utils/tf.py#L193-L194
    #     input_names = ["input"]
    #     output_names = ["output"]
    #     # the output shape is computed from the input shape using
    #     # output_shape[i] = output_scale[i] * input_shape[i] + 2 * output_offset[i]
    #     # same shape as input except for the channel dimension
    #     output_scale = [1] * (ndim_tensor)
    #     output_scale[output_axes.index("c")] = 0
    #     # no offset, except for the input axes, where it is output channel / 2
    #     output_offset = [0.0] * (ndim_tensor)
    #     output_offset[output_axes.index("c")] = output_n_channels / 2.0

    documentation = tmp_dir / "README.md"
    _ = documentation.write_text(
        "# StarDist Model\n"
        + "This is a model for object detection with star-convex shapes.\n"
        + "Please see the [StarDist repository](https://github.com/stardist/stardist) for details."
    )

    input_descr = InputTensorDescr(
        id=TensorId("input"),
        axes=[
            BatchAxis(),
            *(
                SpaceInputAxis(
                    id=a,
                    size=ParameterizedSize(
                        min=input_shape_min[a], step=input_shape_step[a]
                    ),
                    unit=input_space_unit,
                    scale=input_space_scale,
                )
                for a in spatial_axes_bioimageio
            ),
            ChannelAxis(
                channel_names=[Identifier(name) for name in input_channel_names]
            ),
        ],
        data=IntervalOrRatioDataDescr(type="float32"),
        test_tensor=FileDescr(source=test_input_path),
        sample_tensor=FileDescr(source=sample_input_path),
        preprocessing=preprocessing,
    )

    grid = model.config.grid
    postprocessing_grid = (1,) * n_dim if upsample_grid else grid
    # scale up border region when upsampling prediction grid
    b = min(grid) * 2 if upsample_grid else 2
    if n_dim == 2:
        assert overlap_label is None
        assert len(postprocessing_grid) == 2
        stardist_postproc_kwargs = StardistPostprocessingKwargs2D(
            prob_threshold=model.thresholds.prob,
            nms_threshold=model.thresholds.nms,
            grid=postprocessing_grid,
            b=b,
        )
    else:
        assert len(postprocessing_grid) == 3
        stardist_postproc_kwargs = StardistPostprocessingKwargs3D(
            prob_threshold=model.thresholds.prob,
            nms_threshold=model.thresholds.nms,
            grid=postprocessing_grid,
            n_rays=model.config.n_rays,
            anisotropy=model.config.anisotropy or (1.0, 1.0, 1.0),
            b=b,
            overlap_label=overlap_label,
        )
    output_descr = OutputTensorDescr(
        id=TensorId("instance_labels"),
        axes=[
            BatchAxis(),
            *(
                SpaceOutputAxisWithHalo(
                    id=a,
                    halo=input_overlap[a],
                    size=SizeReference(tensor_id=MemberId("input"), axis_id=a),
                    scale=input_space_scale,
                    unit=input_space_unit,
                )
                for a in spatial_axes_bioimageio
            ),
            ChannelAxis(
                description="Background is labeled with 0, instances are labeled with consecutive integers starting from 1.",
                channel_names=[Identifier("instance_labels")],
            ),
        ],
        data=IntervalOrRatioDataDescr(type="int32"),
        test_tensor=FileDescr(source=test_output_instance_labels_path),
        sample_tensor=FileDescr(source=sample_output_instance_labels_path),
        postprocessing=[StardistPostprocessingDescr(kwargs=stardist_postproc_kwargs)],
    )

    descr = ModelDescr.load_from_kwargs(
        name=model_name,
        description=description,
        inputs=[input_descr],
        outputs=[output_descr],
        weights=weights,
        authors=authors,
        git_repo=HttpUrl(stardist_data["Home-Page"]),
        license=LicenseId(stardist_data["License"]),
        cite=[
            CiteEntry(
                text="Cell Detection with Star-Convex Polygons",
                doi=Doi("10.1007/978-3-030-00934-2_30"),
            ),
            CiteEntry(
                text="Star-convex Polyhedra for 3D Object Detection and Segmentation in Microscopy",
                doi=Doi("10.1109/WACV45572.2020.9093435"),
            ),
        ],
        config=config_descr,
        tags=sorted(
            {
                *additional_tags,
                "fiji",
                "instance-segmentation",
                "stardist",
                "tensorflow",
                "keras",
                backbone,
                f"{n_dim}d",
            }
        ),
        documentation=documentation,
        attachments=[FileDescr(source=original_weights_file)],
    )
    if isinstance(descr, InvalidDescr):
        descr.validation_summary.display()
        warnings.warn(f"Invalid model description: {descr.get_reason()}.")

    return descr


def _get_patched_keras_model(
    model: Union[StarDist2D, StarDist3D], upsample_grid: bool
) -> "keras.models.Model":
    """Incorporate stardist specific logic into the keras model.

    This includes:
    - upsampling the outputs if grid > (1,1) and upsample_grid is True,
      to ensure compatibility with the CSBDeep Fiji plugin
    - Clip small dist values to prevent problems with Qhull
      (taken from _predict_generator() base.py).
    """
    if IS_TF_1 or IS_KERAS_3_PLUS:
        from keras.layers import (
            Concatenate,
            Conv2DTranspose,
            Conv3DTranspose,
            ReLU,
            UpSampling2D,
            UpSampling3D,
        )
        from keras.models import Model
    else:
        from tensorflow.keras.layers import (
            Concatenate,  # pyright: ignore[reportAttributeAccessIssue]
            Conv2DTranspose,  # pyright: ignore[reportAttributeAccessIssue]
            Conv3DTranspose,  # pyright: ignore[reportAttributeAccessIssue]
            ReLU,
            UpSampling2D,  # pyright: ignore[reportAttributeAccessIssue]
            UpSampling3D,  # pyright: ignore[reportAttributeAccessIssue]
        )
        from tensorflow.keras.models import Model

    if model.config.n_classes is not None:
        warnings.warn(
            "multi-class mode not supported yet, removing classification output from exported model"
        )

    assert isinstance(model.config, (Config2D, Config3D)), (
        "expected model.config to be an instance of Config2D or Config3D"
    )
    grid = model.config.grid
    prob = model.keras_model.outputs[0]  # pyright: ignore
    dist = model.keras_model.outputs[1]  # pyright: ignore
    assert model.config.n_dim in (2, 3)

    if upsample_grid and any(g > 1 for g in grid):
        # CSBDeep Fiji plugin needs same size input/output
        # -> we need to upsample the outputs if grid > (1,1)
        # note: upsampling prob with a transposed convolution creates sparse
        #       prob output with less candidates than with standard upsampling
        conv_transpose = Conv2DTranspose if model.config.n_dim == 2 else Conv3DTranspose
        upsampling = UpSampling2D if model.config.n_dim == 2 else UpSampling3D
        prob = conv_transpose(
            1,
            (1,) * model.config.n_dim,
            strides=grid,
            padding="same",
            kernel_initializer="ones",
            use_bias=False,
        )(prob)
        dist = upsampling(grid)(dist)

    # Clip dist to 1e-3 to avoid problems with Qhull algorithm
    # (using ReLU to clip for comatibility with Keras 2)
    dist_clipped = ReLU(threshold=1e-3)(dist)

    patched_outputs = Concatenate()([prob, dist_clipped])
    return Model(model.keras_model.inputs[0], patched_outputs)  # pyright: ignore


def import_bioimageio(source: Union[str, Path], outpath: Union[str, Path]):
    """Import stardist model from bioimage.io format, https://github.com/bioimage-io/spec-bioimage-io.

    Load a model in bioimage.io format from the given `source` (e.g. path to zip file, URL, or bioimage.io nickname)
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

    try:
        from bioimageio.spec import load_model_description
        from bioimageio.spec.model import v0_4, v0_5
        from bioimageio.spec.utils import extract_file_name, get_reader
        from typing_extensions import assert_never
    except Exception as e:
        raise RuntimeError(_BIOIMAGEIO_LIBRARIES_ARE_MISSING) from e

    outpath = Path(outpath)
    if outpath.exists():
        raise FileExistsError(f"'{outpath}' already exists")

    biomodel = load_model_description(source)

    stardist_config = None
    if isinstance(biomodel, v0_4.ModelDescr):
        if "stardist" in biomodel.config:
            stardist_config = biomodel.config["stardist"]
    elif isinstance(biomodel, v0_5.ModelDescr):
        if hasattr(biomodel.config, "stardist"):
            stardist_config = biomodel.config.stardist
    else:
        assert_never(biomodel)

    if stardist_config is None:
        raise RuntimeError("bioimage.io model not compatible, no stardist config found")

    assert isinstance(stardist_config, dict), "expected stardist config to be a dict"
    config = stardist_config["config"]

    def is_kwargs(v: Any) -> "TypeGuard[Dict[str, Any]]":
        return isinstance(v, dict) and all(
            isinstance(k, str)
            for k in v  # pyright: ignore[reportUnknownVariableType]
        )

    assert is_kwargs(config), (
        "expected config.stardist.config to be a dict with string keys"
    )

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
    else:
        assert_never(biomodel)

    if weights_source is None:
        raise FileNotFoundError(f"couldn't find weights file '{weights}'")

    outpath.mkdir(parents=True)
    save_json(config, str(outpath / "config.json"))
    save_json(thresholds, str(outpath / "thresholds.json"))

    with BytesIO(get_reader(weights_source).read()) as f:
        with (outpath / "weights_bioimageio.h5").open(mode="wb") as out_f:
            shutil.copyfileobj(f, out_f)

    if config["n_dim"] == 2:
        model = StarDist2D(
            name=outpath.name, basedir=str(outpath.parent), config=Config2D(**config)
        )
    else:
        model = StarDist3D(
            name=outpath.name, basedir=str(outpath.parent), config=Config3D(**config)
        )

    model.load_weights("weights_bioimageio.h5")

    return model
