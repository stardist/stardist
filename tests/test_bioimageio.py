import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence, Type, Union

import numpy as np
import pytest

from stardist.bioimageio_utils import export_bioimageio, import_bioimageio
from stardist.data import test_image_he_2d as _test_image_2d_rgb
from stardist.data import test_image_nuclei_2d as _test_image_2d
from stardist.data import test_image_nuclei_3d as _test_image_3d
from stardist.models import StarDist2D, StarDist3D

try:
    import bioimageio.core  # noqa
except ImportError:
    bioimageio_missing = True
else:
    bioimageio_missing = False

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _test_pretrained(
    tmp_path: Path,
    model_name: str,
    test_image: "NDArray[Any]",
    test_image_axes: str,
    model_type: Type[Union[StarDist2D, StarDist3D]],
    jointly_normalize_channels: bool,
    upsample_grid: bool,
    input_channel_names: Sequence[str],
):
    from bioimageio.core import test_model

    model = model_type.from_pretrained(model_name)
    assert model is not None
    # export model
    export_path = tmp_path / f"{model_name}.zip"
    export_bioimageio(
        model,
        export_path,
        test_input=test_image,
        test_input_axes=test_image_axes,
        jointly_normalize_channels=jointly_normalize_channels,
        upsample_grid=upsample_grid,
        input_channel_names=input_channel_names,
    )
    assert export_path.exists()
    # test exported model
    res = test_model(export_path)
    assert res.status == "passed", res.display()
    # import exported model
    import_path = tmp_path / f"{model_name}_imported"
    model_imported = import_bioimageio(export_path, import_path)

    # test that model and imported exported model are equal
    def _n(d):
        # normalize dict (especially tuples -> lists)
        return json.loads(json.dumps(d))

    assert _n(vars(model.config)) == _n(vars(model_imported.config))
    assert _n(model.thresholds._asdict()) == _n(model_imported.thresholds._asdict())
    assert all(
        np.allclose(u, v)
        for u, v in zip(
            model.keras_model.get_weights(),  # pyright: ignore[reportAttributeAccessIssue]
            model_imported.keras_model.get_weights(),  # pyright: ignore[reportAttributeAccessIssue]
        )
    )


@pytest.mark.skipif(bioimageio_missing, reason="Requires bioimageio dependencies")
@pytest.mark.parametrize("upsample_grid", [False, True])
def test_pretrained_fluo(tmp_path: Path, upsample_grid: bool):
    test_image = _test_image_2d()
    assert isinstance(test_image, np.ndarray)
    model_name = "2D_versatile_fluo"
    _test_pretrained(
        tmp_path,
        model_name,
        test_image,
        test_image_axes="YX",
        model_type=StarDist2D,
        jointly_normalize_channels=False,
        upsample_grid=upsample_grid,
        input_channel_names=["intensity"],
    )


@pytest.mark.skipif(bioimageio_missing, reason="Requires bioimageio dependencies")
@pytest.mark.parametrize("jointly_normalize_channels", [True, False])
@pytest.mark.parametrize("upsample_grid", [False, True])
def test_pretrained_he(
    tmp_path: Path, jointly_normalize_channels: bool, upsample_grid: bool
):
    test_image = _test_image_2d_rgb()
    model_name = "2D_versatile_he"
    _test_pretrained(
        tmp_path,
        model_name,
        test_image,
        test_image_axes="YXC",
        model_type=StarDist2D,
        jointly_normalize_channels=jointly_normalize_channels,
        upsample_grid=upsample_grid,
        input_channel_names="rgb",
    )


@pytest.mark.skipif(bioimageio_missing, reason="Requires bioimageio dependencies")
@pytest.mark.parametrize("upsample_grid", [False, True])
def test_pretrained_3d(tmp_path: Path, upsample_grid: bool):
    test_image = _test_image_3d()
    assert isinstance(test_image, np.ndarray)
    model_name = "3D_demo"
    _test_pretrained(
        tmp_path,
        model_name,
        test_image,
        test_image_axes="ZYX",
        model_type=StarDist3D,
        jointly_normalize_channels=False,
        upsample_grid=upsample_grid,
        input_channel_names=["intensity"],
    )
