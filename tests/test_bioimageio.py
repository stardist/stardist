import pytest
import json
import numpy as np
from stardist.models import StarDist2D, StarDist3D
from stardist.data import test_image_nuclei_2d as _test_image_2d
from stardist.data import test_image_nuclei_3d as _test_image_3d

from stardist.bioimageio_utils import export_bioimageio, import_bioimageio, _import
missing = _import(error=False) is None
if not missing:
    from bioimageio.core.resource_tests import test_model as _test_model


def _test_pretrained(tmp_path, model_name, test_image, model_type=StarDist2D, test_image_norm_axes='ZYX'):
    model = model_type.from_pretrained(model_name)
    assert model is not None
    # export model
    export_path = tmp_path / f"{model_name}.zip"
    export_bioimageio(model, export_path, test_input=test_image, test_input_norm_axes=test_image_norm_axes)
    assert export_path.exists()
    # test exported model
    res = _test_model(export_path)
    assert not res["error"]
    # import exported model
    import_path = tmp_path / f"{model_name}_imported"
    model_imported = import_bioimageio(export_path, import_path)
    # test that model and imported exported model are equal
    def _n(d):
        # normalize dict (especially tuples -> lists)
        return json.loads(json.dumps(d))
    assert _n(vars(model.config)) == _n(vars(model_imported.config))
    assert _n(model.thresholds._asdict()) == _n(model_imported.thresholds._asdict())
    assert all(np.allclose(u,v) for u,v in zip(model.keras_model.get_weights(),model_imported.keras_model.get_weights()))


@pytest.mark.skipif(missing, reason="Requires bioimageio dependencies")
def test_pretrained_fluo(tmp_path):
    test_image = _test_image_2d()
    model_name = "2D_versatile_fluo"
    _test_pretrained(tmp_path, model_name, test_image)


@pytest.mark.skipif(missing, reason="Requires bioimageio dependencies")
@pytest.mark.parametrize('test_image_norm_axes', ['YX', 'YXC'])
def test_pretrained_he(tmp_path, test_image_norm_axes):
    test_image = _test_image_2d()
    test_image = np.stack([test_image, test_image+5, test_image+10], axis=-1)
    model_name = "2D_versatile_he"
    _test_pretrained(tmp_path, model_name, test_image, test_image_norm_axes=test_image_norm_axes)


@pytest.mark.skipif(missing, reason="Requires bioimageio dependencies")
def test_pretrained_3d(tmp_path):
    test_image = _test_image_3d()
    model_name = "3D_demo"
    _test_pretrained(tmp_path, model_name, test_image, model_type=StarDist3D)
