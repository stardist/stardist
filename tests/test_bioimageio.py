from bioimageio.core.resource_tests import test_model as _test
from stardist.models import StarDist2D
from stardist.data import test_image_nuclei_2d as _test_image


# TODO test mode=keras_hdf5
def test_pretrained(tmp_path):
    from stardist import export_bioimageio

    test_image = _test_image()
    for t in ('2D_versatile_fluo', '2D_versatile_he', '2D_paper_dsb2018'):
        model = StarDist2D.from_pretrained(t)
        out_path = tmp_path / f"{t}.zip"
        export_bioimageio(model, out_path, test_input=test_image)
        assert out_path.exists()
        res = _test(out_path)
        breakpoint()
        assert not res["error"]
        # breakpoint()
