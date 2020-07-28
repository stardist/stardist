import pytest


def pytest_addoption(parser):
    parser.addoption("--use_gpu", action="store_true")


@pytest.fixture
def use_gpu(request):
    return request.config.getoption("--use_gpu")

def _model2d():
    from utils import path_model2d
    from stardist.models import StarDist2D
    model_path = path_model2d()
    return StarDist2D(None, name=model_path.name, basedir=str(model_path.parent))

@pytest.fixture(scope='session')
def model2d():
    return _model2d()


def _model3d():
    from utils import path_model3d
    from stardist.models import StarDist3D
    model_path = path_model3d()
    return StarDist3D(None, name=model_path.name, basedir=str(model_path.parent))


@pytest.fixture(scope='session')
def model3d():
    return _model3d()
