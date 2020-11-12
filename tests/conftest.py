import pytest


def _limit_tf_gpu_memory():
    from csbdeep.utils.tf import IS_TF_1, limit_gpu_memory
    limit_gpu_memory(0.75, total_memory=(None if IS_TF_1 else 8000))

def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: run opencl-based tests on the gpu")
    _limit_tf_gpu_memory()


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
