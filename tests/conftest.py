import pytest


def pytest_addoption(parser):
    parser.addoption("--use_gpu", action="store_true")


@pytest.fixture
def use_gpu(request):
    return request.config.getoption("--use_gpu")
