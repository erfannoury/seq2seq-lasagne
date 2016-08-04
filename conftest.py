from mock import Mock
import pytest

tests_require = [
    'mock',
    'pytest',
    'pytest-cov',
    'pytest-pep8',
    ]

@pytest.fixture
def dummy_input_layer():
    from lasagne.layers.input import InputLayer
    input_layer = InputLayer((2, 3, 4))
    mock = Mock(input_layer)
    mock.shape = input_layer.shape
    mock.input_var = input_layer.input_var
    mock.output_shape = input_layer.output_shape
    return mock
