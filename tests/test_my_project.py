import numpy as np
from my_project import Function


def test_generate_random_array():
    shape = (3, 3)
    arr = Function.generate_random_array(shape)
    assert arr.shape == shape
    assert (arr >= 0).all() and (arr <= 1).all()


def test_logistic_function():
    x = np.array([0, 1, 2])
    expected_output = np.array([0.5, 0.73105858, 0.88079708])
    np.testing.assert_allclose(Function.logistic_function(x), expected_output)
