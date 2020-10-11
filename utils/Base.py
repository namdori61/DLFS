import numpy as np


def assert_same_shape(input1: np.ndarray,
                      input2: np.ndarray):
    '''
    Check shape of two inputs.
    :param `input1`: First input vector or matrix
    :param `input2`: Second input vector or matrix
    :return:
    '''
    assert input1.shape == input2.shape