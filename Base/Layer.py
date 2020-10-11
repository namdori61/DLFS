from typing import List

import numpy as np

from Base import Operation, ParamOperation
from utils import assert_same_shape


class Layer(object):
    """
    Layer class for neural network
    """

    def __init__(self,
                 neurons: int = None):
        """
        Initializer `neurons` in layer
        :param neurons: The number of output nodes
        """
        self.neurons = neurons
        self.first = True
        self.params: List[np.ndarray] = []
        self.param_grads: List[np.ndarray] = []
        self.operations: List[Operation] = []

    def _setup_layer(self,
                     num_in: int = None) -> None:
        """
        Define succession and save params of Operation class object
        :param num_in: The number of input nodes
        :return:
        """
        raise NotImplementedError()

    def forward(self,
                input_: np.ndarray = None) -> np.ndarray:
        """
        Forward propagation
        :param input_: Input to process
        :return: Output of forward propagation
        """
        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input = input_

        for operation in self.operations:
            input_ = operation.forward(input_)

        self.output = input_

        return self.output

    def backward(self,
                 output_grad: np.ndarray) -> np.ndarray:
        """
        Backward propagation
        :param output_grad: Gradient of output
        :return: Output of backward propagation
        """
        assert_same_shape(self.output, output_grad)

        for operation in self.operations:
            output_grad = operation.backward(output_grad)

        input_grad = output_grad

        self._param_grads()

        return input_grad

    def _param_grads(self) -> np.ndarray:
        """
        Get _param_grads from Operation objects
        :return: Gradient of params
        """
        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self) -> np.ndarray:
        """
        Get _params from Operation objects
        :return: Params
        """
        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)