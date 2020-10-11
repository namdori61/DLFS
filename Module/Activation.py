import numpy as np

from Base import Operation


class Sigmoid(Operation):
    """
    Sigmoid activation function
    """

    def __init__(self) -> None:
        """
        Pass
        """
        super().__init__()

    def _output(self) -> np.ndarray:
        """
        Operation for output
        :return: Forward operation of sigmoid function
        """
        return 1.0 / (1.0 + np.exp(-1.0 * self.input_))

    def _input_grad(self,
                    output_grad: np.ndarray) -> np.ndarray:
        """
        Operation for gradients of input
        :param output_grad: gradients of output
        :return: gradients of input
        """
        sigmoid_backward = self.output * (1.0 - self.output)
        input_grad = sigmoid_backward * output_grad
        return input_grad
