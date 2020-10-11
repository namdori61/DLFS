import numpy as np

from utils import assert_same_shape


class Loss(object):
    """
    Loss class to calculate loss of network
    """
    def __init__(self):
        pass

    def forward(self,
                prediction: np.ndarray,
                target: np.ndarray) -> float:
        """
        Calculate loss from prediction and target values
        :param prediction: Predicted values
        :param target: True values
        :return:
        """
        assert_same_shape(prediction, target)

        self.prediction = prediction
        self.target = target

        loss_value = self._output()

        return loss_value

    def backward(self) -> np.ndarray:
        """
        Return gradients of input with loss function
        :return: Gradients of input with loss
        """
        self.input_grad = self._input_grad()

        assert_same_shape(self.prediction, self.input_grad)

        return self.input_grad

    def _output(self) -> float:
        """
        Child class of Loss class must implement `_output` method.
        :return: Loss
        """
        raise NotImplementedError()

    def _input_grad(self) -> np.ndarray:
        """
        Child class of Loss class must implement `_input_grad` method.
        :return: Gradients of input with loss
        """
        raise NotImplementedError()