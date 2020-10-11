from Base import Loss

import numpy as np


class MeanSquaredError(Loss):
    """
    MSE loss class
    """
    def __init__(self):
        super().__init__()

    def _output(self) -> float:
        """
        Mean squared error loss function of input
        :return: MSE error value
        """
        loss = np.sum(np.power(self.prediction - self.target, 2)) / self.prediction.shape[0]

        return loss

    def _input_grad(self) -> np.ndarray:
        """
        Gradients of predicted value with MSE loss
        :return: Gradients of predicted value
        """
        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]