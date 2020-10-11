import numpy as np

from Base import Operation, WeightMultiply, BiasAdd, Layer
from Module import Sigmoid


class Dense(Layer):
    """
    Fully connected layer
    """
    def __init__(self,
                 neurons: int,
                 activation: Operation = Sigmoid()) -> None:
        """
        Decide activation function when initialize class
        :param neurons: The number of output nodes
        :param activation: Activation function of network
        """
        super().__init__(neurons)
        self.activation = activation

    def _setup_layer(self,
                     input_: np.ndarray) -> None:
        """
        Define operation of fully connected layer
        :param input_: Input nodes
        :return:
        """
        if self.seed:
            np.random.seed(self.seed)

        self.params = []

        # Weights
        self.params.append(np.random.randn(input_.shape[1], self.neurons))

        # Bias
        self.params.append(np.random.randn(1, self.neurons))

        self.operations = [WeightMultiply(self.params[0]),
                           BiasAdd(self.params[1]),
                           self.activation]

        return None