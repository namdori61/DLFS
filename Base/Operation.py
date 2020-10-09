import numpy as np

from utils import assert_same_shape


class Operation(object):
    """
    Base class for operation of neural network
    """

    def __init__(self):
        pass

    def forward(self,
                input_: np.ndarray):
        """
        Save param `input_` to instance variable `self.input`
        and return result of callable `self._output()`
        :param `input_`: input vector or matrix
        :return: Return of callable `self._output()`
        """
        self.input_ = input_

        self.output = self._output()

        return self.output

    def backward(self,
                 output_grad: np.ndarray) -> np.ndarray:
        """
        Call `self.input_grad()`. First, check shape.
        :param `output_grad`: gradients of output
        :return: `input_grad`: gradients of input
        """
        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)
        return self.input_grad

    def _output(self) -> np.ndarray:
        """
        Child class of Operation class must implement `_output` method.
        :return:
        """
        raise NotImplementedError()

    def _input_grad(self,
                    output_grad: np.ndarray) -> np.ndarray:
        """
        Child class of Operation class must implement `_input_grad` method.
        :param `output_grad`: gradients of output
        :return:
        """
        raise NotImplementedError()


class ParamOperation(Operation):
    """
    Operation class with param
    """

    def __init__(self,
                 param: np.ndarray):
        """
        Initiation method
        :param param: Weight matrix or Bias
        """
        super().__init__()
        self.param = param

    def backward(self,
                 output_grad: np.ndarray) -> np.ndarray:
        """
        Call `self._input_grad` and `self._param_grad`.
        Check shape of objects.
        :param output_grad: gradients of output
        :return: input_grad: gradients of input
        """
        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)
        assert_same_shape(self.param, self.param_grad)

        return self.input_grad

    def _param_grad(self,
                    output_grad: np.ndarray)  -> np.ndarray:
        """
        Child class of ParamOperation class must implement `_param_grad` method.
        :param output_grad: gradients of output
        :return:
        """
        raise NotImplementedError()


class WeightMultiply(ParamOperation):
    """
    Weight matrix multiplication of neural network
    """

    def __init__(self,
                 W: np.ndarray):
        """
        Initiate `self.param` with `W`.
        :param W: weight matrix
        """
        super().__init__(W)

    def _output(self) -> np.ndarray:
        """
        Operation for output
        :return: Forward operation of weight matrix
        """
        return np.dot(self.input_, self.param)

    def _input_grad(self,
                    output_grad: np.ndarray) -> np.ndarray:
        """
        Operation for gradients of input
        :param output_grad: gradients of output
        :return: gradients of input
        """
        return np.dot(output_grad, np.transpose(self.param, (1, 0)))

    def _param_grad(self,
                    output_grad: np.ndarray) -> np.ndarray:
        """
        Operation for gradients of param
        :param output_grad: gradients of output
        :return: gradients of param
        """
        return np.dot(np.transpose(self.input_, (1, 0)), output_grad)


class BiasAdd(ParamOperation):
    """
    Bias addition of neural network
    """

    def __init__(self,
                 B: np.ndarray):
        """
        Initiate `self.param` with `B`.
        Check shape of B.
        :param B: bias matrix
        """
        assert B.shape[0] == 1

        super().__init__(B)

    def _output(self) -> np.ndarray:
        """
        Operation for output
        :return: Forward operation of bias matrix
        """
        return self.input_ + self.param

    def _input_grad(self,
                    output_grad: np.ndarray) -> np.ndarray:
        """
        Operation for gradients of input
        :param output_grad: gradients of output
        :return: gradients of input
        """
        return np.ones_like(self.input_) * output_grad

    def _param_grad(self,
                    output_grad: np.ndarray) -> np.ndarray:
        """
        Operation for gradients of param
        :param output_grad: gradients of output
        :return: gradients of param
        """
        param_grad = np.ones_like(self.param) * output_grad
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])