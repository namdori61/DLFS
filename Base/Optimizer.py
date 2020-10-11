class Optimizer(object):
    """
    Base class for optimizer of neural network
    """
    def __init__(self,
                 lr: float = 0.01):
        self.lr = lr

    def step(self) -> None:
        """
        Child class of Operation class must implement `step` method.
        :return:
        """
        raise NotImplementedError()