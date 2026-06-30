"""
Optimization algorithms for neural network training.
Implements SGD, Adam, and AdaGrad with momentum support.
"""

import numpy as np
from typing import List, Any


class Optimizer:
    """Base optimizer class."""

    def __init__(self, lr: float = 0.001):
        self.lr = lr

    def update(self, layers: List[Any]) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent with momentum."""

    def __init__(self, lr: float = 0.01, momentum: float = 0.0, nesterov: bool = False):
        super().__init__(lr)
        self.momentum = momentum
        self.nesterov = nesterov
        self._velocity: dict = {}

    def update(self, layers: List[Any]) -> None:
        for i, layer in enumerate(layers):
            if layer.dw is None:
                continue

            key = id(layer)
            if key not in self._velocity:
                self._velocity[key] = {
                    "vw": np.zeros_like(layer.weights),
                    "vb": np.zeros_like(layer.bias),
                }

            v = self._velocity[key]

            if self.nesterov:
                # Nesterov momentum
                v["vw"] = self.momentum * v["vw"] - self.lr * layer.dw
                v["vb"] = self.momentum * v["vb"] - self.lr * layer.db
                layer.weights += self.momentum * v["vw"] - self.lr * layer.dw
                layer.bias += self.momentum * v["vb"] - self.lr * layer.db
            else:
                v["vw"] = self.momentum * v["vw"] - self.lr * layer.dw
                v["vb"] = self.momentum * v["vb"] - self.lr * layer.db
                layer.weights += v["vw"]
                layer.bias += v["vb"]


class Adam(Optimizer):
    """Adam optimizer with bias correction."""

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._state: dict = {}
        self._t = 0

    def update(self, layers: List[Any]) -> None:
        self._t += 1

        for layer in layers:
            if layer.dw is None:
                continue

            key = id(layer)
            if key not in self._state:
                self._state[key] = {
                    "mw": np.zeros_like(layer.weights),
                    "vw": np.zeros_like(layer.weights),
                    "mb": np.zeros_like(layer.bias),
                    "vb": np.zeros_like(layer.bias),
                }

            s = self._state[key]

            # Update biased first moment
            s["mw"] = self.beta1 * s["mw"] + (1 - self.beta1) * layer.dw
            s["mb"] = self.beta1 * s["mb"] + (1 - self.beta1) * layer.db

            # Update biased second raw moment
            s["vw"] = self.beta2 * s["vw"] + (1 - self.beta2) * (layer.dw ** 2)
            s["vb"] = self.beta2 * s["vb"] + (1 - self.beta2) * (layer.db ** 2)

            # Bias correction
            mw_corrected = s["mw"] / (1 - self.beta1 ** self._t)
            mb_corrected = s["mb"] / (1 - self.beta1 ** self._t)
            vw_corrected = s["vw"] / (1 - self.beta2 ** self._t)
            vb_corrected = s["vb"] / (1 - self.beta2 ** self._t)

            # Update weights
            layer.weights -= self.lr * mw_corrected / (np.sqrt(vw_corrected) + self.epsilon)
            layer.bias -= self.lr * mb_corrected / (np.sqrt(vb_corrected) + self.epsilon)


class AdaGrad(Optimizer):
    """Adaptive gradient algorithm."""

    def __init__(self, lr: float = 0.01, epsilon: float = 1e-8):
        super().__init__(lr)
        self.epsilon = epsilon
        self._cache: dict = {}

    def update(self, layers: List[Any]) -> None:
        for layer in layers:
            if layer.dw is None:
                continue

            key = id(layer)
            if key not in self._cache:
                self._cache[key] = {
                    "gw": np.zeros_like(layer.weights),
                    "gb": np.zeros_like(layer.bias),
                }

            c = self._cache[key]
            c["gw"] += layer.dw ** 2
            c["gb"] += layer.db ** 2

            layer.weights -= self.lr * layer.dw / (np.sqrt(c["gw"]) + self.epsilon)
            layer.bias -= self.lr * layer.db / (np.sqrt(c["gb"]) + self.epsilon)


class RMSProp(Optimizer):
    """RMSProp optimizer."""

    def __init__(self, lr: float = 0.001, decay: float = 0.9, epsilon: float = 1e-8):
        super().__init__(lr)
        self.decay = decay
        self.epsilon = epsilon
        self._cache: dict = {}

    def update(self, layers: List[Any]) -> None:
        for layer in layers:
            if layer.dw is None:
                continue

            key = id(layer)
            if key not in self._cache:
                self._cache[key] = {
                    "gw": np.zeros_like(layer.weights),
                    "gb": np.zeros_like(layer.bias),
                }

            c = self._cache[key]
            c["gw"] = self.decay * c["gw"] + (1 - self.decay) * (layer.dw ** 2)
            c["gb"] = self.decay * c["gb"] + (1 - self.decay) * (layer.db ** 2)

            layer.weights -= self.lr * layer.dw / (np.sqrt(c["gw"]) + self.epsilon)
            layer.bias -= self.lr * layer.db / (np.sqrt(c["gb"]) + self.epsilon)
