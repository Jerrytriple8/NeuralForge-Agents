"""
Neural Network implementation from scratch.
Supports dense layers, activation functions, backpropagation, and mini-batch training.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# --- Activation Functions ---

class Activation:
    """Base activation function."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ReLU(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)


class Sigmoid(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def backward(self, x: np.ndarray) -> np.ndarray:
        s = self.forward(x)
        return s * (1 - s)


class Tanh(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2


class LeakyReLU(Activation):
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1.0, self.alpha)


class Softmax(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    def backward(self, x: np.ndarray) -> np.ndarray:
        s = self.forward(x)
        return s * (1 - s)


class Swish(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x * (1 / (1 + np.exp(-x)))

    def backward(self, x: np.ndarray) -> np.ndarray:
        s = 1 / (1 + np.exp(-x))
        return s + x * s * (1 - s)


ACTIVATIONS = {
    "relu": ReLU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "leaky_relu": LeakyReLU,
    "softmax": Softmax,
    "swish": Swish,
}


# --- Loss Functions ---

class Loss:
    def forward(self, predicted: np.ndarray, target: np.ndarray) -> float:
        raise NotImplementedError

    def backward(self, predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MSELoss(Loss):
    def forward(self, predicted: np.ndarray, target: np.ndarray) -> float:
        return float(np.mean((predicted - target) ** 2))

    def backward(self, predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        return 2 * (predicted - target) / target.size


class CrossEntropyLoss(Loss):
    def forward(self, predicted: np.ndarray, target: np.ndarray) -> float:
        predicted = np.clip(predicted, 1e-15, 1 - 1e-15)
        return float(-np.mean(target * np.log(predicted)))

    def backward(self, predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        predicted = np.clip(predicted, 1e-15, 1 - 1e-15)
        return (predicted - target) / (predicted * (1 - predicted))


class BinaryCrossEntropyLoss(Loss):
    def forward(self, predicted: np.ndarray, target: np.ndarray) -> float:
        predicted = np.clip(predicted, 1e-15, 1 - 1e-15)
        return float(-np.mean(
            target * np.log(predicted) + (1 - target) * np.log(1 - predicted)
        ))

    def backward(self, predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        predicted = np.clip(predicted, 1e-15, 1 - 1e-15)
        return (predicted - target) / (predicted * (1 - predicted))


LOSSES = {
    "mse": MSELoss,
    "cross_entropy": CrossEntropyLoss,
    "binary_cross_entropy": BinaryCrossEntropyLoss,
}


# --- Layer ---

@dataclass
class LayerConfig:
    input_size: int
    output_size: int
    activation: str = "relu"
    dropout: float = 0.0
    batch_norm: bool = False


class Layer:
    """Single neural network layer with weights, bias, and activation."""

    def __init__(self, config: LayerConfig):
        self.config = config
        # He initialization
        self.weights = np.random.randn(
            config.input_size, config.output_size
        ) * np.sqrt(2.0 / config.input_size)
        self.bias = np.zeros((1, config.output_size))
        self.activation = ACTIVATIONS[config.activation]()
        self.dropout = config.dropout

        # Batch normalization params
        self.batch_norm = config.batch_norm
        if self.batch_norm:
            self.gamma = np.ones((1, config.output_size))
            self.beta = np.zeros((1, config.output_size))
            self.running_mean = np.zeros((1, config.output_size))
            self.running_var = np.ones((1, config.output_size))
            self.bn_epsilon = 1e-8
            self.bn_momentum = 0.1

        # Gradient storage
        self.dw = None
        self.db = None
        self.dgamma = None
        self.dbeta = None

        # Cache for backprop
        self._input = None
        self._pre_activation = None
        self._output = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self._input = x
        z = x @ self.weights + self.bias

        if self.batch_norm:
            if training:
                mean = np.mean(z, axis=0, keepdims=True)
                var = np.var(z, axis=0, keepdims=True)
                self.running_mean = (
                    self.bn_momentum * mean + (1 - self.bn_momentum) * self.running_mean
                )
                self.running_var = (
                    self.bn_momentum * var + (1 - self.bn_momentum) * self.running_var
                )
            else:
                mean = self.running_mean
                var = self.running_var
            z_norm = (z - mean) / np.sqrt(var + self.bn_epsilon)
            z = self.gamma * z_norm + self.beta

        self._pre_activation = z
        a = self.activation.forward(z)

        if training and self.dropout > 0:
            mask = (np.random.rand(*a.shape) > self.dropout).astype(float)
            a = a * mask / (1 - self.dropout)

        self._output = a
        return a

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self.dropout > 0 and self._output is not None:
            mask = (self._output != 0).astype(float)
            grad = grad * mask / (1 - self.dropout)

        act_grad = grad * self.activation.backward(self._pre_activation)

        if self.batch_norm:
            self.dgamma = np.sum(act_grad * (self._pre_activation - self.running_mean) / 
                                np.sqrt(self.running_var + self.bn_epsilon), axis=0, keepdims=True)
            self.dbeta = np.sum(act_grad, axis=0, keepdims=True)

        self.dw = self._input.T @ act_grad
        self.db = np.sum(act_grad, axis=0, keepdims=True)
        return act_grad @ self.weights.T

    @property
    def params(self) -> dict:
        return {"weights": self.weights, "bias": self.bias}


# --- Neural Network ---

class NeuralNetwork:
    """
    Multi-layer neural network with configurable architecture.
    
    Supports:
    - Dense layers with various activations
    - Dropout regularization
    - Batch normalization
    - Mini-batch gradient descent
    - Multiple optimizers (SGD, Adam, AdaGrad)
    - Training history tracking
    """

    def __init__(self):
        self.layers: List[Layer] = []
        self.loss_fn: Optional[Loss] = None
        self.optimizer = None
        self.history: List[dict] = []

    def add_layer(self, config: LayerConfig) -> "NeuralNetwork":
        """Add a layer to the network."""
        if self.layers:
            prev = self.layers[-1].config.output_size
            if config.input_size != prev:
                raise ValueError(
                    f"Layer input size {config.input_size} != prev output {prev}"
                )
        self.layers.append(Layer(config))
        return self

    def set_loss(self, loss_name: str) -> "NeuralNetwork":
        if loss_name not in LOSSES:
            raise ValueError(f"Unknown loss: {loss_name}")
        self.loss_fn = LOSSES[loss_name]()
        return self

    def set_optimizer(self, optimizer: Any) -> "NeuralNetwork":
        self.optimizer = optimizer
        return self

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def backward(self, grad: np.ndarray) -> None:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, lr: float) -> None:
        if self.optimizer:
            self.optimizer.update(self.layers)
        else:
            for layer in self.layers:
                layer.weights -= lr * layer.dw
                layer.bias -= lr * layer.db

    def train_step(
        self, x: np.ndarray, y: np.ndarray, lr: float
    ) -> float:
        """Single training step."""
        pred = self.forward(x, training=True)
        loss = self.loss_fn.forward(pred, y)
        grad = self.loss_fn.backward(pred, y)
        self.backward(grad)
        self.update(lr)
        return loss

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 0.001,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: bool = True,
    ) -> List[dict]:
        """Train the network."""
        n_samples = X.shape[0]
        history = []

        for epoch in range(epochs):
            # Shuffle
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]
                loss = self.train_step(X_batch, y_batch, lr)
                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            record = {"epoch": epoch, "loss": avg_loss}

            if validation_data:
                val_pred = self.forward(validation_data[0], training=False)
                val_loss = self.loss_fn.forward(val_pred, validation_data[1])
                record["val_loss"] = val_loss

            history.append(record)
            self.history.append(record)

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                msg = f"Epoch {epoch:4d} | Loss: {avg_loss:.6f}"
                if "val_loss" in record:
                    msg += f" | Val Loss: {record['val_loss']:.6f}"
                logger.info(msg)

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X, training=False)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        pred = self.forward(X, training=False)
        loss = self.loss_fn.forward(pred, y)
        return {"loss": loss}

    def summary(self) -> str:
        lines = ["=" * 60, "NeuralNetwork Summary", "=" * 60]
        total_params = 0
        for i, layer in enumerate(self.layers):
            n_params = layer.weights.size + layer.bias.size
            total_params += n_params
            lines.append(
                f"Layer {i}: ({layer.config.input_size}, {layer.config.output_size}) "
                f"| {layer.config.activation} | Params: {n_params}"
            )
        lines.append("=" * 60)
        lines.append(f"Total parameters: {total_params:,}")
        return "\n".join(lines)

    def save(self, path: str) -> None:
        data = []
        for layer in self.layers:
            data.append({
                "weights": layer.weights,
                "bias": layer.bias,
                "config": layer.config.__dict__,
            })
        np.save(path, np.array(data, dtype=object), allow_pickle=True)
        logger.info("Model saved to %s", path)

    def load(self, path: str) -> None:
        data = np.load(path, allow_pickle=True)
        for i, layer_data in enumerate(data):
            if i < len(self.layers):
                self.layers[i].weights = layer_data["weights"]
                self.layers[i].bias = layer_data["bias"]
        logger.info("Model loaded from %s", path)
