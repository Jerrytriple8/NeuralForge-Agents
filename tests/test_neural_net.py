"""
Tests for NeuralNetwork.
"""

import pytest
import numpy as np
from core.neural_net import (
    NeuralNetwork, Layer, LayerConfig, ReLU, Sigmoid, Tanh, Softmax,
    MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss,
)
from core.optimizer import Adam, SGD, AdaGrad, RMSProp


class TestActivations:
    def test_relu(self):
        relu = ReLU()
        x = np.array([-1, 0, 1, 2])
        result = relu.forward(x)
        np.testing.assert_array_equal(result, [0, 0, 1, 2])

    def test_relu_backward(self):
        relu = ReLU()
        x = np.array([-1, 0, 1, 2])
        grad = relu.backward(x)
        np.testing.assert_array_equal(grad, [0, 0, 1, 1])

    def test_sigmoid(self):
        sig = Sigmoid()
        x = np.array([0, 100, -100])
        result = sig.forward(x)
        assert abs(result[0] - 0.5) < 0.01
        assert result[1] > 0.99
        assert result[2] < 0.01

    def test_tanh(self):
        tanh = Tanh()
        x = np.array([0, 1, -1])
        result = tanh.forward(x)
        assert abs(result[0]) < 0.01
        assert abs(result[1] - np.tanh(1)) < 0.01

    def test_softmax(self):
        softmax = Softmax()
        x = np.array([[1, 2, 3]])
        result = softmax.forward(x)
        assert abs(result.sum() - 1.0) < 0.01


class TestLosses:
    def test_mse(self):
        mse = MSELoss()
        pred = np.array([[1, 2, 3]])
        target = np.array([[1, 2, 3]])
        loss = mse.forward(pred, target)
        assert loss == 0.0

    def test_mse_nonzero(self):
        mse = MSELoss()
        pred = np.array([[1, 2, 3]])
        target = np.array([[0, 0, 0]])
        loss = mse.forward(pred, target)
        assert loss > 0

    def test_cross_entropy(self):
        ce = CrossEntropyLoss()
        pred = np.array([[0.9, 0.1]])
        target = np.array([[1, 0]])
        loss = ce.forward(pred, target)
        assert loss < 0.2  # Low loss for correct prediction

    def test_binary_cross_entropy(self):
        bce = BinaryCrossEntropyLoss()
        pred = np.array([[0.9]])
        target = np.array([[1]])
        loss = bce.forward(pred, target)
        assert loss < 0.2


class TestLayer:
    def test_layer_creation(self):
        config = LayerConfig(input_size=10, output_size=5)
        layer = Layer(config)
        assert layer.weights.shape == (10, 5)
        assert layer.bias.shape == (1, 5)

    def test_layer_forward(self):
        config = LayerConfig(input_size=3, output_size=2)
        layer = Layer(config)
        x = np.random.randn(1, 3)
        output = layer.forward(x)
        assert output.shape == (1, 2)

    def test_layer_backward(self):
        config = LayerConfig(input_size=3, output_size=2)
        layer = Layer(config)
        x = np.random.randn(1, 3)
        output = layer.forward(x)
        grad = layer.backward(np.ones_like(output))
        assert grad.shape == (1, 3)
        assert layer.dw is not None
        assert layer.db is not None


class TestNeuralNetwork:
    def test_build_network(self):
        nn = NeuralNetwork()
        nn.add_layer(LayerConfig(input_size=2, output_size=4))
        nn.add_layer(LayerConfig(input_size=4, output_size=1))
        assert len(nn.layers) == 2

    def test_build_with_mismatched_sizes(self):
        nn = NeuralNetwork()
        nn.add_layer(LayerConfig(input_size=2, output_size=4))
        with pytest.raises(ValueError, match="input size"):
            nn.add_layer(LayerConfig(input_size=3, output_size=1))

    def test_forward(self):
        nn = NeuralNetwork()
        nn.add_layer(LayerConfig(input_size=2, output_size=4))
        nn.add_layer(LayerConfig(input_size=4, output_size=1, activation="sigmoid"))
        x = np.array([[0, 1]])
        output = nn.forward(x)
        assert output.shape == (1, 1)
        assert 0 <= output[0, 0] <= 1

    def test_train_xor(self):
        nn = NeuralNetwork()
        nn.add_layer(LayerConfig(input_size=2, output_size=8, activation="relu"))
        nn.add_layer(LayerConfig(input_size=8, output_size=4, activation="relu"))
        nn.add_layer(LayerConfig(input_size=4, output_size=1, activation="sigmoid"))
        nn.set_loss("mse")
        nn.set_optimizer(Adam(lr=0.01))

        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y = np.array([[0], [1], [1], [0]], dtype=np.float32)

        history = nn.fit(X, y, epochs=200, batch_size=4, lr=0.01, verbose=False)
        assert history[-1]["loss"] < history[0]["loss"]

    def test_predict(self):
        nn = NeuralNetwork()
        nn.add_layer(LayerConfig(input_size=2, output_size=1, activation="sigmoid"))
        nn.set_loss("mse")
        x = np.array([[1, 2], [3, 4]])
        predictions = nn.predict(x)
        assert predictions.shape == (2, 1)

    def test_summary(self):
        nn = NeuralNetwork()
        nn.add_layer(LayerConfig(input_size=2, output_size=4))
        nn.add_layer(LayerConfig(input_size=4, output_size=1))
        summary = nn.summary()
        assert "NeuralNetwork" in summary
        assert "Layer 0" in summary

    def test_save_load(self, tmp_path):
        nn = NeuralNetwork()
        nn.add_layer(LayerConfig(input_size=2, output_size=4))
        nn.add_layer(LayerConfig(input_size=4, output_size=1))
        nn.set_loss("mse")

        path = str(tmp_path / "model.npy")
        nn.save(path)
        nn.load(path)


class TestOptimizers:
    def test_adam(self):
        nn = NeuralNetwork()
        nn.add_layer(LayerConfig(input_size=2, output_size=4))
        nn.add_layer(LayerConfig(input_size=4, output_size=1))
        nn.set_loss("mse")
        nn.set_optimizer(Adam(lr=0.001))

        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        history = nn.fit(X, y, epochs=50, verbose=False)
        assert len(history) == 50

    def test_sgd(self):
        nn = NeuralNetwork()
        nn.add_layer(LayerConfig(input_size=2, output_size=4))
        nn.add_layer(LayerConfig(input_size=4, output_size=1))
        nn.set_loss("mse")
        nn.set_optimizer(SGD(lr=0.01, momentum=0.9))

        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        history = nn.fit(X, y, epochs=50, verbose=False)
        assert len(history) == 50

    def test_adagrad(self):
        nn = NeuralNetwork()
        nn.add_layer(LayerConfig(input_size=2, output_size=4))
        nn.add_layer(LayerConfig(input_size=4, output_size=1))
        nn.set_loss("mse")
        nn.set_optimizer(AdaGrad(lr=0.01))

        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        history = nn.fit(X, y, epochs=50, verbose=False)
        assert len(history) == 50

    def test_rmsprop(self):
        nn = NeuralNetwork()
        nn.add_layer(LayerConfig(input_size=2, output_size=4))
        nn.add_layer(LayerConfig(input_size=4, output_size=1))
        nn.set_loss("mse")
        nn.set_optimizer(RMSProp(lr=0.001))

        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        history = nn.fit(X, y, epochs=50, verbose=False)
        assert len(history) == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
