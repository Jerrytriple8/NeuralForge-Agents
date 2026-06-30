"""
Example: Neural network training.

Demonstrates building and training a neural network
for the XOR problem using NeuralForge.
"""

import numpy as np
from core import NeuralNetwork
from core.neural_net import LayerConfig
from core.optimizer import Adam, SGD


def main():
    print("=== NeuralForge Neural Network Example ===\n")

    # XOR dataset
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=np.float32)

    y = np.array([
        [0],
        [1],
        [1],
        [0],
    ], dtype=np.float32)

    print("Training data:")
    for i in range(4):
        print(f"  {X[i]} -> {y[i][0]}")
    print()

    # Build network
    nn = NeuralNetwork()
    nn.add_layer(LayerConfig(input_size=2, output_size=8, activation="relu"))
    nn.add_layer(LayerConfig(input_size=8, output_size=4, activation="relu"))
    nn.add_layer(LayerConfig(input_size=4, output_size=1, activation="sigmoid"))
    nn.set_loss("mse")
    nn.set_optimizer(Adam(lr=0.01))

    print(nn.summary())
    print()

    # Train
    print("Training...")
    history = nn.fit(X, y, epochs=500, batch_size=4, lr=0.01, verbose=False)

    print(f"Final loss: {history[-1]['loss']:.6f}")
    print(f"Initial loss: {history[0]['loss']:.6f}")
    print(f"Improvement: {(1 - history[-1]['loss'] / history[0]['loss']) * 100:.1f}%")
    print()

    # Predictions
    print("Predictions:")
    predictions = nn.predict(X)
    for i in range(4):
        pred = predictions[i][0]
        target = y[i][0]
        status = "OK" if abs(pred - target) < 0.3 else "MISS"
        print(f"  {X[i]} -> {pred:.4f} (target: {target}) [{status}]")

    # Compare optimizers
    print("\n=== Optimizer Comparison ===")
    optimizers = {
        "Adam": Adam(lr=0.01),
        "SGD (momentum)": SGD(lr=0.01, momentum=0.9),
        "SGD (plain)": SGD(lr=0.01),
    }

    for name, opt in optimizers.items():
        nn_opt = NeuralNetwork()
        nn_opt.add_layer(LayerConfig(input_size=2, output_size=8, activation="relu"))
        nn_opt.add_layer(LayerConfig(input_size=8, output_size=4, activation="relu"))
        nn_opt.add_layer(LayerConfig(input_size=4, output_size=1, activation="sigmoid"))
        nn_opt.set_loss("mse")
        nn_opt.set_optimizer(opt)
        history = nn_opt.fit(X, y, epochs=200, batch_size=4, lr=0.01, verbose=False)
        print(f"  {name:20} | Final loss: {history[-1]['loss']:.6f}")


if __name__ == "__main__":
    main()
