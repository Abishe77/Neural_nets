#  Neural Network from Scratch — Python

A fully hand-built neural network trained on the MNIST dataset using only NumPy. No TensorFlow. No PyTorch. No shortcuts.

Every component — forward propagation, activation functions, softmax, cross-entropy loss, backpropagation, and gradient descent — implemented from mathematical first principles.

**Achieved 89% accuracy on the MNIST test set.**

---

## Architecture

```
Input Layer     →  784 neurons  (28x28 pixel image flattened)
Hidden Layer 1  →  10 neurons   (ReLU activation)
Hidden Layer 2  →  10 neurons   (ReLU activation)
Output Layer    →  10 neurons   (Softmax activation → digit 0–9)
```

---

## Features

- **Forward Propagation** — Matrix multiplication `z = WX + b` across two hidden layers
- **ReLU Activation** — Introduces non-linearity; handles curves the linear model can't
- **Softmax Output** — Converts raw scores to probabilities summing to 1
- **Cross-Entropy Loss** — Measures prediction error against one-hot encoded labels
- **Backpropagation** — Computes gradients layer by layer using the chain rule
- **Gradient Descent** — Updates weights and biases to minimize loss
- **Loss Visualization** — Matplotlib plot of training loss over iterations

---

## How It Works

### Data Preprocessing
MNIST images are 28×28 pixels. Each image is flattened to a 784-dimensional vector and normalized from `[0, 255]` to `[0, 1]` for numerical stability.

### Forward Pass
```
z1 = W1 @ X + b1        →  (10 x 60000)
a1 = ReLU(z1)
z2 = W2 @ a1 + b2       →  (10 x 60000)
predictions = softmax(z2)
```

### Loss Function
Cross-entropy loss measures how far the predicted probability distribution is from the true label:
```
L = -sum(y_true * log(predictions))
```

### Backpropagation
Gradients are computed in reverse using the chain rule:
```
dz2 = predictions - y_true
dW2 = (dz2 @ a1.T) / m
dz1 = (W2.T @ dz2) * ReLU'(z1)
dW1 = (dz1 @ X.T) / m
```

### Weight Update
```
W = W - learning_rate * dW
b = b - learning_rate * db
```

---

## Results

| Dataset   | Accuracy |
|-----------|----------|
| Test Set  | **89%**  |

Trained for 500 epochs with learning rate `0.1`.

---

## Dependencies

```
numpy
matplotlib
tensorflow  ← dataset only (mnist.load_data())
```

---

## Built With

- Jupyter Notebook
- NumPy (all math)
- Matplotlib (visualization)
- A strong understanding of linear algebra and calculus
