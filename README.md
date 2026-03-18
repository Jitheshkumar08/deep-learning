# Deep Learning Backpropagation (NumPy)

A compact educational repository that demonstrates **manual forward pass + backpropagation** in three neural network styles using only NumPy:

- `ANN_Backpropagation.py` — single-neuron feedforward network with sigmoid + MSE
- `CNN_Backpropagation.py` — manual 2D convolution and filter gradient update
- `RNN_Backpropagation.py` — simple recurrent sequence model with a tanh state update

This project is designed for learning and classroom demos, not production training.

## Repository Structure

- `ANN_Backpropagation.py`
- `CNN_Backpropagation.py`
- `RNN_Backpropagation.py`

## What Each Script Demonstrates

### 1) ANN Backpropagation (`ANN_Backpropagation.py`)

Implements a one-input, one-output neuron:

- Forward pass:  
  `z = X·W + b`  
  `y_pred = sigmoid(z)`
- Loss: Mean Squared Error (MSE) style term  
  `L = 0.5 * (y - y_pred)^2`
- Backprop:
  - `error = y_pred - y`
  - `d_pred = error * sigmoid_derivative(y_pred)`
  - `dW = X^T · d_pred`
  - `db = d_pred`
- Parameter update:
  - `W -= lr * dW`
  - `b -= lr * db`

Runs for 5 epochs and prints prediction, loss, and updated weight each epoch.

---

### 2) CNN Backpropagation (`CNN_Backpropagation.py`)

Demonstrates manual convolution on a small 3x3 image using a 2x2 filter:

- Builds a 2x2 feature map by sliding the filter over image patches
- Uses a fixed upstream gradient map (`0.02` everywhere)
- Computes filter gradient by accumulating:
  - `grad_filter += image_patch * grad_map[r, c]`
- Updates filter with gradient descent:
  - `flt -= lr * grad_filter`

Prints generated feature map, filter gradient, and updated filter values.

---

### 3) RNN Backpropagation (`RNN_Backpropagation.py`)

Shows simplified sequence learning with recurrent hidden state:

- Hidden update over sequence `[2, 1, 3]`:
  - `h_t = tanh(w_in * x_t + w_rec * h_{t-1})`
- Prediction uses final hidden state
- Loss:
  - `L = 0.5 * (target - pred)^2`
- Simplified BPTT-style gradients (single-step approximation at final state):
  - `delta = (pred - target) * (1 - pred^2)`
  - `grad_in = delta * x_last`
  - `grad_rec = delta * h_prev`
- Update:
  - `w_in -= lr * grad_in`
  - `w_rec -= lr * grad_rec`

Prints sequence states, loss, and updated recurrent parameters.

## Requirements

- Python 3.8+
- NumPy

Install dependency:

```bash
pip install numpy
```

## How to Run

From the repository root:

```bash
python ANN_Backpropagation.py
python CNN_Backpropagation.py
python RNN_Backpropagation.py
```

## Learning Notes

- This repo intentionally uses **small numeric examples** for clarity.
- Gradients are written manually to help understand backprop mechanics.
- The RNN example uses a simplified final-step gradient, which is useful pedagogically but not a full production BPTT implementation.

## Suggested Next Improvements

- Add reusable functions/classes for each model
- Add command-line arguments for learning rate, epochs, and inputs
- Add plotting for loss curves
- Add unit tests for forward and gradient calculations

## License

Use for educational purposes. Add a formal license file if you plan to distribute or publish this project.
