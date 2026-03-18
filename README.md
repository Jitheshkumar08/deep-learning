# Mini Seminar - Deep Learning

## Backpropagation in ANN, CNN, and RNN

### Team Members

- Member 1:
- Member 2:

---

## 1. Introduction

Deep learning models learn by adjusting weights based on the error between predicted and actual output. This learning process is done using **backpropagation**.

Backpropagation:

- Computes error (loss)
- Finds gradients using calculus
- Updates weights using gradient descent

Two main steps:

1. **Forward Propagation** -> prediction
2. **Backward Propagation** -> error correction

Even though the idea is the same, the method changes in **ANN, CNN, and RNN** because of different architectures.

---

## 2. Backpropagation Overview

### Core Formula (Chain Rule)

$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}
$$

Where:

- $L$ = loss
- $w$ = weight
- $a$ = activation
- $z$ = input

---

### Weight Update Rule

$$
w_{new} = w_{old} - \eta \frac{\partial L}{\partial w}
$$

- $\eta$ = learning rate

---

### Basic Algorithm

1. Initialize weights
2. Forward pass
3. Compute loss
4. Backward pass (gradients)
5. Update weights
6. Repeat

---

## 3. Backpropagation in ANN

### Architecture

- Fully connected layers
- Every neuron connects to the next layer

---

### Formulas

**Forward:**

$$
z = Wx + b
$$

$$
a = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

**Loss (MSE):**

$$
L = \frac{1}{2}(y - \hat{y})^2
$$

---

### Gradient

$$
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial W}
$$

---

### Algorithm (ANN)

1. Initialize weights
2. Compute $z = Wx + b$
3. Apply activation
4. Compute loss
5. Calculate error
6. Backpropagate gradients
7. Update weights

---

### Example Calculation

Given:

- $x = 1$
- $W = 0.5$
- $b = 0$

Forward:

- $z = 0.5$
- $\hat{y} \approx 0.62$

Loss:

- $L \approx 0.072$

Backpropagation adjusts the weight slightly to reduce error.

---

## 4. Backpropagation in CNN

### Architecture

- Uses filters (kernels)
- Works on images (spatial data)
- Includes convolution + pooling

---

### Convolution Formula

$$
F(i,j) = \sum I(i+m, j+n) \cdot K(m,n)
$$

---

### Gradient Formula

$$
\frac{\partial L}{\partial K} = I * \delta
$$

- $I$ = input
- $\delta$ = error map

---

### Algorithm (CNN)

1. Apply convolution
2. Generate feature maps
3. Apply activation (ReLU)
4. Pooling
5. Compute loss
6. Backpropagate through filters
7. Update kernels

---

### Example Calculation

Input:

```
1 2 3
4 5 6
7 8 9
```

Kernel:

```
1 0
0 -1
```

Output:

$$
(1\times1 + 2\times0 + 4\times0 + 5\times(-1)) = -4
$$

Backpropagation updates kernel values.

---

## 5. Backpropagation in RNN

### Architecture

- Used for sequences (text, speech)
- Has memory (hidden state)

---

### Hidden State Formula

$$
h_t = \tanh(Wx_t + Uh_{t-1} + b)
$$

---

### Loss

$$
L = \sum_{t=1}^{T}(y_t - \hat{y}_t)^2
$$

---

### BPTT (Backpropagation Through Time)

Gradients flow **back through time steps**.

---

### Algorithm (RNN)

1. Input sequence
2. Compute hidden states
3. Compute outputs
4. Calculate loss
5. Unroll network
6. Backpropagate through time
7. Update weights

---

### Example

Input: $(1, 2, 3)$

Each step updates hidden state:

$$
h_t = \tanh(...)
$$

Errors accumulate across time steps.

---

## 6. Differences: ANN vs CNN vs RNN

| Feature | ANN | CNN | RNN |
| --- | --- | --- | --- |
| Structure | Fully connected | Filters | Recurrent |
| Data Type | Tabular | Image | Sequential |
| Gradient Flow | Layer to layer | Spatial | Through time |
| Weight Sharing | No | Yes | Yes |
| Method | Standard BP | Conv Backprop | BPTT |
| Memory | No | No | Yes |

---

## 7. Code Implementation (Concept)

From this repository:

### ANN (`ANN_Backpropagation.py`)

- Uses NumPy
- Manual forward + backward pass

### CNN (`CNN_Backpropagation.py`)

- Uses NumPy
- Manual convolution + filter gradient update

### RNN (`RNN_Backpropagation.py`)

- Uses NumPy
- Simplified BPTT-style weight update on sequence states

---

## 8. Output (Expected)

- ANN -> updated weights
- CNN -> updated filters + feature map
- RNN -> updated recurrent weights and states

---

## 9. References

- GeeksforGeeks - Backpropagation in Neural Network
- GeeksforGeeks - Convolutional Neural Network (CNN)
- GeeksforGeeks - Recurrent Neural Network (RNN)
- Deep Learning Book - Ian Goodfellow, Yoshua Bengio, and Aaron Courville

---

## 10. Conclusion

Backpropagation is the **core learning mechanism** in deep learning.

- ANN -> simple layer-based learning
- CNN -> learns spatial features using filters
- RNN -> learns sequences using time-based learning

Understanding these differences helps choose the right model:

- Images -> CNN
- Sequences -> RNN
- General data -> ANN
