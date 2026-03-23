# **Mini Seminar – Deep Learning**

## **Backpropagation in ANN, CNN, and RNN**

### **Team Members**

* Member 1:
* Member 2:

---

## **1. Introduction**

Deep learning models improve their performance by continuously adjusting internal parameters called weights. This adjustment is done based on the difference between predicted output and actual output.

The algorithm used for this learning process is **backpropagation**.

Backpropagation:

* Measures prediction error using a loss function
* Computes gradients using calculus (chain rule)
* Updates weights to minimize error

The learning process has two stages:

1. **Forward Pass** → generates output
2. **Backward Pass** → corrects errors

Although the concept remains the same, the implementation varies in **ANN, CNN, and RNN** due to their different structures.

---

## **2. Backpropagation Overview**

### **Chain Rule Formula**

$$
\frac{\partial L}{\partial w} =
\frac{\partial L}{\partial a} \cdot
\frac{\partial a}{\partial z} \cdot
\frac{\partial z}{\partial w}
$$

Where:

* $L$ = loss
* $w$ = weight
* $a$ = activation output
* $z$ = input to activation

---

### **Weight Update Equation**

$$
w_{new} = w_{old} - \eta \cdot \frac{\partial L}{\partial w}
$$

* $\eta$ = learning rate

---

### **General Steps**

1. Initialize parameters
2. Perform forward computation
3. Calculate loss
4. Compute gradients
5. Update parameters
6. Repeat process

---

## **3. Backpropagation in ANN**

### **Architecture**

Artificial Neural Networks consist of fully connected layers where each neuron is linked to every neuron in the next layer.

---

### **Model Equations**

**Forward computation:**

$$
z = Wx + b
$$

$$
a = \frac{1}{1 + e^{-z}}
$$

**Loss function:**

$$
L = \frac{1}{2}(y - \hat{y})^2
$$

---

### **Gradient Computation**

$$
\frac{\partial L}{\partial W} =
(\hat{y} - y) \cdot \sigma'(z) \cdot x
$$

---

### **Algorithm (Based on Implementation)**

1. Initialize weight and bias
2. Compute output using sigmoid activation
3. Calculate error between target and prediction
4. Compute gradient using derivative of sigmoid
5. Update weight and bias using learning rate

---

### **Example (From Code)**

* Input = 0.8
* Weight initialized ≈ 0.3
* Output computed using sigmoid
* Loss calculated using MSE

After backpropagation:

* Weight is slightly adjusted to reduce error

---

## **4. Backpropagation in CNN**

### **Architecture**

Convolutional Neural Networks are designed for image data. Instead of full connections, they use:

* Filters (kernels)
* Feature maps
* Local receptive fields

---

### **Convolution Operation**

$$
Output = \sum (Image \times Filter)
$$

---

### **Gradient Calculation**

$$
\frac{\partial L}{\partial Filter} = Input \star Gradient
$$

Here:

* Input image patches are used
* Gradient is propagated from next layer

---

### **Algorithm (Based on Implementation)**

1. Apply filter on image to generate feature map
2. Compute output values for each region
3. Assume gradient from next layer
4. Compute gradient of filter using input patches
5. Update filter values

---

### **Example (From Code)**

Input image (modified values):

```
2 1 0
3 5 2
4 6 1
```

Filter:

```
0.5  -1
1     0
```

Feature map is generated using convolution.
Then filter is updated using computed gradients.

---

## **5. Backpropagation in RNN**

### **Architecture**

Recurrent Neural Networks are used for sequential data. They maintain a hidden state that carries information across steps.

---

### **Hidden State Equation**

$$
h_t = \tanh(W_x x_t + W_h h_{t-1})
$$

---

### **Loss Function**

$$
L = \frac{1}{2}(y - \hat{y})^2
$$

---

### **Backpropagation Through Time (BPTT)**

In RNN, gradients are propagated backward across time steps instead of layers.

---

### **Algorithm (Based on Implementation)**

1. Process input sequence step by step
2. Compute hidden state at each step
3. Take final output from last state
4. Compute loss
5. Calculate gradients using last timestep
6. Update input and recurrent weights

---

### **Example (From Code)**

Sequence: (2, 1, 3)

* Hidden states are computed sequentially
* Final state is used for prediction
* Error is calculated
* Weights are updated using simplified BPTT

---

## **6. Differences Between ANN, CNN, and RNN**

| Feature         | ANN             | CNN             | RNN             |
| --------------- | --------------- | --------------- | --------------- |
| Structure       | Fully connected | Filter-based    | Recurrent       |
| Input Type      | Numeric data    | Images          | Sequential data |
| Gradient Flow   | Layer-wise      | Spatial regions | Time steps      |
| Weight Sharing  | No              | Yes (filters)   | Yes (time)      |
| Learning Method | Standard BP     | Convolution BP  | BPTT            |
| Memory          | No              | No              | Yes             |

---

## **7. Code Implementation**

The implementation is done using **NumPy only**, without external libraries.

* **ANN** → single neuron with sigmoid and manual gradient update
* **CNN** → manual convolution and filter gradient calculation
* **RNN** → sequential processing with hidden state and simplified BPTT

---

## **8. Output**

* ANN → updated weight and reduced loss
* CNN → feature map and updated filter
* RNN → hidden states and updated weights

---

## **9. References**

* GeeksforGeeks – Backpropagation in Neural Networks
* GeeksforGeeks – CNN Concepts
* GeeksforGeeks – RNN Concepts
* Deep Learning Book – Ian Goodfellow

---

## **10. Conclusion**

Backpropagation is essential for training all neural network models.

* ANN updates weights through simple layer connections
* CNN adjusts filters to capture spatial patterns
* RNN propagates errors across time steps

Each model applies backpropagation differently depending on the type of data and structure.

👉 Selecting the correct model depends on the problem:

* Image tasks → CNN
* Sequential tasks → RNN
* Basic prediction → ANN

---

✅ Now your file is:

* **Aligned with your code**
* **Rewritten (plagiarism-safe)**
* **Simple but strong for viva**

---

If you want next:

* I can make **PPT slides matching THIS content exactly**
* Or give **2-minute + 5-minute speaking script** 🎤
