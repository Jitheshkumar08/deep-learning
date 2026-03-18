import numpy as np

# Input and target
X = np.array([[1.0]])
y = np.array([[1.0]])

# Initialize weights
W = np.array([[0.5]])
b = np.array([[0.0]])

lr = 0.1

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Training loop
for epoch in range(5):
    # Forward pass
    z = np.dot(X, W) + b
    y_pred = sigmoid(z)

    # Loss (MSE)
    loss = 0.5 * (y - y_pred) ** 2

    # Backpropagation
    error = y_pred - y
    d_pred = error * sigmoid_derivative(y_pred)

    dW = np.dot(X.T, d_pred)
    db = d_pred

    # Update weights
    W -= lr * dW
    b -= lr * db

    print(f"Epoch {epoch+1}")
    print("Prediction:", y_pred)
    print("Loss:", loss)
    print("Updated Weight:", W)
    print("-" * 30)