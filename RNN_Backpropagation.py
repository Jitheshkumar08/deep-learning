import numpy as np

print("\n---- RNN Sequence Learning ----")

# Initial parameters
w_in = 0.4
w_rec = 0.7
lr = 0.05

sequence = [2, 1, 3]
target_val = 1

h_state = 0
states = []

# Forward sequence pass
for val in sequence:
    h_state = np.tanh(w_in * val + w_rec * h_state)
    states.append(h_state)

print("Sequence states:", states)

# Final output
pred = states[-1]

# Loss
loss = 0.5 * (target_val - pred) ** 2
print("Loss:", loss)

# Backprop (simplified BPTT)
delta = (pred - target_val) * (1 - pred**2)

grad_in = delta * sequence[-1]
grad_rec = delta * states[-2]

# Update
w_in -= lr * grad_in
w_rec -= lr * grad_rec

print("Updated input weight:", w_in)
print("Updated recurrent weight:", w_rec)
print("----------------------")