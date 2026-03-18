import numpy as np

# Image (changed values)
img = np.array([
    [2, 1, 0],
    [3, 5, 2],
    [4, 6, 1]
], dtype=float)

# Filter (changed slightly)
flt = np.array([
    [0.5, -1],
    [1, 0]
], dtype=float)

lr = 0.05

print("\n---- CNN Operation ----")

# Convolution manually
result = []
for r in range(2):
    row = []
    for c in range(2):
        patch = img[r:r+2, c:c+2]
        val = np.sum(patch * flt)
        row.append(val)
    result.append(row)

result = np.array(result)

print("Generated Feature Map:\n", result)

# Gradient from next layer (different style)
grad_map = np.full(result.shape, 0.02)

grad_filter = np.zeros_like(flt)

# Backprop logic
for r in range(2):
    for c in range(2):
        grad_filter += img[r:r+2, c:c+2] * grad_map[r, c]

# Update filter
flt -= lr * grad_filter

print("\nFilter Gradient:\n", grad_filter)
print("Updated Filter:\n", flt)
print("----------------------")