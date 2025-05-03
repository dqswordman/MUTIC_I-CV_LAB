import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

# 1. Choose a random L
L = random.randint(30, 60)  # pixels
print(f"Chosen L = {L}")

# Define image canvas size (make it big enough: 5L × 5L)
canvas_size = 5 * L
A = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

# Center coordinates
cx, cy = canvas_size // 2, canvas_size // 2

# Draw vertical bar of the cross (width=L, height=3L)
v_top = cy - (3 * L) // 2
v_bottom = cy + (3 * L) // 2
v_left = cx - L // 2
v_right = cx + (L + 1) // 2  # inclusive
cv2.rectangle(A, (v_left, v_top), (v_right - 1, v_bottom - 1), 255, thickness=-1)

# Draw horizontal bar of the cross (height=L, width=3L)
h_left = cx - (3 * L) // 2
h_right = cx + (3 * L) // 2
h_top = cy - L // 2
h_bottom = cy + (L + 1) // 2
cv2.rectangle(A, (h_left, h_top), (h_right - 1, h_bottom - 1), 255, thickness=-1)

# B1: vertical rectangle (width L/4, height L)
w1 = max(1, L // 4)
B1 = np.zeros((L, w1), dtype=np.uint8)
B1[:, :] = 1

# B2: mini cross arms size L/4
a2 = max(1, L // 4)
cross_size = 2 * a2 + 1
B2 = np.zeros((cross_size, cross_size), dtype=np.uint8)
B2[a2, :] = 1
B2[:, a2] = 1

# B3: square side L/2
s3 = max(1, L // 2)
B3 = np.ones((s3, s3), dtype=np.uint8)

# B4: disk radius L/2
r4 = max(1, L // 2)
diam4 = 2 * r4 + 1
B4 = np.zeros((diam4, diam4), dtype=np.uint8)
cv2.circle(B4, (r4, r4), r4, 1, thickness=-1)

# Helper functions for binary morphology with arbitrary kernels
def erode(img, kernel):
    return cv2.erode(img, kernel, iterations=1)

def dilate(img, kernel):
    return cv2.dilate(img, kernel, iterations=1)

# (a) (A⊖B4)⊕B2
A_bin = (A > 0).astype(np.uint8)
E_a = erode(A_bin, B4)
R_a = dilate(E_a, B2)

# (b) (A⊖B1)⊕B3
E_b = erode(A_bin, B1)
R_b = dilate(E_b, B3)

# (c) (A⊕B1)⊕B3   (due to associativity we directly chain dilations)
D_c = dilate(A_bin, B1)
R_c = dilate(D_c, B3)

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.ravel()
axes[0].imshow(A_bin, cmap='gray')
axes[0].set_title("Original A")

axes[1].imshow(R_a, cmap='gray')
axes[1].set_title("(a) (A⊖B4)⊕B2")

axes[2].imshow(R_b, cmap='gray')
axes[2].set_title("(b) (A⊖B1)⊕B3")

axes[3].imshow(R_c, cmap='gray')
axes[3].set_title("(c) (A⊕B1)⊕B3")

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()
