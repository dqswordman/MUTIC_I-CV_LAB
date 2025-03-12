import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


##############################################
# 1. Color space conversion helper functions #
##############################################

def rgb_to_cmyk(r, g, b):
    # r, g, b ∈ [0,1]
    if (r, g, b) == (0, 0, 0):
        return 0, 0, 0, 1

    k = 1 - max(r, g, b)
    c = (1 - r - k) / (1 - k) if (1 - k) != 0 else 0
    m = (1 - g - k) / (1 - k) if (1 - k) != 0 else 0
    y = (1 - b - k) / (1 - k) if (1 - k) != 0 else 0
    return c, m, y, k


def cmyk_to_rgb(c, m, y, k):
    # c, m, y, k ∈ [0,1]
    r = 1 - min(1, c * (1 - k) + k)
    g = 1 - min(1, m * (1 - k) + k)
    b = 1 - min(1, y * (1 - k) + k)
    return r, g, b


def rgb_to_hsi(r, g, b):
    # r, g, b ∈ [0,1]
    eps = 1e-8
    i = (r + g + b) / 3.0
    min_val = min(r, g, b)

    # Saturation
    if i > 0:
        s = 1 - (min_val / i)
    else:
        s = 0

    # Hue
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + eps
    theta = np.arccos(num / den)

    if b <= g:
        h = theta / (2 * np.pi)
    else:
        h = (2 * np.pi - theta) / (2 * np.pi)

    return h, s, i


def hsi_to_rgb(h, s, i):
    # h, s, i ∈ [0,1], h is normalized to [0,2π] internally
    eps = 1e-8
    H = 2 * np.pi * h
    if H < 2 * np.pi / 3:
        b = i * (1 - s)
        r = i * (1 + s * np.cos(H) / (np.cos(np.pi / 3 - H) + eps))
        g = 3 * i - (r + b)
    elif H < 4 * np.pi / 3:
        H -= 2 * np.pi / 3
        r = i * (1 - s)
        g = i * (1 + s * np.cos(H) / (np.cos(np.pi / 3 - H) + eps))
        b = 3 * i - (r + g)
    else:
        H -= 4 * np.pi / 3
        g = i * (1 - s)
        b = i * (1 + s * np.cos(H) / (np.cos(np.pi / 3 - H) + eps))
        r = 3 * i - (g + b)

    r = np.clip(r, 0, 1)
    g = np.clip(g, 0, 1)
    b = np.clip(b, 0, 1)
    return r, g, b


#########################
# 2. Load the image data #
#########################
image_path = 'pic/a.jpg'
img_pil = Image.open(image_path).convert('RGB')
img_arr = np.array(img_pil, dtype=np.float32) / 255.0  # shape: (H, W, 3)

height, width, _ = img_arr.shape

######################################################
# 3. Reduce brightness by 30% in three color spaces  #
######################################################

# 3.1 RGB approach
rgb_darker = np.clip(img_arr * 0.7, 0, 1)

# 3.2 CMYK approach
cmyk_darker = np.zeros_like(img_arr)
for h_ in range(height):
    for w_ in range(width):
        r, g, b = img_arr[h_, w_]
        c, m, y, k = rgb_to_cmyk(r, g, b)
        # Brightness ~ (1 - k)
        # We want new brightness = 0.7 * (1 - k)
        # => new k = 1 - 0.7*(1 - k) = 0.3 + 0.7*k
        k_new = 0.3 + 0.7 * k
        r_new, g_new, b_new = cmyk_to_rgb(c, m, y, k_new)
        cmyk_darker[h_, w_] = [r_new, g_new, b_new]

cmyk_darker = np.clip(cmyk_darker, 0, 1)

# 3.3 HSI approach
hsi_darker = np.zeros_like(img_arr)
for h_ in range(height):
    for w_ in range(width):
        r, g, b = img_arr[h_, w_]
        H, S, I = rgb_to_hsi(r, g, b)
        I_new = 0.7 * I
        r_new, g_new, b_new = hsi_to_rgb(H, S, I_new)
        hsi_darker[h_, w_] = [r_new, g_new, b_new]

hsi_darker = np.clip(hsi_darker, 0, 1)

############################
# 4. Display comparison    #
############################
fig1, axes1 = plt.subplots(1, 4, figsize=(16, 5))
axes1[0].imshow(img_arr)
axes1[0].set_title("Original")
axes1[0].axis("off")

axes1[1].imshow(rgb_darker)
axes1[1].set_title("RGB Darker 30%")
axes1[1].axis("off")

axes1[2].imshow(cmyk_darker)
axes1[2].set_title("CMYK Darker 30%")
axes1[2].axis("off")

axes1[3].imshow(hsi_darker)
axes1[3].set_title("HSI Darker 30%")
axes1[3].axis("off")

plt.tight_layout()
plt.show()

##############################
# 5. Compute difference imgs #
##############################
diff_rgb_cmyk = np.abs(rgb_darker - cmyk_darker)
diff_rgb_hsi = np.abs(rgb_darker - hsi_darker)
diff_cmyk_hsi = np.abs(cmyk_darker - hsi_darker)

# Grayscale differences by averaging channels
diff_rgb_cmyk_gray = np.mean(diff_rgb_cmyk, axis=2)
diff_rgb_hsi_gray = np.mean(diff_rgb_hsi, axis=2)
diff_cmyk_hsi_gray = np.mean(diff_cmyk_hsi, axis=2)

##############################
# 6. Display difference imgs #
##############################
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 6))

# Top row: color difference
axes2[0, 0].imshow(diff_rgb_cmyk)
axes2[0, 0].set_title("RGB vs CMYK (Color Diff)")
axes2[0, 0].axis("off")

axes2[0, 1].imshow(diff_rgb_hsi)
axes2[0, 1].set_title("RGB vs HSI (Color Diff)")
axes2[0, 1].axis("off")

axes2[0, 2].imshow(diff_cmyk_hsi)
axes2[0, 2].set_title("CMYK vs HSI (Color Diff)")
axes2[0, 2].axis("off")

# Bottom row: grayscale difference
axes2[1, 0].imshow(diff_rgb_cmyk_gray, cmap='gray')
axes2[1, 0].set_title("RGB vs CMYK (Gray Diff)")
axes2[1, 0].axis("off")

axes2[1, 1].imshow(diff_rgb_hsi_gray, cmap='gray')
axes2[1, 1].set_title("RGB vs HSI (Gray Diff)")
axes2[1, 1].axis("off")

axes2[1, 2].imshow(diff_cmyk_hsi_gray, cmap='gray')
axes2[1, 2].set_title("CMYK vs HSI (Gray Diff)")
axes2[1, 2].axis("off")

plt.tight_layout()
plt.show()
