#upgrade by 10.1
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取灰度图像
image_path = "pic/pic_lab10_3.png"  # <-- 请替换为你的图片路径
original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if original is None:
    raise ValueError("图像读取失败，请检查文件路径是否正确。")

# 2. 将灰度图按照阈值128二值化：<128=0, >=128=255
threshold_value = 128
ret, binary_default = cv2.threshold(original, threshold_value, 255, cv2.THRESH_BINARY)

# 3. 让你选择前景：这里我们假设前景是白色(255)
#   如果前景是白色，则直接使用 binary_default
#   如果前景是黑色，请改用 binary_black_fg = cv2.bitwise_not(binary_default)
binary_fg = binary_default  # 当前示例：白色为前景
#binary_fg = cv2.bitwise_not(binary_default)  # 如果需要黑色为前景，可取消注释这一行

# 4. 构建11x11全1结构元素
kernel_size = 21
kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

# 5. 形态学腐蚀和膨胀
eroded = cv2.erode(binary_fg, kernel, iterations=1)
dilated = cv2.dilate(binary_fg, kernel, iterations=1)

# 6. 可视化对比
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(original, cmap='gray')
plt.title("Original (gray)")

plt.subplot(1, 3, 2)
plt.imshow(eroded, cmap='gray')
plt.title("Eroded (eroded)")

plt.subplot(1, 3, 3)
plt.imshow(dilated, cmap='gray')
plt.title("Dilated (dilated)")

plt.tight_layout()
plt.show()
