#腐蚀和膨胀测试
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取图像（请根据实际情况修改图片路径）
image_path = "pic/pic_lab10_3.png"
original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 如果读取失败，original会是None，可以先简单判断
if original is None:
    raise ValueError("图像读取失败，请检查文件路径是否正确。")

# 2. 创建11x11的全1结构元素（uint8类型即可）
kernel = np.ones((15, 15), dtype=np.uint8)

# 3. 进行腐蚀操作
eroded = cv2.erode(original, kernel)

# 4. 进行膨胀操作
dilated = cv2.dilate(original, kernel)

# 5. 统计腐蚀和膨胀改变的像素数
#   - 对腐蚀而言：原本是前景(255)的像素，在腐蚀结果里变成0(背景)，就是“被腐蚀”的像素。
#   - 对膨胀而言：原本是背景(0)的像素，在膨胀结果里变成255(前景)，就是“被膨胀”的像素。
eroded_pixels = np.sum((original == 255) & (eroded == 0))
dilated_pixels = np.sum((original == 0) & (dilated == 255))

print("被腐蚀的像素数量:", eroded_pixels)
print("被膨胀的像素数量:", dilated_pixels)

# 6. 使用Matplotlib可视化
plt.figure(figsize=(12, 4))

# 显示原图
plt.subplot(1, 3, 1)
plt.imshow(original, cmap='gray')
plt.title("Original")

# 显示腐蚀后图像
plt.subplot(1, 3, 2)
plt.imshow(eroded, cmap='gray')
plt.title("Eroded (eroded)")

# 显示膨胀后图像
plt.subplot(1, 3, 3)
plt.imshow(dilated, cmap='gray')
plt.title("Dilated (dilated)")

plt.tight_layout()
plt.show()
