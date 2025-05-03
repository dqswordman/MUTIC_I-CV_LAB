import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import reconstruction

# 1. 读取图像并转为灰度图
img = cv2.imread('pic/lab12.1.png', cv2.IMREAD_GRAYSCALE)

# 判断图像是否成功读取
if img is None:
    raise FileNotFoundError("图像未找到，请确保路径为 'pic/pic_lab_12.png' 且文件存在。")

# 2. 构造结构元素（1x21 的水平核）
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 1))

# 3. 对图像进行灰度膨胀（增强横向结构）
dilated = cv2.dilate(img, kernel)

# 4. 使用灰度形态学重建（基于膨胀的重建）
# 原图作为种子，膨胀图作为掩模，限制其重建范围
reconstructed = reconstruction(seed=img, mask=dilated, method='dilation')

# 5. Top-hat by reconstruction = 原图 - 重建图（突出细节区域）
tophat = cv2.subtract(img, reconstructed.astype(np.uint8))

# 6. 显示所有中间结果和最终图像
titles = ['Original Image', 'Dilated Image (1x21)', 'Reconstruction', 'Top-hat by Reconstruction']
images = [img, dilated, reconstructed, tophat]

plt.figure(figsize=(14, 6))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
