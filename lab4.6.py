#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

def bitplane_slicing(image):
    """
    返回一个列表 [plane0, plane1, ..., plane7]
    plane0是最低位(LSB)，plane7是最高位(MSB)
    """
    planes = []
    for i in range(8):
        plane = ((image >> i) & 1) * 255
        planes.append(plane.astype(np.uint8))
    return planes

def main():
    # 1) 读取图像
    img = cv2.imread('pic/a.jpg', cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("无法读取图像 'pic/a.jpg'，请确认路径是否正确。")

    # 2) 提取8个位平面
    planes = bitplane_slicing(img)

    # 3) 可视化8个位平面
    plt.figure(figsize=(12,6))
    for idx in range(8):
        # 最好从最高位到最低位显示，这里 plane7在最前
        plane_index = 7 - idx
        plt.subplot(2, 4, idx+1)
        plt.imshow(planes[plane_index], cmap='gray')
        plt.title(f"Bit Plane {plane_index}")
        plt.axis('off')
    plt.suptitle("Bit Plane Slicing (Highest -> Lowest)", fontsize=14)
    plt.tight_layout()
    plt.show()

    # 4) 示例：只用最高两位(plane7, plane6)重构图像
    bit7 = planes[7] // 255  # 0或1
    bit6 = planes[6] // 255
    recon_76 = (bit7 << 7) + (bit6 << 6)

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original"), plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(recon_76, cmap='gray')
    plt.title("Reconstruct from Bit7 & Bit6"), plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
