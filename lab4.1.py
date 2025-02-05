#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

def negative_transform(image):
    """
    负片变换：s = 255 - r
    """
    return 255 - image

def main():
    # 1) 读取图像（灰度模式）
    img = cv2.imread('pic/a.jpg', cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("无法读取图像 'pic/a.jpg'，请确认路径是否正确。")

    # 2) 应用负片变换
    neg_img = negative_transform(img)

    # 3) 可视化对比
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original"), plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(neg_img, cmap='gray')
    plt.title("Negative Transform"), plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
