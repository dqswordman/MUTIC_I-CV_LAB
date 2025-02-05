#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

def gamma_transform(image, gamma=1.0):
    """
    幂律变换： s = r^gamma (r先归一化)
    """
    img_norm = image.astype(np.float32) / 255.0
    gamma_img = np.power(img_norm, gamma)
    out = (gamma_img * 255).astype(np.uint8)
    return out

def main():
    # 1) 读取图像
    img = cv2.imread('pic/a.jpg', cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("无法读取图像 'pic/a.jpg'，请确认路径是否正确。")

    # 2) 演示多种 gamma
    gamma_list = [0.4, 0.7, 1.5, 2.0]

    plt.figure(figsize=(10, 6))
    for i, gm in enumerate(gamma_list, 1):
        gm_img = gamma_transform(img, gm)
        plt.subplot(2, 2, i)
        plt.imshow(gm_img, cmap='gray')
        plt.title(f"Gamma = {gm}")
        plt.axis('off')

    plt.suptitle("Gamma Correction with Different Gamma", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
