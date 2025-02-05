#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

def threshold_transform(image, thresh_value):
    """
    二值化：
    r > thresh_value -> 255, 否则 -> 0
    """
    out_img = np.zeros_like(image, dtype=np.uint8)
    out_img[image > thresh_value] = 255
    return out_img

def main():
    # 1) 读取图像
    img = cv2.imread('pic/a.jpg', cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("无法读取图像 'pic/a.jpg'，请确认路径是否正确。")

    # 2) 设定多个阈值进行测试
    thresholds = [50, 128, 200]

    # 3) 可视化对比
    plt.figure(figsize=(12, 4))
    for i, th in enumerate(thresholds, 1):
        bin_img = threshold_transform(img, th)
        plt.subplot(1, len(thresholds), i)
        plt.imshow(bin_img, cmap='gray')
        plt.title(f"Threshold = {th}")
        plt.axis('off')

    plt.suptitle("Thresholding with Different T", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
