#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

def log_transform(image, c=1.0):
    """
    对数变换： s = c * log(1 + r_norm)
    (先归一化 r 到 [0,1]，再变换，最后映射回[0,255])
    """
    # 转为float并归一化
    img_float = image.astype(np.float32) / 255.0
    # 对数变换
    log_img = c * np.log1p(img_float)  # log1p(x) = log(1+x)
    # 归一化后映射回[0,255]
    log_img_norm = cv2.normalize(log_img, None, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    out = (log_img_norm * 255).astype(np.uint8)
    return out

def main():
    # 1) 读取图像
    img = cv2.imread('pic/a.jpg', cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("无法读取图像 'pic/a.jpg'，请确认路径是否正确。")

    # 2) 设置多个 c 值进行演示
    c_values = [1.0, 2.0, 5.0]

    plt.figure(figsize=(12, 4))
    for i, c in enumerate(c_values, 1):
        log_img = log_transform(img, c=c)
        plt.subplot(1, len(c_values), i)
        plt.imshow(log_img, cmap='gray')
        plt.title(f"Log Transform c={c}")
        plt.axis('off')

    plt.suptitle("Log Transform with Different c", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
