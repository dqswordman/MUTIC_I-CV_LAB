#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

def piecewise_linear_slice(image, lower, upper, high_val=255):
    """
    灰度切片：
    - [lower, upper] 范围内的像素设为 high_val
    - 其余不变（也可设置为0或其他）
    """
    out = image.copy()
    mask = (image >= lower) & (image <= upper)
    out[mask] = high_val
    return out

def piecewise_linear_stretch(image, r1, s1, r2, s2):
    """
    分段对比度拉伸：
    - 0~r1 -> 0~s1 线性映射
    - r1~r2 -> s1~s2 线性映射
    - r2~255 -> s2~255 线性映射
    """
    out = np.zeros_like(image, dtype=np.float32)

    r1, s1, r2, s2 = float(r1), float(s1), float(r2), float(s2)
    # 计算三段斜率
    a1 = s1 / (r1 + 1e-5)
    a2 = (s2 - s1) / ((r2 + 1e-5) - r1)
    a3 = (255.0 - s2) / (255.0 - r2 + 1e-5)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            r = image[y, x]
            if r < r1:
                out[y, x] = a1 * r
            elif r <= r2:
                out[y, x] = s1 + a2 * (r - r1)
            else:
                out[y, x] = s2 + a3 * (r - r2)

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def main():
    # 1) 读取图像
    img = cv2.imread('pic/a.jpg', cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("无法读取图像 'pic/a.jpg'，请确认路径是否正确。")

    # 2) 灰度切片
    slice_img = piecewise_linear_slice(img, lower=100, upper=150, high_val=255)

    # 3) 三段对比度拉伸
    #   设置r1=50->s1=20, r2=200->s2=230 等参数可自行调节
    stretch_img = piecewise_linear_stretch(img, r1=50, s1=20, r2=200, s2=230)

    # 4) 可视化对比
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original"), plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(slice_img, cmap='gray')
    plt.title("Slice [100,150] -> 255"), plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(stretch_img, cmap='gray')
    plt.title("Piecewise Stretch"), plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
