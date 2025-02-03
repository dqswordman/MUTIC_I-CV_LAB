# lab 3.3
"""
局部邻域均值滤波示例，分别使用窗口大小 (10,10), (20,20), (30,30)
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. 读取原始图像
    #   - 可以是灰度或彩色图像。此处假设灰度，若是彩色会自动使用三通道均值滤波
    img = cv2.imread("pic/lab3_3.png", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: 无法读取图像，请检查文件路径。")
        return

    # 2. 构建不同的窗口尺寸列表
    kernel_sizes = [(10, 10), (20, 20), (30, 30)]

    # 3. 对图像分别应用均值滤波
    results = []
    for ks in kernel_sizes:
        # 使用 OpenCV 提供的 blur( ) 函数即可实现邻域平均
        smoothed = cv2.blur(img, ks)
        results.append((ks, smoothed))

    # 4. 显示原图和滤波结果对比
    #   - 行数=1+len(kernel_sizes)，因为还要包含原图
    row_count = 1 + len(kernel_sizes)
    fig, axes = plt.subplots(row_count, 1, figsize=(6, 8))

    # (a) 显示原图
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # (b) 分别显示不同窗口尺寸的滤波结果
    for i, (ks, res_img) in enumerate(results, start=1):
        axes[i].imshow(res_img, cmap='gray')
        axes[i].set_title(f"Mean Filter (Kernel = {ks[0]}×{ks[1]})")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    # 5. 如果想保存结果到硬盘，可以对每个结果单独保存
    # for ks, res_img in results:
    #     filename = f"smooth_{ks[0]}x{ks[1]}.jpg"
    #     cv2.imwrite(filename, res_img)
    # print("所有结果已保存到当前目录。")

if __name__ == "__main__":
    main()
