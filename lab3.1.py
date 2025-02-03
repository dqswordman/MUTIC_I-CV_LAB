#lab 3.1
"""
单张图像负片操作示例
1. 读取原图(灰度模式)
2. 对图像做负片变换
3. 输出"Original Image"和"Negative Image"并排显示
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. 读取原图，灰度模式
    img_gray = cv2.imread("pic/a.jpg", cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print("Error: 无法读取图像，请检查文件名或路径。")
        return

    # 2. 获取负片图像(方法1: 255 - img)
    negative_img = 255 - img_gray

    # 如果你喜欢使用OpenCV提供的bitwise_not，也可用下面替代：
    # negative_img = cv2.bitwise_not(img_gray)

    # 3. 用Matplotlib对比显示
    # 创建并排的两个子图（1行2列）
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))

    # 左侧显示原始灰度图
    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('on')  # 显示坐标刻度，如不需要可改为 off

    # 右侧显示负片图
    axes[1].imshow(negative_img, cmap='gray')
    axes[1].set_title('Negative Image')
    axes[1].axis('on')  # 同理，可改为 off

    # 自动调整子图布局，防止标题或坐标轴信息遮挡
    plt.tight_layout()

    # 显示最终窗口
    plt.show()

    # 4. 如果想把负片图保存到硬盘，可以使用OpenCV或Matplotlib
    # cv2.imwrite("portrait_negative.jpg", negative_img)
    # plt.imsave("compare_display.png", negative_img, cmap='gray')

if __name__ == "__main__":
    main()
