#upgrade by 10.2

import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    # 1. 读取灰度图像
    image_path = "pic/pic_lab10_4.png"  # 请替换为你的图片路径
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if original is None:
        raise ValueError("图像读取失败，请检查文件路径是否正确。")

    # 2. 将灰度图按照阈值128二值化：<128=0, >=128=255
    threshold_value = 128
    ret, binary_default = cv2.threshold(original, threshold_value, 255, cv2.THRESH_BINARY)

    # ==========  2.1 选择前景是白色还是黑色  ==========
    # 如果前景是白色(255)，直接用 binary_default 即可
    # 如果前景是黑色(0)，可把下面注释解开以使用 black_fg
    #black_fg = cv2.bitwise_not(binary_default)
    #binary_fg = black_fg

    # 这里演示：以白色为前景
    binary_fg = binary_default

    # 3. 构建 11x11 全1结构元素
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    # 4. 基础形态学操作：腐蚀、膨胀
    eroded = cv2.erode(binary_fg, kernel, iterations=1)  # 腐蚀
    dilated = cv2.dilate(binary_fg, kernel, iterations=1)  # 膨胀

    # 5. 开运算(Opening) 与 闭运算(Closing)
    opened = cv2.morphologyEx(binary_fg, cv2.MORPH_OPEN, kernel)  # 先腐蚀再膨胀
    closed = cv2.morphologyEx(binary_fg, cv2.MORPH_CLOSE, kernel)  # 先膨胀再腐蚀

    # 6. 先开后闭 (Opening → Closing)
    opened_then_closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    # 7. 先闭后开 (Closing → Opening)
    closed_then_opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # 8. 可视化对比
    plt.figure(figsize=(16, 8))

    # (1) 原图
    plt.subplot(2, 4, 1)
    plt.imshow(original, cmap='gray')
    plt.title("1. Original")
    plt.axis('off')

    # (2) 二值化结果
    plt.subplot(2, 4, 2)
    plt.imshow(binary_fg, cmap='gray')
    plt.title("2. Binary")
    plt.axis('off')

    # (3) 腐蚀
    plt.subplot(2, 4, 3)
    plt.imshow(eroded, cmap='gray')
    plt.title("3. Eroded (eroded)")
    plt.axis('off')

    # (4) 膨胀
    plt.subplot(2, 4, 4)
    plt.imshow(dilated, cmap='gray')
    plt.title("4. Dilated (dilated)")
    plt.axis('off')

    # (5) 开运算
    plt.subplot(2, 4, 5)
    plt.imshow(opened, cmap='gray')
    plt.title("5. Opening (opening)")
    plt.axis('off')

    # (6) 闭运算
    plt.subplot(2, 4, 6)
    plt.imshow(closed, cmap='gray')
    plt.title("6. Closing (closing)")
    plt.axis('off')

    # (7) 先开后闭
    plt.subplot(2, 4, 7)
    plt.imshow(opened_then_closed, cmap='gray')
    plt.title("7. Opening → Closing")
    plt.axis('off')

    # (8) 先闭后开
    plt.subplot(2, 4, 8)
    plt.imshow(closed_then_opened, cmap='gray')
    plt.title("8. Closing → Opening")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
