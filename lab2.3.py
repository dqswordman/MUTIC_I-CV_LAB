import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1) 读取图像 (以灰度模式)
    #    若原图是彩色，会被自动转换为单通道
    img = cv.imread('pic/grayscale2.jpg', cv.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Could not open or find 'pic/grayscale.jpg'.")
        return

    # 2) 进行直方图均衡化
    equalized = cv.equalizeHist(img)

    # 3) 用 OpenCV 的窗口显示处理结果
    cv.imshow('Original Image', img)
    cv.imshow('Equalized Image', equalized)

    # 4) 计算并可视化直方图（使用 Matplotlib）
    #    计算直方图
    hist_original = cv.calcHist([img], [0], None, [256], [0,256])
    hist_equalized = cv.calcHist([equalized], [0], None, [256], [0,256])

    # 画出直方图对比
    plt.figure(figsize=(10,5))

    # 原图直方图
    plt.subplot(1, 2, 1)
    plt.title('Original Histogram')
    plt.xlabel('Gray Level')
    plt.ylabel('Pixel Count')
    plt.plot(hist_original, color='gray')

    # 均衡化后直方图
    plt.subplot(1, 2, 2)
    plt.title('Equalized Histogram')
    plt.xlabel('Gray Level')
    plt.ylabel('Pixel Count')
    plt.plot(hist_equalized, color='black')

    plt.tight_layout()
    plt.show()

    # 5) 等待按键再关闭所有窗口
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
