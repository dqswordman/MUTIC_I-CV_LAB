import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    # 读取原图
    img = cv2.imread('pic/a.jpg')
    if img is None:
        raise FileNotFoundError("找不到图片，请检查路径 pic/a.jpg 是否正确。")

    # 1. 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 拉普拉斯算子增强
    lap = cv2.Laplacian(gray, cv2.CV_64F)  # 计算二阶导数
    lap = cv2.convertScaleAbs(lap)  # 转回 uint8 类型

    # 3. 反锐化掩模（Unsharp Mask）
    #    先做高斯模糊，再与原图做加权融合
    blur = cv2.GaussianBlur(gray, (9, 9), 10)
    unsharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    # 4. 组合空间增强 —— 将反锐化结果与拉普拉斯结果融合
    combined = cv2.addWeighted(unsharp, 0.5, lap, 0.5, 0)

    # 可视化：依次展示 原图 → 灰度图 → 拉普拉斯 → 反锐化 → 组合结果
    titles = [
        'Original Image',
        'Grayscale Image',
        'Laplacian Result',
        'Unsharp Mask Result',
        'Combined Enhancement'
    ]
    images = [
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),  # 转为 RGB 以正确显示
        gray,
        lap,
        unsharp,
        combined
    ]

    for title, image in zip(titles, images):
        plt.figure()
        if image.ndim == 2:  # 灰度图
            plt.imshow(image, cmap='gray')
        else:  # 彩色图
            plt.imshow(image)
        plt.title(title)
        plt.axis('off')

    plt.show()


if __name__ == '__main__':
    main()
