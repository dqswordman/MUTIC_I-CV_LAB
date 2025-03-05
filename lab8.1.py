import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_gaussian_noise(image, mean=0, sigma=20):
    """
    向图像添加高斯噪声
    :param image: 输入图像(灰度图)
    :param mean: 均值
    :param sigma: 标准差
    :return: 添加噪声后图像(灰度)
    """
    # 生成与原图相同大小的高斯随机数矩阵
    gauss = np.random.normal(mean, sigma, image.shape)
    # 将随机数加到原图上
    noisy_image = image + gauss
    # 结果可能越界(>255 或 <0)，需进行裁剪
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

def add_rayleigh_noise(image, sigma=30):
    """
    向图像添加瑞利噪声
    :param image: 输入图像(灰度图)
    :param sigma: 瑞利分布参数
    :return: 添加噪声后图像(灰度)
    """
    # 生成瑞利分布随机数矩阵
    rayleigh = sigma * np.sqrt(-2 * np.log(1 - np.random.rand(*image.shape)))
    # 将随机数加到原图上
    noisy_image = image + rayleigh
    # 结果可能越界，需进行裁剪
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

def add_erlang_noise(image, k=2, lam=0.05):
    """
    向图像添加 Erlang/Gamma 噪声 (这里将形状参数 k, 率参数 lam 作为可调节量)
    :param image: 输入图像(灰度图)
    :param k: 形状参数(整数)
    :param lam: 率参数
    :return: 添加噪声后图像(灰度)
    """
    # 生成 Erlang 分布的随机噪声矩阵
    # Erlang(k, lam) 的生成方法之一：Gamma(k, theta=1/lam)
    # np.random.gamma(shape, scale, size) 中:
    # - shape: k
    # - scale: 1/lam
    erlang_noise = np.random.gamma(k, 1/lam, image.shape)
    # 将随机数加到原图上
    noisy_image = image + erlang_noise
    # 结果可能越界，需进行裁剪
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

# 1. 读取原图并转为灰度图
img_path = 'pic/fire.png'
original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# 如果读取失败，original 可能为 None
if original is None:
    print("图片读取失败，请检查图片路径是否正确。")
else:
    # 2. 添加三种不同噪声
    gauss_noisy = add_gaussian_noise(original, mean=0, sigma=20)
    rayleigh_noisy = add_rayleigh_noise(original, sigma=30)
    erlang_noisy = add_erlang_noise(original, k=2, lam=0.05)

    # 3. 绘制并排对比
    # 设置画布大小
    plt.figure(figsize=(12, 9))

    # 定义一个工具函数，用于绘制子图(原图/噪声图 + 直方图)
    def show_image_and_hist(image, title, position):
        """
        显示指定图像以及它对应的直方图
        :param image: 要显示的图像
        :param title: 标题，用于区分
        :param position: 子图起始位置
        """
        # 显示图像
        plt.subplot(position[0], position[1], position[2])
        plt.imshow(image, cmap='gray')
        plt.title(title + ' Image')
        plt.axis('off')  # 不显示坐标轴

        # 显示直方图
        plt.subplot(position[0], position[1], position[3])
        # 计算并绘制直方图
        plt.hist(image.ravel(), bins=256, range=(0, 256))
        plt.title(title + ' Histogram')

    # 原图及其直方图 (子图1, 2)
    show_image_and_hist(original, 'Original', (3, 4, 1, 2))
    # 高斯噪声图及其直方图 (子图3, 4)
    show_image_and_hist(gauss_noisy, 'Gaussian Noise', (3, 4, 3, 4))
    # 瑞利噪声图及其直方图 (子图5, 6)
    show_image_and_hist(rayleigh_noisy, 'Rayleigh Noise', (3, 4, 5, 6))
    # Erlang/Gamma 噪声图及其直方图 (子图7, 8)
    show_image_and_hist(erlang_noisy, 'Erlang Noise', (3, 4, 7, 8))

    # 调整子图间距，避免标题或坐标轴重叠
    plt.tight_layout()
    # 显示结果
    plt.show()
