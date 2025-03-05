import cv2
import numpy as np
import matplotlib.pyplot as plt

# ========== 已有的三类噪声函数 ==========

def add_gaussian_noise(image, mean=0, sigma=20):
    """
    向图像添加高斯(Gaussian)噪声
    :param image: 输入图像(灰度图)
    :param mean: 高斯噪声均值
    :param sigma: 高斯噪声标准差
    :return: 添加噪声后图像(灰度)
    """
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + gauss
    # 裁剪至 [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

def add_rayleigh_noise(image, sigma=30):
    """
    向图像添加瑞利(Rayleigh)噪声
    :param image: 输入图像(灰度图)
    :param sigma: 瑞利分布参数
    :return: 添加噪声后图像(灰度)
    """
    # 瑞利分布逆变换采样
    rayleigh = sigma * np.sqrt(-2 * np.log(1 - np.random.rand(*image.shape)))
    noisy_image = image + rayleigh
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

def add_erlang_noise(image, k=2, lam=0.05):
    """
    向图像添加 Erlang/Gamma 噪声
    :param image: 输入图像(灰度图)
    :param k: Erlang/Gamma 形状参数(正整数)
    :param lam: 率参数
    :return: 添加噪声后图像(灰度)
    """
    erlang_noise = np.random.gamma(k, 1/lam, image.shape)
    noisy_image = image + erlang_noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

# ========== 新增的三类噪声函数 ==========

def add_exponential_noise(image, lam=0.05):
    """
    向图像添加指数(Exponential)噪声
    :param image: 输入图像(灰度图)
    :param lam: 指数分布参数 lambda
    :return: 添加噪声后图像(灰度)
    """
    # Exponential(λ) 的随机数，可由 np.random.exponential(scale=1/lam) 生成
    expo_noise = np.random.exponential(scale=1/lam, size=image.shape)
    noisy_image = image + expo_noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

def add_uniform_noise(image, a=-20, b=20):
    """
    向图像添加均匀(Uniform)噪声
    :param image: 输入图像(灰度图)
    :param a: 噪声下界
    :param b: 噪声上界
    :return: 添加噪声后图像(灰度)
    """
    # 生成 [a, b] 区间上的均匀分布随机数
    uni_noise = np.random.uniform(a, b, image.shape)
    noisy_image = image + uni_noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

def add_salt_pepper_noise(image, prob=0.05):
    """
    向图像添加脉冲(椒盐)噪声
    :param image: 输入图像(灰度图)
    :param prob: 椒盐噪声的总破坏概率，默认 5%
    :return: 添加噪声后图像(灰度)
    """
    noisy_image = image.copy()
    # 生成与图像同大小的随机数矩阵
    rand_matrix = np.random.rand(*image.shape)

    # 将概率中的一半视作盐(白)，另一半视作椒(黑)
    pepper_thres = prob / 2
    salt_thres   = prob

    # 小于 pepper_thres 的像素置 0 (黑)
    noisy_image[rand_matrix < pepper_thres] = 0
    # 介于 [pepper_thres, salt_thres) 的像素置 255 (白)
    noisy_image[(rand_matrix >= pepper_thres) & (rand_matrix < salt_thres)] = 255

    return noisy_image

# ========== 主流程 ==========

def main():
    # 1. 读取原图并转为灰度图
    img_path = 'pic/fire.png'
    original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if original is None:
        print("图片读取失败，请检查图片路径是否正确。")
        return

    # 2. 分别添加六种不同噪声
    gauss_noisy   = add_gaussian_noise(original, mean=0, sigma=20)
    rayleigh_noisy= add_rayleigh_noise(original, sigma=30)
    erlang_noisy  = add_erlang_noise(original, k=2, lam=0.05)
    expo_noisy    = add_exponential_noise(original, lam=0.05)
    uniform_noisy = add_uniform_noise(original, a=-20, b=20)
    sp_noisy      = add_salt_pepper_noise(original, prob=0.05)

    # 3. 统一进行并排绘制(原图 + 六种噪声)，共 7 幅图，各自附带直方图
    #    我们可以采用 7 行 x 2 列的子图布局：左边是图像，右边是直方图

    # 定义要展示的 (标题, 图像) 列表
    images_info = [
        ("Original", original),
        ("Gaussian Noise", gauss_noisy),
        ("Rayleigh Noise", rayleigh_noisy),
        ("Erlang Noise", erlang_noisy),
        ("Exponential Noise", expo_noisy),
        ("Uniform Noise", uniform_noisy),
        ("Salt & Pepper Noise", sp_noisy)
    ]

    plt.figure(figsize=(10, 20))  # 画布大小可根据需求调整

    for i, (title, img) in enumerate(images_info):
        # i*2 + 1 : 图像显示
        plt.subplot(len(images_info), 2, i*2 + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title + " Image")
        plt.axis("off")

        # i*2 + 2 : 直方图
        plt.subplot(len(images_info), 2, i*2 + 2)
        plt.hist(img.ravel(), bins=256, range=(0, 256))
        plt.title(title + " Histogram")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
