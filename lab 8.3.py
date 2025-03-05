import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener

# ========== 六种添加噪声的函数 (与前相同) ==========

def add_gaussian_noise(image, mean=0, sigma=20):
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + gauss
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def add_rayleigh_noise(image, sigma=30):
    rayleigh = sigma * np.sqrt(-2 * np.log(1 - np.random.rand(*image.shape)))
    noisy_image = image + rayleigh
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def add_erlang_noise(image, k=2, lam=0.05):
    erlang_noise = np.random.gamma(k, 1/lam, image.shape)
    noisy_image = image + erlang_noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def add_exponential_noise(image, lam=0.05):
    expo_noise = np.random.exponential(scale=1/lam, size=image.shape)
    noisy_image = image + expo_noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def add_uniform_noise(image, a=-20, b=20):
    uni_noise = np.random.uniform(a, b, image.shape)
    noisy_image = image + uni_noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(image, prob=0.05):
    noisy_image = image.copy()
    rand_matrix = np.random.rand(*image.shape)
    pepper_thres = prob / 2
    salt_thres   = prob
    noisy_image[rand_matrix < pepper_thres] = 0
    noisy_image[(rand_matrix >= pepper_thres) & (rand_matrix < salt_thres)] = 255
    return noisy_image

# ========== 六种去噪滤波示例函数 ==========

def mean_filter(noisy, ksize=3):
    """
    均值滤波: 利用cv2.blur()实现
    """
    return cv2.blur(noisy, (ksize, ksize))

def gaussian_filter(noisy, ksize=3, sigma=0):
    """
    高斯滤波: 利用cv2.GaussianBlur()实现
    sigma=0时，OpenCV会根据ksize自动计算
    """
    return cv2.GaussianBlur(noisy, (ksize, ksize), sigma)

def median_filter(noisy, ksize=3):
    """
    中值滤波: 利用cv2.medianBlur()实现
    """
    return cv2.medianBlur(noisy, ksize)

def bilateral_filter(noisy, d=5, sigmaColor=75, sigmaSpace=75):
    """
    双边滤波: 保留边缘的同时平滑噪声
    d表示滤波时领域直径
    sigmaColor、sigmaSpace控制空间高斯分布和像素值相似度
    """
    return cv2.bilateralFilter(noisy, d, sigmaColor, sigmaSpace)

def non_local_means_filter(noisy, h=10):
    """
    非局部均值滤波: 利用cv2.fastNlMeansDenoising()实现 (仅限单通道)
    h参数：滤波强度
    """
    # 该函数仅支持单通道图像，故此处需先转换为单通道再处理
    if len(noisy.shape) == 3:
        gray = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
    else:
        gray = noisy
    dst = cv2.fastNlMeansDenoising(gray, None, h, 7, 21)
    return dst

def wiener_filter(noisy, mysize=3):
    """
    维纳滤波: 利用scipy.signal.wiener()实现
    mysize为滤波窗口大小
    - 注意: wiener()需要浮点类型数组
    - 返回结果可能为浮点，需clip并转为uint8
    """
    # 转为float计算
    noisy_float = noisy.astype(np.float64)
    filtered = wiener(noisy_float, mysize)
    return np.clip(filtered, 0, 255).astype(np.uint8)

def main():
    # 1. 读取原图并转换为灰度
    img_path = 'pic/fire.png'
    original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        print("图片读取失败，请检查路径。")
        return

    # 2. 分别生成六种噪声图
    gauss_img   = add_gaussian_noise(original, mean=0, sigma=20)
    rayleigh_img= add_rayleigh_noise(original, sigma=30)
    erlang_img  = add_erlang_noise(original, k=2, lam=0.05)
    expo_img    = add_exponential_noise(original, lam=0.05)
    uniform_img = add_uniform_noise(original, a=-20, b=20)
    sp_img      = add_salt_pepper_noise(original, prob=0.05)

    noise_types = [
        ("Gaussian Noise", gauss_img),
        ("Rayleigh Noise", rayleigh_img),
        ("Erlang Noise", erlang_img),
        ("Exponential Noise", expo_img),
        ("Uniform Noise", uniform_img),
        ("Salt & Pepper Noise", sp_img)
    ]

    # 3. 随机选择一种噪声
    chosen_noise_title, chosen_noisy_img = noise_types[np.random.randint(0, len(noise_types))]

    # 4. 在不知道原图和噪声类型的前提下，分别用六种滤波去除噪声
    #    注意：为了统一对比，所有滤波器的参数可在此处进行适度调整
    denoised_mean    = mean_filter(chosen_noisy_img, ksize=3)
    denoised_gauss   = gaussian_filter(chosen_noisy_img, ksize=3, sigma=0)
    denoised_median  = median_filter(chosen_noisy_img, ksize=3)
    denoised_bilateral = bilateral_filter(chosen_noisy_img, d=5, sigmaColor=75, sigmaSpace=75)
    denoised_nlm     = non_local_means_filter(chosen_noisy_img, h=10)
    denoised_wiener  = wiener_filter(chosen_noisy_img, mysize=3)

    # 5. 展示结果对比：共8幅图
    #    1) 原图
    #    2) 选中噪声图
    #    3~8) 六种滤波去噪结果
    titles = [
        "Original (Ground Truth)",
        f"Chosen Noisy ({chosen_noise_title})",
        "Mean Filter",
        "Gaussian Filter",
        "Median Filter",
        "Bilateral Filter",
        "Non-Local Means",
        "Wiener Filter"
    ]

    results = [
        original,
        chosen_noisy_img,
        denoised_mean,
        denoised_gauss,
        denoised_median,
        denoised_bilateral,
        denoised_nlm,
        denoised_wiener
    ]

    plt.figure(figsize=(12, 8))
    for i in range(len(results)):
        plt.subplot(2, 4, i+1)
        plt.imshow(results[i], cmap='gray')
        plt.title(titles[i], fontsize=9)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
