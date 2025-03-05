import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============ 加噪函数示例(可自由替换其他噪声) ============
def add_gaussian_noise(image, mean=0, sigma=20):
    """ 高斯噪声 """
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(image, prob=0.05):
    """ 椒盐噪声 """
    noisy = image.copy()
    rand_matrix = np.random.rand(*image.shape)
    pepper_thres = prob / 2
    salt_thres   = prob
    # 小于pepper_thres的变0(黑), 大于pepper_thres且小于salt_thres的变255(白)
    noisy[rand_matrix < pepper_thres] = 0
    noisy[(rand_matrix >= pepper_thres) & (rand_matrix < salt_thres)] = 255
    return noisy

# ============ 八种滤波器的实现 ============

def arithmetic_mean_filter(image, ksize=3):
    """
    算术平均滤波
    窗口大小为 ksize x ksize
    """
    pad = ksize // 2
    padded = np.pad(image, pad, mode='reflect')
    h, w = image.shape
    filtered = np.zeros_like(image, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            roi = padded[i:i+ksize, j:j+ksize]
            filtered[i, j] = np.mean(roi)
    return np.clip(filtered, 0, 255).astype(np.uint8)

def geometric_mean_filter(image, ksize=3):
    """
    几何均值滤波
    g(x,y) = (Π f(s,t))^(1/(m*n))
    遇到0值会导致结果=0；为了数值稳定可以用对数运算
    """
    pad = ksize // 2
    padded = np.pad(image, pad, mode='reflect')
    h, w = image.shape
    filtered = np.zeros_like(image, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            roi = padded[i:i+ksize, j:j+ksize].astype(np.float32)
            # 为防止log(0)，可以+1e-5避免完全为0
            roi_log = np.log(roi + 1e-5)
            mean_log = np.mean(roi_log)
            gm = np.exp(mean_log)  # 几何均值
            filtered[i, j] = gm

    return np.clip(filtered, 0, 255).astype(np.uint8)

def harmonic_mean_filter(image, ksize=3):
    """
    谐波均值滤波
    g(x,y) = (mn) / Σ(1/f(s,t))
    对0值像素要格外小心
    """
    pad = ksize // 2
    padded = np.pad(image, pad, mode='reflect').astype(np.float32)
    h, w = image.shape
    filtered = np.zeros_like(image, dtype=np.float32)
    mn = ksize * ksize

    for i in range(h):
        for j in range(w):
            roi = padded[i:i+ksize, j:j+ksize]
            # 对roi中 0值 会导致分母无穷大; 用一个小epsilon避免除零
            roi = np.where(roi == 0, 1e-5, roi)
            filtered[i,j] = mn / np.sum(1.0 / roi)

    return np.clip(filtered, 0, 255).astype(np.uint8)

def contraharmonic_mean_filter(image, ksize=3, Q=1.5):
    """
    逆谐波均值滤波
    g(x,y) = Σ[f(s,t)^(Q+1)] / Σ[f(s,t)^Q]
    Q>0: 去除黑(椒)噪声  Q<0: 去除白(盐)噪声
    """
    pad = ksize // 2
    padded = np.pad(image, pad, mode='reflect').astype(np.float32)
    h, w = image.shape
    filtered = np.zeros_like(image, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            roi = padded[i:i+ksize, j:j+ksize]
            numerator   = np.sum(np.power(roi + 1e-5, Q+1))
            denominator = np.sum(np.power(roi + 1e-5, Q))
            filtered[i,j] = numerator / (denominator + 1e-5)

    return np.clip(filtered, 0, 255).astype(np.uint8)

def median_filter(image, ksize=3):
    """
    中值滤波 (也可用 cv2.medianBlur)
    """
    pad = ksize // 2
    padded = np.pad(image, pad, mode='reflect')
    h, w = image.shape
    filtered = np.zeros_like(image, dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            roi = padded[i:i+ksize, j:j+ksize]
            med = np.median(roi)
            filtered[i, j] = med

    return filtered

def max_filter(image, ksize=3):
    """
    最大滤波
    """
    pad = ksize // 2
    padded = np.pad(image, pad, mode='reflect')
    h, w = image.shape
    filtered = np.zeros_like(image, dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            roi = padded[i:i+ksize, j:j+ksize]
            filtered[i,j] = np.max(roi)

    return filtered

def min_filter(image, ksize=3):
    """
    最小滤波
    """
    pad = ksize // 2
    padded = np.pad(image, pad, mode='reflect')
    h, w = image.shape
    filtered = np.zeros_like(image, dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            roi = padded[i:i+ksize, j:j+ksize]
            filtered[i,j] = np.min(roi)

    return filtered

def midpoint_filter(image, ksize=3):
    """
    中点滤波: (max + min)/2
    """
    pad = ksize // 2
    padded = np.pad(image, pad, mode='reflect').astype(np.float32)
    h, w = image.shape
    filtered = np.zeros_like(image, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            roi = padded[i:i+ksize, j:j+ksize]
            filtered[i,j] = (np.max(roi) + np.min(roi)) / 2.0

    return np.clip(filtered, 0, 255).astype(np.uint8)

def alpha_trimmed_mean_filter(image, ksize=3, alpha=2):
    """
    α-截断均值滤波
    将窗口内排序后, 去掉前 alpha/2 个最小值 和 后 alpha/2 个最大值, 其余做平均
    alpha 必须小于 m*n
    """
    pad = ksize // 2
    padded = np.pad(image, pad, mode='reflect')
    h, w = image.shape
    filtered = np.zeros_like(image, dtype=np.float32)
    mn = ksize * ksize

    trim = int(alpha // 2)  # 每端剔除的数量
    if alpha >= mn:
        alpha = 0
        trim = 0

    for i in range(h):
        for j in range(w):
            roi = padded[i:i+ksize, j:j+ksize].flatten()
            roi_sorted = np.sort(roi)
            # 去掉alpha/2的最小与最大
            roi_used = roi_sorted[trim:mn-trim]
            filtered[i,j] = np.mean(roi_used)

    return np.clip(filtered, 0, 255).astype(np.uint8)

# ============ 自适应中值滤波器实现 ============
def adaptive_median_filter(image, max_ksize=7):
    """
    自适应中值滤波 (Adaptive Median Filter)
    :param image:   输入灰度图 (H x W)
    :param max_ksize:  最大允许的窗口大小(必须为奇数)
    :return:        滤波后的图像
    """
    # 确保最大窗口为奇数
    if max_ksize % 2 == 0:
        max_ksize += 1

    h, w = image.shape
    filtered = np.zeros_like(image, dtype=np.uint8)

    # 先做一个最大扩边，以便可变窗口时无需多次pad
    pad_size = max_ksize // 2
    padded = np.pad(image, pad_size, mode='reflect').astype(np.float32)

    for i in range(h):
        for j in range(w):
            # 记录中心像素在padded中的坐标
            center_i = i + pad_size
            center_j = j + pad_size

            # 初始窗口大小
            ksize = 3
            out_value = 0
            done = False

            while not done:
                half = ksize // 2
                roi = padded[center_i - half:center_i + half + 1,
                             center_j - half:center_j + half + 1]

                z_min = np.min(roi)
                z_max = np.max(roi)
                z_med = np.median(roi)

                A1 = z_med - z_min
                A2 = z_med - z_max

                if A1 > 0 and A2 < 0:
                    # z_med不是极值 -> 进入Stage B
                    z_xy = padded[center_i, center_j]
                    B1 = z_xy - z_min
                    B2 = z_xy - z_max

                    if B1 > 0 and B2 < 0:
                        # 当前像素也非极值 -> 保留原值
                        out_value = z_xy
                    else:
                        # 当前像素是极值 -> 用z_med替换
                        out_value = z_med
                    done = True
                else:
                    # z_med还是极值 -> 扩大窗口
                    ksize += 2
                    if ksize > max_ksize:
                        # 超过最大允许窗口 -> 用当前z_med
                        out_value = z_med
                        done = True

            filtered[i,j] = np.clip(out_value, 0, 255)

    return filtered.astype(np.uint8)

def main():
    # 1. 读取图像并转灰度
    img_path = "pic/fire.png"  # 需自行修改为本地图像路径
    original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        print("图片读取失败，请检查路径。")
        return

    # 2. 加噪 (这里示例使用椒盐噪声 prob=0.1)
    noisy_img = add_salt_pepper_noise(original, prob=0.1)

    # 3. 分别应用9种滤波 (含Adaptive Median)
    ksize = 3
    arithmetic_res      = arithmetic_mean_filter(noisy_img, ksize)
    geometric_res       = geometric_mean_filter(noisy_img, ksize)
    harmonic_res        = harmonic_mean_filter(noisy_img, ksize)
    contraharmonic_res  = contraharmonic_mean_filter(noisy_img, ksize, Q=-1.5)
    median_res          = median_filter(noisy_img, ksize)
    max_res             = max_filter(noisy_img, ksize)
    min_res             = min_filter(noisy_img, ksize)
    midpoint_res        = midpoint_filter(noisy_img, ksize)
    alpha_res           = alpha_trimmed_mean_filter(noisy_img, ksize, alpha=4)
    # 新增: 自适应中值滤波
    adapt_median_res    = adaptive_median_filter(noisy_img, max_ksize=7)

    # 4. 可视化对比
    titles = [
        "Original",
        "Noisy (Salt & Pepper)",
        "Arithmetic Mean",
        "Geometric Mean",
        "Harmonic Mean",
        "Contraharmonic Mean(Q=-1.5)",
        "Median Filter",
        "Max Filter",
        "Min Filter",
        "Midpoint Filter",
        "Alpha-Trimmed(alpha=4)",
        "Adaptive Median(max_ksize=7)"
    ]
    images = [
        original,
        noisy_img,
        arithmetic_res,
        geometric_res,
        harmonic_res,
        contraharmonic_res,
        median_res,
        max_res,
        min_res,
        midpoint_res,
        alpha_res,
        adapt_median_res
    ]

    plt.figure(figsize=(18,10))
    for i, (title, img) in enumerate(zip(titles, images)):
        plt.subplot(3, 4, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(title, fontsize=8)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
