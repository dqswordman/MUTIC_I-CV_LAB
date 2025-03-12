import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


##############################################
# 1. RGB <-> HSI 转换函数
##############################################
def rgb_to_hsi(r, g, b):
    eps = 1e-8
    i = (r + g + b) / 3.0
    min_val = min(r, g, b)
    if i > 0:
        s = 1 - (min_val / (i + eps))
    else:
        s = 0

    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + eps
    theta = np.arccos(num / den)

    if b <= g:
        h = theta / (2 * np.pi)
    else:
        h = (2 * np.pi - theta) / (2 * np.pi)
    return h, s, i


def hsi_to_rgb(h, s, i):
    eps = 1e-8
    H = 2 * np.pi * h
    if H < 2 * np.pi / 3:
        b = i * (1 - s)
        r = i * (1 + s * np.cos(H) / (np.cos(np.pi / 3 - H) + eps))
        g = 3 * i - (r + b)
    elif H < 4 * np.pi / 3:
        H -= 2 * np.pi / 3
        r = i * (1 - s)
        g = i * (1 + s * np.cos(H) / (np.cos(np.pi / 3 - H) + eps))
        b = 3 * i - (r + g)
    else:
        H -= 4 * np.pi / 3
        g = i * (1 - s)
        b = i * (1 + s * np.cos(H) / (np.cos(np.pi / 3 - H) + eps))
        r = 3 * i - (g + b)
    # 裁剪到 [0,1]
    r = np.clip(r, 0, 1)
    g = np.clip(g, 0, 1)
    b = np.clip(b, 0, 1)
    return r, g, b


##############################################
# 2. 对 I 通道进行直方图均衡
##############################################
def histogram_equalization_intensity(I_channel):
    """
    对 2D 的 I 通道进行直方图均衡 (范围 [0,1])，
    返回均衡后的 I 通道，形状相同，仍在 [0,1]。
    """
    flat_I = I_channel.flatten()

    # 统计直方图
    bins = 256
    hist, bin_edges = np.histogram(flat_I, bins=bins, range=(0, 1))
    cdf = np.cumsum(hist).astype(np.float32)
    cdf_normalized = cdf / cdf[-1]  # 归一化到 [0,1]

    # 用插值实现旧亮度到新亮度的映射
    equalized_flat = np.interp(flat_I, bin_edges[:-1], cdf_normalized)
    equalized_I = equalized_flat.reshape(I_channel.shape)
    return equalized_I


##############################################
# 3. 主函数
##############################################
def main():
    # 3.1 读取图像
    img_path = 'pic/a.jpg'  # 修改为你的实际图片路径
    img_pil = Image.open(img_path).convert('RGB')
    img_arr = np.array(img_pil, dtype=np.float32) / 255.0  # (H,W,3)

    H, W, _ = img_arr.shape

    # 3.2 RGB -> HSI
    hsi_arr = np.zeros_like(img_arr)
    for row in range(H):
        for col in range(W):
            r, g, b = img_arr[row, col]
            h, s, i = rgb_to_hsi(r, g, b)
            hsi_arr[row, col] = [h, s, i]

    # 3.3 对 I 通道做直方图均衡
    I_channel = hsi_arr[:, :, 2]
    I_eq = histogram_equalization_intensity(I_channel)
    hsi_arr[:, :, 2] = I_eq

    # 3.4 可选：对 S 通道稍微增减
    saturation_factor = 1.10  # 比如增加 10%
    s_channel = hsi_arr[:, :, 1]
    s_channel_adj = np.clip(s_channel * saturation_factor, 0, 1)
    hsi_arr[:, :, 1] = s_channel_adj

    # 3.5 HSI -> RGB
    result_arr = np.zeros_like(img_arr)
    for row in range(H):
        for col in range(W):
            h, s, i = hsi_arr[row, col]
            r_new, g_new, b_new = hsi_to_rgb(h, s, i)
            result_arr[row, col] = [r_new, g_new, b_new]

    # 4. 准备可视化：原图/结果 + 亮度直方图/红通道直方图
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    # 布局：
    # (0,0): 原图
    # (0,1): 处理后图
    # (0,2): 原图 I 通道直方图
    # (1,0): 处理后 I 通道直方图
    # (1,1): 原图 R 通道直方图
    # (1,2): 处理后 R 通道直方图

    # 4.1 显示原图
    axes[0, 0].imshow(img_arr)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # 4.2 显示处理后图
    axes[0, 1].imshow(result_arr)
    axes[0, 1].set_title("Processed (HSI HistEq + Saturation)")
    axes[0, 1].axis("off")

    # 4.3 原图 I 通道直方图
    original_I = (img_arr[:, :, 0] + img_arr[:, :, 1] + img_arr[:, :, 2]) / 3.0
    axes[0, 2].hist(original_I.flatten(), bins=256, range=(0, 1), color='gray')
    axes[0, 2].set_title("Original Intensity Hist")
    axes[0, 2].set_xlabel("Intensity")
    axes[0, 2].set_ylabel("Count")

    # 4.4 处理后 I 通道直方图
    processed_I = (result_arr[:, :, 0] + result_arr[:, :, 1] + result_arr[:, :, 2]) / 3.0
    axes[1, 0].hist(processed_I.flatten(), bins=256, range=(0, 1), color='gray')
    axes[1, 0].set_title("Processed Intensity Hist")
    axes[1, 0].set_xlabel("Intensity")
    axes[1, 0].set_ylabel("Count")

    # 4.5 原图 R 通道直方图
    original_R = img_arr[:, :, 0].flatten()
    axes[1, 1].hist(original_R, bins=256, range=(0, 1), color='red')
    axes[1, 1].set_title("Original R Channel Hist")
    axes[1, 1].set_xlabel("R Value")
    axes[1, 1].set_ylabel("Count")

    # 4.6 处理后 R 通道直方图
    processed_R = result_arr[:, :, 0].flatten()
    axes[1, 2].hist(processed_R, bins=256, range=(0, 1), color='red')
    axes[1, 2].set_title("Processed R Channel Hist")
    axes[1, 2].set_xlabel("R Value")
    axes[1, 2].set_ylabel("Count")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
