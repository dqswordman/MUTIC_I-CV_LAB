import numpy as np
from PIL import Image


def compute_grayscale_histogram(image_path, bins=256):
    """
    读取图像(可能是RGB也可能是灰度)，统一强制转为灰度(L)。
    然后统计其直方图(频数)，并归一化为[0,1]的概率分布。
    返回一个长度为bins的一维数组 hist。
    """
    # 1) 打开图像并强制转换为灰度(L)
    image = Image.open(image_path).convert('L')

    # 2) 转为 NumPy 数组
    data = np.array(image)

    # 3) 初始化直方图
    hist = np.zeros(bins, dtype=np.float64)

    # 4) 逐像素统计各灰度值(0~255)的出现次数
    for row in data:
        for pixel in row:
            hist[pixel] += 1

    # 5) 归一化 => 转为[0,1]区间的概率分布
    total_pixels = hist.sum()
    if total_pixels > 0:
        hist = hist / total_pixels

    return hist


def chi_squared_test(hist1, hist2):
    """
    依据课件中描述的Chi-Squared(卡方)检验伪代码，
    比较两个“已归一化”的直方图的差异度。

      chi_squared_value = 0
      对于每个bin i:
         expected = (hist1[i] + hist2[i]) / 2
         若 expected != 0:
             diff = hist1[i] - expected
             chi_squared_value += (diff * diff) / expected
    """
    chi_squared_value = 0.0

    for i in range(len(hist1)):
        expected = (hist1[i] + hist2[i]) / 2
        if expected != 0:
            diff = hist1[i] - expected
            chi_squared_value += (diff ** 2) / expected

    return chi_squared_value


if __name__ == "__main__":
    # ========== 使用示例 ==========

    # 第一张图 (已是灰度图, pic/grayscale.jpg)
    histA = compute_grayscale_histogram("pic/grayscale.jpg")

    # 第二张图 (假设原本是RGB, 这里也能强制转成灰度)
    histB = compute_grayscale_histogram("pic/grayscale2.jpg")

    # 进行卡方检验，比较两张图像(灰度化后)的分布差异
    chi_value = chi_squared_test(histA, histB)

    # 打印结果
    print(f"Chi-Squared value = {chi_value:.6f}")

    # 说明:
    # 值越小 => 两张图在灰度直方图分布上越相似
    # 值越大 => 二者差异越明显
