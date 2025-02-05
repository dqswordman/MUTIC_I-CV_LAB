

"""
示例：在同一张灰度图像上进行以下点处理图像增强方法：
1. 负片变换
2. 阈值化（多阈值对比）
3. 对数变换（多系数对比）
4. 幂律变换/伽马校正（多gamma值对比）
5. 分段线性变换（举例两种映射方式）
6. 比特平面切割（展示8个位平面）

请先准备一张名为 'pic/a.jpg' 的图像，保证其是灰度或可被读取为灰度。
运行后，将使用 matplotlib 分别可视化每一种方法的多组结果。
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取灰度图像
img = cv2.imread('pic/a.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("找不到图像文件 'pic/a.jpg'，请检查路径。")


###############################################################################
#                            1. 负片变换
###############################################################################
def negative_transform(image):
    # 假设图像像素范围为[0, 255]
    # 负片 s = 255 - r
    return 255 - image


###############################################################################
#                            2. 阈值化（多阈值演示）
###############################################################################
def threshold_transform(image, thresh_value):
    """
    二值化：若灰度>阈值，则为255，否则为0
    """
    # 注意：OpenCV里的阈值化也可用 cv2.threshold，但这里演示手动实现
    binary_img = np.zeros_like(image)
    binary_img[image > thresh_value] = 255
    return binary_img


###############################################################################
#                            3. 对数变换（多系数演示）
###############################################################################
def log_transform(image, c=1.0):
    """
    s = c * log(1 + r)
    需先将图像像素归一化到[0,1]，再做对数映射，最后映射回[0,255]
    """
    # 转为float并归一化
    img_float = image.astype(np.float32) / 255.0
    # 对数变换
    log_img = c * np.log1p(img_float)  # log1p(x) = log(1+x)
    # 归一化到[0,1]再映射回[0,255]
    log_img_norm = cv2.normalize(log_img, None, alpha=0, beta=1.0,
                                 norm_type=cv2.NORM_MINMAX)
    out = (log_img_norm * 255).astype(np.uint8)
    return out


###############################################################################
#                            4. 幂律变换（Gamma校正，多gamma值演示）
###############################################################################
def gamma_transform(image, gamma=1.0):
    """
    s = r^gamma （若r先归一化到[0,1]）
    """
    # 转为float并归一化
    img_float = image.astype(np.float32) / 255.0
    # 幂运算
    gamma_img = np.power(img_float, gamma)
    # 映射回[0,255]
    out = (gamma_img * 255).astype(np.uint8)
    return out


###############################################################################
#                    5. 分段线性变换（举例：灰度切片 和 对比度拉伸）
###############################################################################
def piecewise_linear_slice(image, lower, upper, high_val=255):
    """
    简单灰度切片示例：
    - 将 [lower, upper] 范围内的像素抬升为 high_val
    - 其余像素保持原状(也可设为0或其它)
    """
    out = image.copy()
    # 在切片范围内的像素设为 high_val
    mask = (image >= lower) & (image <= upper)
    out[mask] = high_val
    return out


def piecewise_linear_stretch(image, r1, s1, r2, s2):
    """
    分段对比度拉伸示例：
    - 0~r1 映射到 0~s1（线性）
    - r1~r2 映射到 s1~s2（线性）
    - r2~255 映射到 s2~255（线性）
    """
    out = np.zeros_like(image, dtype=np.float32)
    # 三段斜率
    # 避免分母为0，做float运算
    r1, s1, r2, s2 = float(r1), float(s1), float(r2), float(s2)
    # 第一段斜率
    a1 = s1 / (r1 + 1e-5)  # +1e-5避免除0
    # 第二段斜率
    a2 = (s2 - s1) / ((r2 + 1e-5) - r1)
    # 第三段斜率
    a3 = (255.0 - s2) / (255.0 - r2 + 1e-5)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            r = image[y, x]
            if r < r1:
                out[y, x] = a1 * r
            elif r <= r2:
                out[y, x] = s1 + a2 * (r - r1)
            else:
                out[y, x] = s2 + a3 * (r - r2)

    # 转回uint8
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


###############################################################################
#                           6. 比特平面切割
###############################################################################
def bitplane_slicing(image):
    """
    返回一个包含8个bit平面的列表 [plane0, plane1, ..., plane7]
    plane0 对应最低位(LSB)，plane7 对应最高位(MSB)
    """
    planes = []
    for i in range(8):
        # 提取第 i 位 (0=LSB, 7=MSB)
        # (image >> i) & 1 得到第 i 位的二值图
        plane = ((image >> i) & 1) * 255
        planes.append(plane.astype(np.uint8))
    return planes


###############################################################################
#                           主流程：可视化多组结果
###############################################################################
def main():
    original = img
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1), plt.imshow(original, cmap='gray')
    plt.title("Original"), plt.axis('off')

    # ------------------- 1. 负片变换 --------------------
    neg_img = negative_transform(original)
    plt.subplot(1, 2, 2), plt.imshow(neg_img, cmap='gray')
    plt.title("Negative Image"), plt.axis('off')
    plt.tight_layout()
    plt.show()

    # ------------------- 2. 阈值化（多阈值） --------------
    thresholds = [50, 128, 200]  # 三个代表性阈值
    plt.figure(figsize=(12, 4))
    for i, th in enumerate(thresholds):
        bin_img = threshold_transform(original, th)
        plt.subplot(1, len(thresholds), i + 1)
        plt.imshow(bin_img, cmap='gray')
        plt.title(f"Threshold = {th}")
        plt.axis('off')
    plt.suptitle("Thresholding Comparison", fontsize=14)
    plt.tight_layout()
    plt.show()

    # ------------------- 3. 对数变换（多系数） --------------
    c_values = [1.0, 2.0, 5.0]  # 不同 c 做对比
    plt.figure(figsize=(12, 4))
    for i, c in enumerate(c_values):
        log_img = log_transform(original, c=c)
        plt.subplot(1, len(c_values), i + 1)
        plt.imshow(log_img, cmap='gray')
        plt.title(f"Log Transform (c={c})")
        plt.axis('off')
    plt.suptitle("Log Transform Comparison", fontsize=14)
    plt.tight_layout()
    plt.show()

    # ------------------- 4. 幂律变换（Gamma变换，多gamma） ----
    gamma_values = [0.4, 0.7, 1.5, 2.0]
    plt.figure(figsize=(12, 6))
    for i, gm in enumerate(gamma_values):
        gm_img = gamma_transform(original, gm)
        plt.subplot(2, 2, i + 1)
        plt.imshow(gm_img, cmap='gray')
        plt.title(f"Gamma={gm}")
        plt.axis('off')
    plt.suptitle("Gamma Correction Comparison", fontsize=14)
    plt.tight_layout()
    plt.show()

    # ------------------- 5. 分段线性变换示例 ---------------
    # 5.1 灰度切片
    slice_img = piecewise_linear_slice(original, lower=100, upper=150, high_val=255)
    # 5.2 简易三段对比度拉伸
    stretch_img = piecewise_linear_stretch(original, r1=50, s1=20, r2=200, s2=230)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1), plt.imshow(original, cmap='gray')
    plt.title("Original"), plt.axis('off')
    plt.subplot(1, 3, 2), plt.imshow(slice_img, cmap='gray')
    plt.title("Piecewise Slice [100,150]->255"), plt.axis('off')
    plt.subplot(1, 3, 3), plt.imshow(stretch_img, cmap='gray')
    plt.title("Piecewise Stretch"), plt.axis('off')
    plt.tight_layout()
    plt.show()

    # ------------------- 6. 比特平面切割 --------------
    planes = bitplane_slicing(original)
    # plane0=LSB, plane7=MSB
    plt.figure(figsize=(12, 6))
    for i, plane in enumerate(planes[::-1]):  # 逆序显示，让最高位(7)排在前
        plt.subplot(2, 4, i + 1)
        plt.imshow(plane, cmap='gray')
        plt.title(f"Bit Plane {7 - i}")
        plt.axis('off')
    plt.suptitle("Bit Plane Slicing (bit7 ~ bit0)", fontsize=14)
    plt.tight_layout()
    plt.show()

    # 额外：示例如何只用高位重构图像
    # 这里取高位 bit7 和 bit6:
    bit7 = planes[7] // 255  # 0或1
    bit6 = planes[6] // 255
    # 只重构bit7+bit6 => 2^7 * bit7 + 2^6 * bit6
    recon_76 = (bit7 << 7) + (bit6 << 6)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1), plt.imshow(original, cmap='gray')
    plt.title("Original"), plt.axis('off')
    plt.subplot(1, 2, 2), plt.imshow(recon_76, cmap='gray')
    plt.title("Reconstructed from bit7 & bit6"), plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
