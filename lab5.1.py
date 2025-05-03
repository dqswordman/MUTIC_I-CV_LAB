import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_image_and_hist(img, title="Image"):
    """
    显示图像和对应直方图的辅助函数
    """
    plt.figure(figsize=(10, 4))

    # 显示图像
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

    # 显示直方图
    plt.subplot(1, 2, 2)
    # 对于灰度图来说，range=[0,256], bins=256
    plt.hist(img.ravel(), bins=256, range=[0, 256], color='black')
    plt.title(f"{title} Histogram")
    plt.tight_layout()
    plt.show()


def histogram_equalization(img):
    """
    使用 OpenCV 提供的直方图均衡化函数
    返回均衡化后的图像
    """
    # OpenCV 的直方图均衡化只支持单通道图像
    # cv2.equalizeHist: 只支持 uint8 类型的单通道图像
    eq_img = cv2.equalizeHist(img)
    return eq_img


def histogram_specification(source_img, target_ranges):
    """
    直方图规定化（Histogram Specification）的简化示例。
    source_img: 原始图像（灰度）
    target_ranges: 例如 [(0,50), (50,200), (200,255)]，
                   代表我们想把像素值重新分配到这些区间范围内，
                   以实现特定风格或亮度分布的规定化。

    返回规定化后的图像

    说明：
    - 该函数是一个“示例”实现，用于演示手动指定想要的分布范围。
    - 实际使用中，常见做法是读取另一个“参考图像(reference image)”的分布并做匹配，
      或者自定义一个详细的直方图(或CDF)再进行“匹配”。
    - 这里用一个相对简单的方式，假设目标分布只是将像素值按某些区间重新拉伸到指定区间。
    """
    # 先计算原图的最小和最大像素值，用于后续归一化
    src_min = np.min(source_img)
    src_max = np.max(source_img)

    # 将原图像素值归一化到 [0,1]
    src_norm = (source_img - src_min) / (src_max - src_min + 1e-8)

    # target_ranges 是一个列表，每个元素是 (start, end)，表示一个区间
    # 我们希望把0-1之间的数值，重新分段映射到这些指定区间
    # 假设 target_ranges 中所有区间连续，且总长度为 [0,1] 的映射
    # 为了简化，这里做一个“分段均匀分配”的示例：
    #   - 将 [0,1] 按数量等分拆分给 target_ranges
    #   - 例如有3个区间，那么 0~1 就三等分: [0,1/3], [1/3,2/3], [2/3,1]
    #   - 分别把对应范围内的像素，拉伸到 (start,end)

    # 计算每段的长度
    total_segments = len(target_ranges)
    seg_length = 1.0 / total_segments  # 每段在 [0,1] 中所占区间

    # 准备一个输出图像(浮点形式)，最后再转成uint8
    specified_img = np.zeros_like(src_norm, dtype=np.float32)

    for i, (t_start, t_end) in enumerate(target_ranges):
        # [low_bound, high_bound) 在 [0,1] 上对应哪一段
        low_bound = i * seg_length
        high_bound = (i + 1) * seg_length
        # 找到 src_norm 中属于这一段的mask
        mask = (src_norm >= low_bound) & (src_norm < high_bound)
        # 在该 mask 下，将像素值映射到 [t_start, t_end]
        # 首先对分段内的值做0~1归一化: (x - low_bound)/(high_bound - low_bound)
        # 然后映射到 [t_start, t_end]
        specified_img[mask] = ((src_norm[mask] - low_bound) / (high_bound - low_bound + 1e-8)) \
                              * (t_end - t_start) + t_start

    # 避免因分段边界影响，最后单独处理 ==1 的情况
    specified_img[src_norm >= 1.0] = target_ranges[-1][1]

    # 转回 [0,255] 范围的 uint8
    specified_img_uint8 = np.clip(specified_img, 0, 255).astype(np.uint8)
    return specified_img_uint8


def main():
    # 1. 读取灰度图像
    # 注意：cv2.imread 默认读取三通道BGR，要想读取灰度图必须加 flag = 0
    source_img = cv2.imread("pic/test-image3.png", 0)
    if source_img is None:
        print("错误：无法读取图像，请检查路径。")
        return

    # 显示原图与其直方图
    show_image_and_hist(source_img, title="Original Grayscale Image")

    # 2. 进行直方图均衡化
    eq_img = histogram_equalization(source_img)
    show_image_and_hist(eq_img, title="Equalized Image")

    # 3. 进行直方图规定化
    # 这里提供一个简化示例：假设我们想将像素分为三段，
    #   0 ~ (1/3) -> [0,80]
    #   (1/3)~(2/3)-> [80,180]
    #   (2/3)~(1)  -> [180,255]
    # 这只是演示，你可以根据需要自定义更多区间或更细致的映射
    target_ranges = [(0, 80), (80, 180), (180, 255)]
    spec_img = histogram_specification(source_img, target_ranges)
    show_image_and_hist(spec_img, title="Specified Image")


if __name__ == "__main__":
    main()
