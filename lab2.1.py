from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def create_histogram(image_path):
    """
    加载图像并计算其直方图。
    若为彩色(RGB)图像，返回三个数组(histRed, histGreen, histBlue)；
    若为灰度图像(L)或强制转换后的图像，返回一个数组(histGray)；
    若为其他格式且无法处理，则返回 None。
    """
    # 1. 打开图像
    image = Image.open(image_path)

    # 2. 若图像不是灰度(L)也不是RGB，则可根据需要进行转换或直接报错
    #   这里演示：如果不是RGB就转为灰度
    if image.mode not in ('RGB', 'L'):
        image = image.convert('L')

    # 3. 转换为numpy数组
    data = np.array(image)

    # 4. 根据数组形状和 image.mode 判断是彩色还是灰度
    if image.mode == 'RGB' and len(data.shape) == 3:
        # (height, width, 3) => 统计 RGB
        histogramRed = np.zeros(256, dtype=np.int64)
        histogramGreen = np.zeros(256, dtype=np.int64)
        histogramBlue = np.zeros(256, dtype=np.int64)
        # 遍历每个像素，统计RGB分量
        for row in data:
            for pixel in row:
                r, g, b = pixel[:3]
                histogramRed[r]   += 1
                histogramGreen[g] += 1
                histogramBlue[b]  += 1
        return histogramRed, histogramGreen, histogramBlue

    elif image.mode == 'L' and len(data.shape) == 2:
        # (height, width) => 灰度图
        histogramGray = np.zeros(256, dtype=np.int64)
        for row in data:
            for pixel in row:
                histogramGray[pixel] += 1
        return histogramGray

    else:
        print("Unsupported image format or dimension.")
        return None


def create_grayscale_histogram(image_path):
    """
    演示：无论原图是什么模式，都强制转为灰度(L)后统计直方图。
    """
    image = Image.open(image_path).convert('L')
    data = np.array(image)
    histogramGray = np.zeros(256, dtype=np.int64)
    for row in data:
        for pixel in row:
            histogramGray[pixel] += 1
    return histogramGray


def show_histogram(hist, title="Histogram", color="blue"):
    """
    将给定的一维直方图数组可视化 (长度为256)。
    """
    plt.bar(range(256), hist, color=color)
    plt.title(title)
    plt.xlabel("Intensity Value")
    plt.ylabel("Count")
    plt.show()


# ========== 使用示例 ==========

if __name__ == "__main__":

    # 1) 验证图像是否真的是灰度
    #    如果图像原本是RGB，但视觉上看似灰度，也会显示 "RGB"
    test_image = Image.open('pic/grayscale.jpg')
    print("测试图像模式:", test_image.mode)

    # 2) 使用 create_histogram() 函数处理“灰度图像”
    gray_hist = create_histogram('pic/grayscale.jpg')
    if gray_hist is not None and isinstance(gray_hist, np.ndarray):
        # 显示灰度图的直方图
        show_histogram(gray_hist, title="Grayscale Histogram", color="gray")

    # 3) 使用 create_histogram() 函数处理“彩色图像”
    rgb_hist = create_histogram('pic/color.jpg')
    if rgb_hist is not None and isinstance(rgb_hist, tuple):
        histRed, histGreen, histBlue = rgb_hist
        # 分别显示三个通道的直方图
        show_histogram(histRed,   title="Red Channel Histogram",   color="red")
        show_histogram(histGreen, title="Green Channel Histogram", color="green")
        show_histogram(histBlue,  title="Blue Channel Histogram",  color="blue")

    # 4) 演示“强制转灰度”的用法
    forced_gray_hist = create_grayscale_histogram('pic/color.jpg')
    show_histogram(forced_gray_hist, title="Forced Grayscale Histogram", color="gray")
