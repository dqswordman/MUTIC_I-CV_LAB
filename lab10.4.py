import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. 读取灰度图像
    image_path = "pic/pic_lab10_4.png"  # <-- 请替换为你的实际图像路径
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise ValueError("图像读取失败，请检查文件路径是否正确。")

    # 2. 将灰度图以128为阈值进行二值化：<128=0, >=128=255
    threshold_value = 128
    ret, binary_default = cv2.threshold(original, threshold_value, 255, cv2.THRESH_BINARY)

    # ========== 2.1 选择前景是白色还是黑色 ==========
    # 如果你想让黑色为前景，可以对二值图进行取反:
    #   black_fg = cv2.bitwise_not(binary_default)
    #   binary_fg = black_fg
    # 这里演示“白色(255)为前景”，直接使用 binary_default:
    binary_fg = binary_default

    # 3. 构建 11x11 全1结构元素
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    # 4. 基础形态学操作：腐蚀、膨胀
    eroded = cv2.erode(binary_fg, kernel, iterations=1)        # 腐蚀
    dilated = cv2.dilate(binary_fg, kernel, iterations=1)      # 膨胀

    # 5. 开运算(Opening)与闭运算(Closing)
    opened = cv2.morphologyEx(binary_fg, cv2.MORPH_OPEN, kernel)   # 开运算: 先腐蚀再膨胀
    closed = cv2.morphologyEx(binary_fg, cv2.MORPH_CLOSE, kernel)  # 闭运算: 先膨胀再腐蚀

    # 6. 先开后闭(Opening→Closing) & 先闭后开(Closing→Opening)
    opened_then_closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    closed_then_opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # 7. 击中-击不中(Hit-or-Miss)
    #    注意：OpenCV 的 Hit-or-Miss 操作需要图像为 0/1 二值，而不是 0/255。
    #    因此先将 binary_fg 转为0/1，再进行操作。kernel 中 1 表示“必须为1”，-1 表示“必须为0”，0 表示“不关心”。
    #    这里给一个简易示例 kernel，仅作演示:
    hitmiss_kernel = np.array([
        [ 0,  1,  0],
        [ 1,  1,  1],
        [ 0,  1,  0]
    ], dtype=np.int8)

    #    先将(0,255)的图像转换为(0,1)
    binary_fg_01 = (binary_fg // 255).astype(np.uint8)
    #    进行击中-击不中运算
    hit_miss_01 = cv2.morphologyEx(binary_fg_01, cv2.MORPH_HITMISS, hitmiss_kernel)
    #    若想便于显示，再转换回(0,255)
    hit_miss = (hit_miss_01 * 255).astype(np.uint8)

    # 8. 可视化对比：总共9张图
    #    准备 3x3 网格来显示 (row=3, col=3)
    plt.figure(figsize=(15, 10))

    # 1) 原图
    plt.subplot(3, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title("1. Original")
    plt.axis('off')

    # 2) 二值化
    plt.subplot(3, 3, 2)
    plt.imshow(binary_fg, cmap='gray')
    plt.title("2. Binary")
    plt.axis('off')

    # 3) 腐蚀
    plt.subplot(3, 3, 3)
    plt.imshow(eroded, cmap='gray')
    plt.title("3. Eroded (ERODED)")
    plt.axis('off')

    # 4) 膨胀
    plt.subplot(3, 3, 4)
    plt.imshow(dilated, cmap='gray')
    plt.title("4. Dilated (DILATED)")
    plt.axis('off')

    # 5) 开运算
    plt.subplot(3, 3, 5)
    plt.imshow(opened, cmap='gray')
    plt.title("5. Opening (OPENING)")
    plt.axis('off')

    # 6) 闭运算
    plt.subplot(3, 3, 6)
    plt.imshow(closed, cmap='gray')
    plt.title("6. Closing (CLOSING)")
    plt.axis('off')

    # 7) 先开后闭
    plt.subplot(3, 3, 7)
    plt.imshow(opened_then_closed, cmap='gray')
    plt.title("7. Opening → Closing")
    plt.axis('off')

    # 8) 先闭后开
    plt.subplot(3, 3, 8)
    plt.imshow(closed_then_opened, cmap='gray')
    plt.title("8. Closing → Opening")
    plt.axis('off')

    # 9) 击中-击不中
    plt.subplot(3, 3, 9)
    plt.imshow(hit_miss, cmap='gray')
    plt.title("9. Hit-or-Miss (HIT-OR-MISS)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
