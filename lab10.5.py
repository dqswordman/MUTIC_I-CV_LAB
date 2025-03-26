#question 2 failed
import numpy as np
import cv2
import matplotlib.pyplot as plt

def main():
    """
    本示例演示如何使用纯代码生成一个示意图（相当于书中 Fig 9.12 的 A = C ∪ D ∪ E），
    然后定义两个结构元素 B1、B2，对 A 和 A^c 分别做腐蚀，再取交集实现“击中-击不中”运算。
    """

    # -----------------------------
    # 1. 用代码生成图像 A（背景0，前景255）
    # -----------------------------
    # 创建一幅黑色画布（高度300, 宽度400，可自行调节）
    height, width = 300, 400
    A = np.zeros((height, width), dtype=np.uint8)

    # 在画布上画三个白色矩形，模拟集合 C, D, E 的并集
    # (1) C: 左上方大矩形
    cv2.rectangle(A, (30, 30), (130, 130), 255, -1)  # 左上 (x1,y1)=(30,30), 右下=(130,130)

    # (2) D: 中心矩形
    cv2.rectangle(A, (150, 90), (250, 190), 255, -1) # (150,90)->(250,190)

    # (3) E: 右上小矩形
    cv2.rectangle(A, (290, 20), (350, 80), 255, -1)

    # 你也可换成画圆/多边形等，使其更像原图示意

    # -----------------------------
    # 2. 得到 A 的补集 A^c
    # -----------------------------
    A_c = cv2.bitwise_not(A)  # 0<->255 取反

    # -----------------------------
    # 3. 定义结构元素 B1, B2
    # -----------------------------
    # 思路：希望 B1 能“匹配” D 的形状，所以可做一个与 D 大小类似的结构元素
    # 书中示意 B1, B2 不同之处在于 B2 还包含“必须是背景”的位置
    # 这里简单起见，用两个 5x5 不同模式的示例
    B1 = np.ones((5, 5), dtype=np.uint8)  # 全1，可根据需要修改
    # B2 用一个自定义的小十字形(仅演示)
    B2 = np.array([
        [0, 1, 0, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0]
    ], dtype=np.uint8)

    #   你也可以在 B2 中设置 -1 表示“不关心”等，不过 OpenCV 的 erosions 不支持 -1，
    #   若想实现书中“必须是背景/必须是前景”可用 MORPH_HITMISS 或自定义逻辑。
    #   此处主要演示腐蚀+交集的思路。

    # -----------------------------
    # 4. 对 A 做腐蚀 (A⊖B1)，对 A^c 做腐蚀 (A^c⊖B2)
    # -----------------------------
    #   OpenCV 腐蚀默认把 "非零" 视作前景，对 255 值像素腐蚀 => 结果依然是 255/0 二值图
    eroded_A_B1 = cv2.erode(A, B1)
    eroded_Ac_B2 = cv2.erode(A_c, B2)

    # -----------------------------
    # 5. 交集(∩) => 击中-击不中结果
    # -----------------------------
    #   在二值图上，“交集”可用 bitwise_and
    #   eroded_A_B1 与 eroded_Ac_B2 都是 0/255 的图，交集即 => 同时为255的地方，输出255；否则0
    hit_miss_result = cv2.bitwise_and(eroded_A_B1, eroded_Ac_B2)

    # -----------------------------
    # 6. 可视化
    # -----------------------------
    #   这里一次性显示6张：
    #   (1) A   (2) A^c   (3) A⊖B1   (4) A^c⊖B2   (5) Intersection   (6) 结构元素示意
    plt.figure(figsize=(12, 8))

    # (1) A
    plt.subplot(2, 3, 1)
    plt.imshow(A, cmap='gray', vmin=0, vmax=255)
    plt.title("A = C ∪ D ∪ E")
    plt.axis('off')

    # (2) A^c
    plt.subplot(2, 3, 2)
    plt.imshow(A_c, cmap='gray', vmin=0, vmax=255)
    plt.title("A^c (complement)")
    plt.axis('off')

    # (3) A⊖B1
    plt.subplot(2, 3, 3)
    plt.imshow(eroded_A_B1, cmap='gray', vmin=0, vmax=255)
    plt.title("A ⊖ B1 (eroded)")
    plt.axis('off')

    # (4) A^c⊖B2
    plt.subplot(2, 3, 4)
    plt.imshow(eroded_Ac_B2, cmap='gray', vmin=0, vmax=255)
    plt.title("A^c ⊖ B2 (eroded)")
    plt.axis('off')

    # (5) 交集 => 击中-击不中结果
    plt.subplot(2, 3, 5)
    plt.imshow(hit_miss_result, cmap='gray', vmin=0, vmax=255)
    plt.title("(A ⊖ B1) ∩ (A^c ⊖ B2)\nHit-or-Miss result")
    plt.axis('off')

    # (6) 显示一下 B1、B2
    # 为方便观察，这里把 0/1 -> 0/255
    # 显示 B1
    display_B1 = (B1 * 255).astype(np.uint8)
    # 拼接 B1, B2 并一起可视化
    display_B2 = (B2 * 255).astype(np.uint8)
    # 先把它们拼在上下
    Bstack = cv2.vconcat([display_B1, display_B2])

    plt.subplot(2, 3, 6)
    plt.imshow(Bstack, cmap='gray', vmin=0, vmax=255)
    plt.title("SE B1(up) and B2(down)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
