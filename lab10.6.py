#upgrade by 10.5
#question 2 failed to run
import numpy as np
import cv2
import matplotlib.pyplot as plt

def main():
    """
    通过纯代码生成 A=C∪D∪E (三个矩形)，
    再定义 B1, B2，使得击中-击不中 (A ⊖ B1) ∩ (A^c ⊖ B2)
    最终只在 D 的中心像素处保留前景(白)。
    """

    # -----------------------------
    # (A) 用黑色画布 + 3个矩形 生成 A = C∪D∪E
    # -----------------------------
    H, W = 300, 300
    A = np.zeros((H, W), dtype=np.uint8)

    # 1) C：左上方矩形
    cv2.rectangle(A, (30, 30), (80, 80), 255, -1)   # 50×50

    # 2) D：我们要“击中”的目标，做成 21×21
    #    这样在后面能设计一个 21×21 的 SE (B1) 完整覆盖 D
    #    让其中心保留下来
    #    比如：D放在(100,70)->(121,91)
    cv2.rectangle(A, (100, 70), (121, 91), 255, -1)  # 21×21

    # 3) E：右上小矩形
    cv2.rectangle(A, (180, 40), (210, 70), 255, -1)  # 30×30

    # A 即三个对象 C, D, E 的并集 (二值图: 0/255)

    # (B) A^c 补集 (翻转0<->255)
    A_c = cv2.bitwise_not(A)

    # -----------------------------
    # (C) 设计 B1, B2
    # -----------------------------

    # 思路：希望 (A ⊖ B1) 只在 D 的中心点留下 1。
    # 1) B1 : 尺寸与 D 一样 21×21，全1(必须是前景)。
    #    这样对 A 做腐蚀时，只有能完全容纳 21×21 白色的地方，中心点才保留 255。
    #    D正好是21×21，因此腐蚀后只有D的最“中间像素”能保留。
    B1 = np.ones((21, 21), dtype=np.uint8)

    # 2) B2 : 要求排除其它对象 (C, E) 的干扰。
    #    思路：对 A^c 做腐蚀 (A^c⊖B2)，让只有“D周围形状”才能通过。
    #    可以设计一个略比D大一些的 kernel，全1。
    #    在 A^c 中，这意味着D周围那一圈必须是背景(对A来说)。若C或E紧挨D，则会破坏此匹配。
    #    这里示例做成 25×25 的全1，确保D周边4个像素也必须是背景。
    B2 = np.ones((25, 25), dtype=np.uint8)

    # -----------------------------
    # (D) 分别对 A, A^c 做腐蚀，再取交集 => Hit-or-Miss
    # -----------------------------
    eroded_A_B1 = cv2.erode(A, B1)       # A ⊖ B1
    eroded_Ac_B2 = cv2.erode(A_c, B2)    # A^c ⊖ B2

    hit_miss_result = cv2.bitwise_and(eroded_A_B1, eroded_Ac_B2)
    # 在理想情况下, 只在 D 的中心像素保留255

    # -----------------------------
    # (E) 可视化结果
    # -----------------------------
    plt.figure(figsize=(12, 8))

    # 1) A = C∪D∪E
    plt.subplot(2, 3, 1)
    plt.imshow(A, cmap='gray', vmin=0, vmax=255)
    plt.title("A = C ∪ D ∪ E")
    plt.axis('off')

    # 2) A^c
    plt.subplot(2, 3, 2)
    plt.imshow(A_c, cmap='gray', vmin=0, vmax=255)
    plt.title("A^c (complement)")
    plt.axis('off')

    # 3) A ⊖ B1
    plt.subplot(2, 3, 3)
    plt.imshow(eroded_A_B1, cmap='gray', vmin=0, vmax=255)
    plt.title("A ⊖ B1 (eroded)")
    plt.axis('off')

    # 4) A^c ⊖ B2
    plt.subplot(2, 3, 4)
    plt.imshow(eroded_Ac_B2, cmap='gray', vmin=0, vmax=255)
    plt.title("A^c ⊖ B2 (eroded)")
    plt.axis('off')

    # 5) (A⊖B1) ∩ (A^c⊖B2) => Hit-or-Miss
    plt.subplot(2, 3, 5)
    plt.imshow(hit_miss_result, cmap='gray', vmin=0, vmax=255)
    plt.title("(A⊖B1) ∩ (A^c⊖B2)\nHit-or-Miss result")
    plt.axis('off')

    # 6) 结构元素 B1(上) 与 B2(下)
    disp_B1 = (B1 * 255).astype(np.uint8)
    disp_B2 = (B2 * 255).astype(np.uint8)
    stacked_SE = cv2.vconcat([disp_B1, disp_B2])

    plt.subplot(2, 3, 6)
    plt.imshow(stacked_SE, cmap='gray', vmin=0, vmax=255)
    plt.title("SE: B1 (top) & B2 (bottom)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
