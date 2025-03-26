import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    """
    1) 构造 7×7 的二值图 A(0/1)，内含一个 3×3 的前景方块；
    2) 定义 3×3 的结构元素 K(仅中心=1，其余=0)，类型为 uint8；
    3) 使用 cv2.morphologyEx(A, cv2.MORPH_HITMISS, K) 进行 Hit-or-Miss；
    4) 可视化展示：A、K、Hit-or-Miss 结果，并在控制台打印命中数量。
    """

    # 1. 构造 7×7 的二值图 A，元素仅 0/1
    A = np.zeros((7, 7), dtype=np.uint8)
    # 在行列 [1..3] 上填 1，形成 3×3 前景块
    A[1:4, 1:4] = 1

    # 2. 定义 3×3 的结构元素 K, 用 uint8 避免溢出
    #    中心=1(必须前景)，其余=0(不关心)
    K = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ], dtype=np.uint8)

    # 3. 使用 OpenCV 的 MORPH_HITMISS
    #    A 要为 0/1 前景/背景；K 中 1=必须前景, 0=不关心
    hit_miss_raw = cv2.morphologyEx(A, cv2.MORPH_HITMISS, K)

    # 命中结果也是 0/1；统计其中“1”的个数
    hits_count = np.count_nonzero(hit_miss_raw == 1)
    print(f"Hit-or-Miss 结果中前景像素数量: {hits_count}")

    # 4. 可视化
    #    (a) 将 A、K、结果 都从 0/1 转为 0/255 方便显示
    A_255      = (A * 255).astype(np.uint8)
    K_255      = np.where(K > 0, 255, 0).astype(np.uint8)
    hit_miss_255 = (hit_miss_raw * 255).astype(np.uint8)

    plt.figure(figsize=(10, 3))

    # (1) 原图 A
    plt.subplot(1, 3, 1)
    plt.imshow(A_255, cmap='gray', vmin=0, vmax=255)
    plt.title("Binary Image A (7×7)")
    plt.axis('off')

    # (2) 结构元素 K
    plt.subplot(1, 3, 2)
    plt.imshow(K_255, cmap='gray', vmin=0, vmax=255)
    plt.title("Kernel K (3×3)")
    plt.axis('off')

    # (3) Hit-or-Miss 结果
    plt.subplot(1, 3, 3)
    plt.imshow(hit_miss_255, cmap='gray', vmin=0, vmax=255)
    plt.title("Hit-or-Miss Output")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
