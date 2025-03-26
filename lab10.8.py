import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    """
    一个最小可行例:
      A = 5×5 白块(前景1), 其余黑(0).
      B1 = 3×3 kernel, 其中部分=1(要求前景), 其余=0(不关心).
      B2 = 同样大小, 其中部分=1(要求背景), 其余=0(不关心).
    => (A⊖B1) 与 (A^c⊖B2) 在某些像素重叠 => 产生非空hit-or-miss结果.
    """

    # 1) 构造 A (8×8), 在中间画5×5白块
    A = np.zeros((8, 8), dtype=np.uint8)
    A[2:7, 2:7] = 1  # row=2..6, col=2..6

    # 2) 构造 B1, B2(3×3), 用0/1表示 必须前景/背景. 其余0=不关心.
    #   下列写法: 1 => 该处必须是前景(A) / 背景(A^c), 0 => 不关心
    B1 = np.array([
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 0]
    ], dtype=np.uint8)
    #   B2 要求右下角1 => 该位置必须是背景, 其余=0不关心
    B2 = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1]
    ], dtype=np.uint8)

    # 3) MORPH_HITMISS 要求图像是0/1, kernel中1=前景, -1=必为0(背景).
    #   但OpenCV对 "必须背景" 只识别 kernel=-1. 这里我们用 trick:
    #   (A⊖B1)∩(A^c⊖B2) 的"集合定义"做:
    #    (A⊖B1): 先把 B1中=1的地方当 structuring element
    #    (A^c⊖B2): 把 B2中=1的地方当 structuring element
    #   => bitwise_and
    #
    #   为了简化, 我们自己"手动腐蚀"  => any pixel that doesn't match => 0
    #   但OpenCV自带的erode不认"B1里0=不关心". 需自己写代码 or 做 trick:
    #
    #   这里直接使用 "集合论" 方式:
    #   "A ⊖ B1" => 对 B1==1位置做腐蚀, B1==0位置不管 => 只需那些位置=1
    #   "A^c ⊖ B2" => 对 B2==1位置做腐蚀
    #
    #   简易做法: 用 cv2.MORPH_HITMISS 也行, 但它不支持"B2"多值,
    #             这里演示最原始集合论.

    # 3.1) A⊖B1
    #    对图中每个像素p, 若B1==1的邻域像素都=1 => p保留; 否则p=0
    #    => 需自写 or 用二值"mask"比对
    erodedA_B1 = morphological_erode_custom(A, B1)

    # 3.2) A^c ⊖ B2
    A_c = 1 - A  # 0->1, 1->0
    erodedAc_B2 = morphological_erode_custom(A_c, B2)

    # 3.3) 取交集 => hits
    hits = erodedA_B1 & erodedAc_B2

    # 统计命中
    hits_count = np.count_nonzero(hits == 1)
    print(f"Number of foreground pixels in Hit-or-Miss results: {hits_count}")

    # 4) 可视化(2×3子图)
    #   转成0/255以便imshow
    A_255 = (A*255).astype(np.uint8)
    erA_255 = (erodedA_B1*255).astype(np.uint8)
    erAc_255 = (erodedAc_B2*255).astype(np.uint8)
    hits_255 = (hits*255).astype(np.uint8)

    # B1, B2 也转成可视化
    disp_B1 = (B1*255).astype(np.uint8)
    disp_B2 = (B2*255).astype(np.uint8)
    # 上下拼
    disp_B12 = np.zeros((6, 3), dtype=np.uint8)  # 3+3行,3列
    disp_B12[0:3, :] = disp_B1
    disp_B12[3:6, :] = disp_B2

    fig, axes = plt.subplots(2, 3, figsize=(9,6))
    axes[0,0].imshow(A_255,cmap='gray');axes[0,0].set_title("A");axes[0,0].axis('off')
    axes[0,1].imshow(disp_B12,cmap='gray');axes[0,1].set_title("B1(top), B2(bottom)");axes[0,1].axis('off')
    axes[0,2].imshow(erA_255,cmap='gray');axes[0,2].set_title("A ⊖ B1");axes[0,2].axis('off')
    axes[1,0].imshow(erAc_255,cmap='gray');axes[1,0].set_title("A^c ⊖ B2");axes[1,0].axis('off')
    axes[1,1].imshow(hits_255,cmap='gray');axes[1,1].set_title("(A⊖B1) ∩ (A^c⊖B2)");axes[1,1].axis('off')
    axes[1,2].axis('off')
    plt.tight_layout()
    plt.show()

def morphological_erode_custom(bin_img, kernel_01):
    """
    自定义腐蚀: 只对 kernel_01==1 的位置做“必须为1”检查, kernel_01==0=>不关心.
    bin_img=0/1, kernel_01=0/1. => 返回0/1图.
    """
    h, w = bin_img.shape
    kh, kw = kernel_01.shape
    # 以kernel中心对准当前像素 => kernel半径
    # 这里简单起见, 把"origin"置于kernel中心( floor(kh/2, kw/2) )
    origin_y = kh // 2
    origin_x = kw // 2

    out = np.zeros_like(bin_img)
    for r in range(h):
        for c in range(w):
            # 检查 kernel_01==1 的相对位置
            is_ok = True
            for kr in range(kh):
                for kc in range(kw):
                    if kernel_01[kr,kc]==1:
                        # 需要 bin_img[r+(kr-origin_y), c+(kc-origin_x)]==1
                        rr = r + (kr - origin_y)
                        cc = c + (kc - origin_x)
                        # 超出边界 or 不是1 =>fail
                        if rr<0 or rr>=h or cc<0 or cc>=w or bin_img[rr,cc]!=1:
                            is_ok=False
                            break
                if not is_ok: break
            out[r,c] = 1 if is_ok else 0
    return out

if __name__=="__main__":
    main()
