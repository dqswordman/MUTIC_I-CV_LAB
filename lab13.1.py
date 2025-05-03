import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color, morphology, img_as_float, exposure
from skimage.morphology import disk, reconstruction


def demo_morphological_gradient(image_path):
    """
    演示题目一：读取图像、灰度化，然后做开运算/闭运算或直接做形态学梯度，并可视化不同半径效果
    """
    # 1. 读取原图
    img = io.imread(image_path)
    # 如果图像是 RGBA(4通道)，先转换为 RGB(3通道)
    if len(img.shape) == 3 and img.shape[-1] == 4:
        img = color.rgba2rgb(img)

    # 如果本身就是灰度或其他情况，为了统一处理，转为 float 格式
    if len(img.shape) == 2:
        # 本身是灰度图
        img_gray = img_as_float(img)
    else:
        # 彩色图，转灰度
        img_gray = color.rgb2gray(img)

    # 2. 分别使用 radius = 1, 3, 5 做形态学梯度
    radii = [1, 3, 5]
    gradient_results = []

    for r in radii:
        se = disk(r)
        dilated = morphology.dilation(img_gray, se)
        eroded = morphology.erosion(img_gray, se)
        gradient_img = dilated - eroded
        gradient_results.append((r, gradient_img))

    # 3. 可视化：原图 + 不同半径的梯度结果
    fig, axes = plt.subplots(1, 1 + len(radii), figsize=(15, 5))
    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title("Original Gray Image")
    axes[0].axis('off')

    for i, (r, grad) in enumerate(gradient_results):
        axes[i + 1].imshow(grad, cmap='gray')
        axes[i + 1].set_title(f"Gradient (radius={r})")
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.show()


def demo_textural_segmentation(image_path):
    """
    演示题目二：纹理分割，包含开运算、闭运算及形态学梯度叠加显示
    最终可视化4幅图：原图、开运算结果、闭运算结果、梯度叠加
    """
    # 1. 读取原图
    img = io.imread(image_path)
    # 检测 RGBA -> RGB
    if len(img.shape) == 3 and img.shape[-1] == 4:
        img = color.rgba2rgb(img)

    # 判断是否灰度
    if len(img.shape) == 2:
        img_gray = img_as_float(img)
    else:
        img_gray = color.rgb2gray(img)

    # 2. 开运算 (例如半径=60，可根据实验要求调整)
    se_open = disk(60)
    opened_img = morphology.opening(img_gray, se_open)

    # 3. 闭运算 (例如半径=30)
    se_close = disk(30)
    closed_img = morphology.closing(img_gray, se_close)

    # 4. 形态学梯度
    se_grad = disk(3)
    dilated = morphology.dilation(img_gray, se_grad)
    eroded = morphology.erosion(img_gray, se_grad)
    gradient_img = dilated - eroded

    # 简单制作一个梯度叠加到原图的效果
    grad_norm = exposure.rescale_intensity(gradient_img, out_range=(0, 1))
    overlay = 0.7 * img_gray + 0.3 * grad_norm  # 线性混合

    # 5. 可视化四幅图像
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].imshow(img_gray, cmap='gray')
    axes[0, 0].set_title("Original Gray Image")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(opened_img, cmap='gray')
    axes[0, 1].set_title("Opening (radius=60)")
    axes[0, 1].axis('off')

    axes[1, 0].imshow(closed_img, cmap='gray')
    axes[1, 0].set_title("Closing (radius=30)")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(overlay, cmap='gray')
    axes[1, 1].set_title("Morphological Gradient (overlay)")
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()


def demo_morphological_reconstruction(image_path):
    """
    演示题目三：灰度形态重建
    包含Opening by Reconstruction和Top-Hat by Reconstruction的可视化
    最终可视化三幅图：原图、开运算重建图和Top-Hat结果
    """
    # 1. 读取原图
    img = io.imread(image_path)
    # 检测 RGBA -> RGB
    if len(img.shape) == 3 and img.shape[-1] == 4:
        img = color.rgba2rgb(img)

    # 判断是否灰度
    if len(img.shape) == 2:
        img_gray = img_as_float(img)
    else:
        img_gray = color.rgb2gray(img)

    # 2. 准备结构元素(例如半径=5，可根据情况调整)
    se = disk(5)

    # 2.1 先做腐蚀(得到marker)，再做基于原图的重建(Opening by Reconstruction)
    eroded = morphology.erosion(img_gray, se)
    opened_by_recon = reconstruction(eroded, img_gray)

    # 2.2 计算Top-Hat by Reconstruction: 原图 - 开运算重建图
    top_hat = img_gray - opened_by_recon

    # 3. 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title("Original Gray Image")
    axes[0].axis('off')

    axes[1].imshow(opened_by_recon, cmap='gray')
    axes[1].set_title("Opening by Reconstruction")
    axes[1].axis('off')

    axes[2].imshow(top_hat, cmap='gray')
    axes[2].set_title("Top-Hat by Reconstruction")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ===== 题目一：形态学梯度示例 =====
    demo_morphological_gradient("pic/lab13-1.png")

    # ===== 题目二：纹理分割示例 =====
    demo_textural_segmentation("pic/lab13-2.png")

    # ===== 题目三：灰度形态重建示例 =====
    demo_morphological_reconstruction("pic/lab13-3.png")
