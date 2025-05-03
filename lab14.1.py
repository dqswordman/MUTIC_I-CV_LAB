# lab14.py
# ———— 计算机视觉实验：Hough 直线与圆检测 ————
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.feature import canny
from skimage.transform import (hough_line, hough_line_peaks,
                               hough_circle, hough_circle_peaks)

# ---------- 工具函数 ----------
def read_img(path):
    """读图并返回灰度图与 RGB 图"""
    img_rgb = io.imread(path)
    if img_rgb.ndim == 2:  # 灰度图
        img_gray = img_rgb
        img_rgb = np.dstack([img_rgb]*3)  # 方便后续可视化
    else:
        img_gray = color.rgb2gray(img_rgb)
    return img_rgb, img_gray

def show_fig(title, imgs, cmaps=None):
    """简易多 subplot 可视化"""
    n = len(imgs)
    plt.figure(figsize=(4*n, 4))
    for i, (name, im) in enumerate(imgs.items(), 1):
        plt.subplot(1, n, i)
        plt.title(name)
        cmap = None if cmaps is None else cmaps.get(name, None)
        plt.imshow(im, cmap=cmap)
        plt.axis('off')
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# ---------- 1. Hough 直线检测 ----------
def hough_lines_demo(img_path, low=50, high=150, sigma=2):
    rgb, gray = read_img(img_path)
    # 1) Canny 边缘
    edges = canny(gray, sigma=sigma, low_threshold=low/255., high_threshold=high/255.)
    # 2) Hough 累积空间
    h, theta, dist = hough_line(edges)
    # 3) 取局部峰值
    accums, angles, dists = hough_line_peaks(h, theta, dist, threshold=0.3*np.max(h))
    # 4) 将直线绘制到一张副本上
    overlay = rgb.copy()
    for angle, d in zip(angles, dists):
        # ρ = x cosθ + y sinθ
        (x0, y0) = d * np.array([np.cos(angle), np.sin(angle)])
        # 在图像上画两端足够长的线段
        x1 = int(x0 + 1000*(-np.sin(angle)))
        y1 = int(y0 + 1000*( np.cos(angle)))
        x2 = int(x0 - 1000*(-np.sin(angle)))
        y2 = int(y0 - 1000*( np.cos(angle)))
        cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 可视化
    show_fig("Hough 直线检测", {
        "原图": rgb,
        "Canny 边缘": edges,
        "Hough 空间\n(累积器)": h,
        "检测结果": overlay
    }, cmaps={"Canny 边缘": 'gray', "Hough 空间\n(累积器)": 'jet'})

# ---------- 2. Hough 圆检测 ----------
def hough_circles_demo(img_path, radii_range=(20, 100, 2), canny_sigma=2):
    rgb, gray = read_img(img_path)
    edges = canny(gray, sigma=canny_sigma)

    # 构造待检测的半径列表
    min_r, max_r, step = radii_range
    radii = np.arange(min_r, max_r, step)

    # 3D Hough，返回 shape = (len(radii), rows, cols)
    hspaces = hough_circle(edges, radii)
    # 查找峰值 (这里返回中心坐标与半径)
    accums, cx, cy, radii_peaks = hough_circle_peaks(hspaces, radii,
                                                     total_num_peaks=15)

    # 叠加所有半径的累积空间求一张 2D 可视化
    h_sum = np.sum(hspaces, axis=0)

    # 将圆画到图片
    overlay = rgb.copy()
    for x, y, r in zip(cx, cy, radii_peaks):
        cv2.circle(overlay, (x, y), r, (0, 255, 0), 2)

    show_fig("Hough 圆检测", {
        "原图": rgb,
        "Canny 边缘": edges,
        "Hough 累积\n(所有半径求和)": h_sum,
        "检测结果": overlay
    }, cmaps={"Canny 边缘": 'gray', "Hough 累积\n(所有半径求和)": 'jet'})

# ----------------- 主入口 -----------------
# ---------- 主入口（只改这一段） ----------
if __name__ == "__main__":
    # ① 建筑 → 直线检测
    hough_lines_demo(
        "pic/lab14-2.jpg",
        low=80,      # Canny 下阈值
        high=200,    # Canny 上阈值
        sigma=2.0    # 边缘模糊程度
    )

    # ② 硬币 → 圆检测
    hough_circles_demo(
        "pic/lab14-1.jpg",
        radii_range=(35, 85, 2),  # (最小半径, 最大半径, 步长) 需根据图像分辨率调整
        canny_sigma=2.0
    )
