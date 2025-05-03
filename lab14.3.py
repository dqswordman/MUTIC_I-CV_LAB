# lab14_v4.py —— Hough 直线 + 圆检测（最终版）
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color


# ======== 通用 ========
def read_rgb_gray(path):
    rgb = io.imread(path)
    gray = (color.rgb2gray(rgb) * 255).astype(np.uint8) if rgb.ndim == 3 else rgb
    return rgb, gray


def show_row(title, images, cmaps=None, size=(20, 4)):
    n = len(images)
    plt.figure(figsize=size)
    for i, (k, v) in enumerate(images.items(), 1):
        plt.subplot(1, n, i)
        plt.title(k)
        cmap = None if cmaps else None if cmaps is None else cmaps.get(k, None)
        cmap = cmaps.get(k, None) if cmaps else None
        plt.imshow(v, cmap=cmap)
        plt.axis('off')
    plt.suptitle(title, fontsize=18)
    plt.tight_layout()
    plt.show()


# ======== 一、建筑：直线检测 ========
def detect_lines(img_path,
                 canny_low=40, canny_high=120, blur_sigma=1.0,
                 hough_thresh=80, min_len=70, max_gap=6,
                 keep_n=40):
    rgb, gray = read_rgb_gray(img_path)

    # Canny
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), blur_sigma),
                      canny_low, canny_high)

    # HoughLinesP
    segs = cv2.HoughLinesP(edges,
                           rho=1,
                           theta=np.pi / 180,
                           threshold=hough_thresh,
                           minLineLength=min_len,
                           maxLineGap=max_gap)
    segs = segs[:, 0, :] if segs is not None else np.empty((0, 4))

    # 只取竖直 / 水平
    v_lines, h_lines = [], []
    for x1, y1, x2, y2 in segs:
        dx, dy = x2 - x1, y2 - y1
        ang = (np.degrees(np.arctan2(dy, dx)) + 180) % 180
        length = np.hypot(dx, dy)
        if abs(ang - 90) < 8:      # 竖直
            v_lines.append((length, (x1, y1, x2, y2)))
        elif abs(ang) < 8:         # 水平
            h_lines.append((length, (x1, y1, x2, y2)))

    # 各保留 length Top‑N
    v_lines = sorted(v_lines, reverse=True)[:keep_n]
    h_lines = sorted(h_lines, reverse=True)[:keep_n]

    # 绘制
    overlay = rgb.copy()
    for _, (x1, y1, x2, y2) in v_lines + h_lines:
        cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

    show_row("Hough 直线检测（v4）", {
        "原图": rgb,
        "Canny 边缘": edges,
        "检测结果": overlay
    }, cmaps={"Canny 边缘": 'gray'})


# ======== 二、硬币：圆检测 ========
def deduplicate(circles, center_tol=30, radius_tol=5):
    """
    circles: (x,y,r) ndarray
    """
    keep = []
    for c in circles:
        x, y, r = c
        dup = False
        for k in keep:
            if np.hypot(x - k[0], y - k[1]) < center_tol and abs(r - k[2]) < radius_tol:
                dup = True
                break
        if not dup:
            keep.append(c)
    return np.array(keep)


def detect_circles(img_path,
                   blur_ksize=7,
                   dp=1.2,
                   min_dist=None,
                   param1=120,
                   param2=28,
                   min_r=40,
                   max_r=60):
    rgb, gray = read_rgb_gray(img_path)

    g_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    if min_dist is None:
        # 粗估硬币直径 ≈ (min_r+max_r)/2 * 2
        min_dist = (min_r + max_r)

    circles = cv2.HoughCircles(g_blur,
                               cv2.HOUGH_GRADIENT,
                               dp=dp,
                               minDist=min_dist,
                               param1=param1,
                               param2=param2,
                               minRadius=min_r,
                               maxRadius=max_r)

    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        circles = deduplicate(circles, center_tol=30, radius_tol=5)
    else:
        circles = np.empty((0, 3), dtype=int)

    overlay = rgb.copy()
    for x, y, r in circles:
        cv2.circle(overlay, (x, y), r, (0, 255, 0), 2)

    # 为了说明 Hough 空间，这里把梯度版本的“空”占位图也放上
    placeholder = np.zeros_like(gray)

    show_row("Hough 圆检测（v4, OpenCV Gradient）", {
        "原图": rgb,
        "Canny 边缘": cv2.Canny(g_blur, 30, 90),
        "占位": placeholder,       # 仅占位示意，可换成别的可视化
        "检测结果": overlay
    }, cmaps={"Canny 边缘": 'gray', "占位": 'gray'})


# ========= main =========
if __name__ == "__main__":
    # 1) 建筑 — 直线
    detect_lines("pic/lab14-2.jpg",
                 canny_low=40, canny_high=120,
                 hough_thresh=80, min_len=70, max_gap=8,
                 keep_n=40)

    # 2) 硬币 — 圆
    detect_circles("pic/lab14-1.jpg",
                   blur_ksize=7,
                   dp=1.2,
                   param1=120, param2=28,
                   min_r=40, max_r=60)
