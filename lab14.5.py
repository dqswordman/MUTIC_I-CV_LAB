# lab14_v5.py  —— Hough 直线 & 圆检测（最终版）
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color

# =============== 通用工具 ===============
def read_rgb_gray(path):
    rgb = io.imread(path)
    gray = (color.rgb2gray(rgb) * 255).astype(np.uint8) if rgb.ndim == 3 else rgb
    return rgb, gray

def show_row(title, images, cmaps=None, size=(22, 5)):
    plt.figure(figsize=size)
    for i, (name, img) in enumerate(images.items(), 1):
        plt.subplot(1, len(images), i)
        plt.title(name)
        cmap = cmaps.get(name) if cmaps else None
        plt.imshow(img, cmap=cmap)
        plt.axis('off')
    plt.suptitle(title, fontsize=18)
    plt.tight_layout()
    plt.show()

# =============== 1. 建筑 — 直线检测 ===============
def detect_lines(img_path,
                 canny_low=40, canny_high=120, blur_sigma=1.0,
                 hough_thresh=80, min_len=70, max_gap=6,
                 min_strict_len=120, keep_n=40):
    """Probabilistic Hough 细化版：仅保留长竖直/水平线"""
    rgb, gray = read_rgb_gray(img_path)

    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), blur_sigma),
                      canny_low, canny_high)

    segs = cv2.HoughLinesP(edges, 1, np.pi/180,
                           threshold=hough_thresh,
                           minLineLength=min_len,
                           maxLineGap=max_gap)
    segs = segs[:, 0, :] if segs is not None else np.empty((0, 4))

    v_lines, h_lines = [], []
    for x1, y1, x2, y2 in segs:
        dx, dy = x2 - x1, y2 - y1
        ang = (np.degrees(np.arctan2(dy, dx)) + 180) % 180
        length = np.hypot(dx, dy)
        if length < min_strict_len:          # 去掉短线
            continue
        if abs(ang - 90) < 8:                # 竖直
            v_lines.append((length, (x1, y1, x2, y2)))
        elif abs(ang) < 8:                   # 水平
            h_lines.append((length, (x1, y1, x2, y2)))

    v_lines = sorted(v_lines, reverse=True)[:keep_n]
    h_lines = sorted(h_lines, reverse=True)[:keep_n]

    overlay = rgb.copy()
    for _, (x1, y1, x2, y2) in v_lines + h_lines:
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255),
                 1, cv2.LINE_AA)            # 细红线

    # 半透明叠加
    result = cv2.addWeighted(overlay, 0.5, rgb, 0.5, 0)

    show_row("Hough 直线检测（v5）", {
        "原图": rgb,
        "Canny 边缘": edges,
        "检测结果": result
    }, cmaps={"Canny 边缘": 'gray'})

# =============== 2. 硬币 — 圆检测 ===============
def deduplicate(circles, center_tol=35, radius_tol=4):
    keep = []
    for (x, y, r) in circles:
        if all(np.hypot(x - k[0], y - k[1]) > center_tol or
               abs(r - k[2]) > radius_tol for k in keep):
            keep.append((x, y, r))
    return np.array(keep, dtype=int)

def detect_circles(img_path,
                   blur_ksize=7, clahe_clip=2.0,
                   dp=1.2, param1=120, param2=22,
                   min_r=38, max_r=65):
    rgb, gray = read_rgb_gray(img_path)

    # CLAHE 预处理
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray_eq, (blur_ksize, blur_ksize), 0)

    # HoughCircles
    min_dist = int((min_r + max_r))
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
        param1=param1, param2=param2,
        minRadius=min_r, maxRadius=max_r
    )
    circles = np.round(circles[0, :]).astype(int) if circles is not None else np.empty((0, 3))
    circles = deduplicate(circles, center_tol=35, radius_tol=4)

    overlay = rgb.copy()
    for x, y, r in circles:
        cv2.circle(overlay, (x, y), r, (0, 255, 0), 2)

    show_row("Hough 圆检测（v5）", {
        "原图": rgb,
        "Canny 边缘": cv2.Canny(blur, 30, 90),
        "检测结果": overlay
    }, cmaps={"Canny 边缘": 'gray'})

# =============== main ===============
if __name__ == "__main__":
    # 1) 建筑
    detect_lines("pic/lab14-2.jpg")

    # 2) 硬币
    detect_circles("pic/lab14-1.jpg")
