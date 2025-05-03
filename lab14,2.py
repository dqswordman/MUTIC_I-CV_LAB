# lab14_v2.py  —— 计算机视觉实验：Hough 直线&圆检测（优化版）
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, feature, transform

# ======== 通用工具 ========
def read_img(path):
    rgb = io.imread(path)
    if rgb.ndim == 2:
        gray = rgb
        rgb = np.dstack([rgb] * 3)
    else:
        gray = color.rgb2gray(rgb)
    return rgb, (gray * 255).astype(np.uint8)

def show_fig(title, imgs, cmaps=None, figsize=(16, 4)):
    n = len(imgs)
    plt.figure(figsize=figsize)
    for i, (k, v) in enumerate(imgs.items(), 1):
        plt.subplot(1, n, i)
        plt.title(k)
        cmap = None if cmaps is None else cmaps.get(k, None)
        plt.imshow(v, cmap=cmap)
        plt.axis('off')
    plt.suptitle(title, fontsize=18)
    plt.tight_layout()
    plt.show()

# ======== 线段非极大抑制 ========
def nms_lines(lines, angle_thresh=3, dist_thresh=20):
    """
    聚类近似平行且距离接近的线段，只保留每组最长的一条
    lines: (x1,y1,x2,y2,angle,length)
    """
    keep = []
    for l in lines:
        x1, y1, x2, y2, ang, length = l
        ok = True
        for k in keep:
            _, _, _, _, ang_k, _ = k
            if abs(ang - ang_k) < angle_thresh:
                # 同角度，再判距离
                if np.hypot(x1 - k[0], y1 - k[1]) < dist_thresh:
                    ok = False
                    # 更长则替换
                    if length > k[5]:
                        keep.remove(k)
                        keep.append(l)
                    break
        if ok:
            keep.append(l)
    return keep

# ======== 圆非极大抑制 ========
def nms_circles(circles, min_dist=40):
    """
    circles: list of (x,y,r,accum)
    """
    circles.sort(key=lambda c: -c[3])  # 按累积值降序
    keep = []
    for c in circles:
        x, y, r, a = c
        if all(np.hypot(x - k[0], y - k[1]) > min_dist for k in keep):
            keep.append(c)
    return keep

# ======== 1. 直线检测 ========
def hough_lines_demo(img_path,
                     canny_low=50, canny_high=150, sigma=1.5,
                     minlen=120, maxgap=10):
    rgb, gray = read_img(img_path)

    # 预处理 & 边缘
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), sigma),
                      canny_low, canny_high, apertureSize=3)

    # Probabilistic Hough
    raw = cv2.HoughLinesP(edges, 1, np.pi / 180,
                          threshold=120,
                          minLineLength=minlen,
                          maxLineGap=maxgap)

    if raw is None:
        raw = np.empty((0, 1, 4))
    raw = raw[:, 0, :]  # (N,4)

    # 计算角度 & 长度，过滤近水平/垂直
    lines = []
    for x1, y1, x2, y2 in raw:
        dx, dy = x2 - x1, y2 - y1
        ang = np.degrees(np.arctan2(dy, dx))
        ang = (ang + 180) % 180  # 0~180
        if (abs(ang) < 10) or (abs(ang - 90) < 10):
            length = np.hypot(dx, dy)
            lines.append([x1, y1, x2, y2, ang, length])

    # NMS
    lines = nms_lines(lines)

    # 绘制
    overlay = rgb.copy()
    for x1, y1, x2, y2, _, _ in lines:
        cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

    show_fig("Hough 直线检测（优化）", {
        "原图": rgb,
        "Canny 边缘": edges,
        "检测结果": overlay
    }, cmaps={"Canny 边缘": 'gray'})

# ======== 2. 圆检测 ========
def hough_circles_demo(img_path,
                       radii=(35, 80, 2),
                       canny_sigma=1.2,
                       peak_thresh=0.4,  # 累积值阈值（相对）
                       max_peaks=30):
    rgb, gray = read_img(img_path)

    # 自适应 Canny
    v_med = np.median(gray)
    lower = int(max(0, 0.7 * v_med))
    upper = int(min(255, 1.3 * v_med))
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), canny_sigma),
                      lower, upper)

    # Hough
    min_r, max_r, step = radii
    r_range = np.arange(min_r, max_r, step)
    hspaces = transform.hough_circle(edges, r_range)
    accums, cx, cy, r_best = transform.hough_circle_peaks(
        hspaces, r_range,
        threshold=peak_thresh * hspaces.max(),
        min_xdistance=15,
        min_ydistance=15,
        normalize=True,
        total_num_peaks=max_peaks
    )

    # 打包 & NMS
    circles = [(x, y, r, a) for x, y, r, a in zip(cx, cy, r_best, accums)]
    circles = nms_circles(circles, min_dist=min_r)

    # 绘制
    overlay = rgb.copy()
    for x, y, r, _ in circles:
        cv2.circle(overlay, (x, y), r, (0, 255, 0), 2)

    # 把所有半径累加可视化
    hsum = hspaces.sum(0)

    show_fig("Hough 圆检测（优化）", {
        "原图": rgb,
        "Canny 边缘": edges,
        "Hough 累积和": hsum,
        "检测结果": overlay
    }, cmaps={"Canny 边缘": 'gray', "Hough 累积和": 'jet'}, figsize=(18, 4))

# ======== 主函数 ========
if __name__ == "__main__":
    # ① 建筑图：只保留笔直楼体线
    hough_lines_demo("pic/lab14-2.jpg",
                     canny_low=60, canny_high=180,
                     sigma=1.5, minlen=120)

    # ② 硬币图：尽量检测全部硬币
    hough_circles_demo("pic/lab14-1.jpg",
                       radii=(35, 80, 2),
                       canny_sigma=1.0,
                       peak_thresh=0.35,
                       max_peaks=40)
