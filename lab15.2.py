"""
harris_with_eigen_and_interpretation_fixed.py
---------------------------------------------
1) Harris 角点可视化（原图、热力图、二值掩膜、叠加红点）
2) λ₁-λ₂ 散点 + 四类典型 patch 的 3D 灰度面
3) “interpreting eigenvalues” 示意图

依赖：
    pip install opencv-python numpy matplotlib
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ------------- 配置 -------------
IMG_PATH      = r"pic/lab15.png"
GAUSS_K       = (5,5)
BLOCK_SIZE    = 2
KSIZE         = 3
K_HARRIS      = 0.04
THRESH_RATIO  = 0.001
CLOSE_ELIPSE  = (11,11)
DILATE_K      = np.ones((3,3), np.uint8)
# --------------------------------

# 1. 读图与预处理
bgr  = cv2.imread(IMG_PATH)
if bgr is None:
    raise FileNotFoundError(f"Cannot open image: {IMG_PATH}")
rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
gray = cv2.GaussianBlur(gray, GAUSS_K, 0)

# 2. Harris 响应 & 膨胀
R = cv2.cornerHarris(gray, BLOCK_SIZE, KSIZE, K_HARRIS)
R = cv2.dilate(R, DILATE_K)

# 3. 热力图 & 掩膜
R_pos  = np.clip(R, 0, None)
R_norm = cv2.normalize(R_pos, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

thr    = R_pos.max() * THRESH_RATIO
mask0  = (R_pos >= thr).astype(np.uint8)
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CLOSE_ELIPSE)
mask   = cv2.morphologyEx(mask0, cv2.MORPH_CLOSE, kernel_close)
mask   = cv2.dilate(mask, DILATE_K, iterations=1)

# 4. 计算 λ₁, λ₂
eig  = cv2.cornerEigenValsAndVecs(gray, BLOCK_SIZE, KSIZE)
lam1 = eig[...,0]
lam2 = eig[...,1]

# 5. 挑代表点
flat_mask = (np.maximum(abs(lam1), abs(lam2)) < 1e-4)
edge_h    = (abs(lam2) > abs(lam1)*6) & (~flat_mask)
edge_v    = (abs(lam1) > abs(lam2)*6) & (~flat_mask)
corner_m  = (mask0>0) & (~flat_mask) & (~edge_h) & (~edge_v)

rng = np.random.default_rng(0)
def pick(mask):
    ys, xs = np.where(mask)
    if len(xs)==0:
        ys, xs = np.where(~flat_mask)
    idx = rng.integers(0, len(xs))
    return xs[idx], ys[idx]

samples = {
    'flat':   pick(flat_mask),
    'h-edge': pick(edge_h),
    'v-edge': pick(edge_v),
    'corner': pick(corner_m)
}
labels = {'flat':'Flat','h-edge':'Horizontal edge',
          'v-edge':'Vertical edge','corner':'Corner'}

def safe_patch(y,x,r=6):
    y0,y1 = max(y-r,0), min(y+r+1,gray.shape[0])
    x0,x1 = max(x-r,0), min(x+r+1,gray.shape[1])
    return gray[y0:y1,x0:x1]

# 6. 四联图
fig1, axs1 = plt.subplots(1,4,figsize=(18,5))
axs1[0].imshow(rgb); axs1[0].set_title("Original");                    axs1[0].axis("off")
axs1[1].imshow(R_norm, cmap="jet", vmin=0, vmax=255); axs1[1].set_title("Corner response"); axs1[1].axis("off")
axs1[2].imshow(mask, cmap="gray"); axs1[2].set_title("Thresholded corner response"); axs1[2].axis("off")
overlay = rgb.copy()
ys, xs = np.where(mask)
for x,y in zip(xs,ys):
    cv2.circle(overlay,(x,y),2,(255,0,0),1)
axs1[3].imshow(overlay); axs1[3].set_title("Corners overlay"); axs1[3].axis("off")
plt.tight_layout()

# 7. λ₁-λ₂ 散点 + 3D patches (2×3 布局)
fig2 = plt.figure(figsize=(14,8))
# 散点图
ax_sc = fig2.add_subplot(2,3,1)
step = 5
ax_sc.scatter(lam1[::step,::step].ravel(),
              lam2[::step,::step].ravel(),
              s=2, alpha=0.3)
for key,(x,y) in samples.items():
    ax_sc.scatter(lam1[y,x], lam2[y,x], s=80, marker='*', label=labels[key])
ax_sc.set_xlabel("λ₁"); ax_sc.set_ylabel("λ₂"); ax_sc.set_title("λ₁-λ₂ space"); ax_sc.legend()

# 3D patch
for idx,(key,(x,y)) in enumerate(samples.items(), start=2):
    ax3d = fig2.add_subplot(2,3,idx, projection='3d')
    patch = safe_patch(y,x)
    X,Y   = np.meshgrid(range(patch.shape[1]), range(patch.shape[0]))
    ax3d.plot_surface(X,Y,patch, cmap='viridis', rstride=1, cstride=1,
                      linewidth=0, antialiased=False)
    ax3d.set_title(labels[key])
    ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([])
plt.tight_layout()

# 8. interpreting eigenvalues 示意图（同前）
fig3 = plt.figure(figsize=(8,8))
ax3 = fig3.add_subplot(111)
ax3.arrow(0,0,1,0, head_width=0.04, length_includes_head=True, color='navy')
ax3.arrow(0,0,0,1, head_width=0.04, length_includes_head=True, color='navy')
ax3.set_xlim(0,1); ax3.set_ylim(0,1)
ax3.set_xticks([]); ax3.set_yticks([])
ax3.set_xlabel("λ₁"); ax3.set_ylabel("λ₂")
ax3.set_title("Interpreting eigenvalues")
ax3.set_facecolor('#ddeeff')

insets = {
    'flat':   [0.05,0.05,0.3,0.3],
    'h-edge': [0.05,0.65,0.3,0.3],
    'v-edge': [0.65,0.05,0.3,0.3],
    'corner': [0.65,0.65,0.3,0.3]
}
for key,pos in insets.items():
    xpix, ypix = samples[key]
    lamx, lamy = lam1[ypix,xpix], lam2[ypix,xpix]
    ax3.plot(lamx/lam1.max(), lamy/lam2.max(), 'o', color='k')
    axins = fig3.add_axes(pos, projection='3d')
    patch = safe_patch(ypix,xpix)
    X,Y   = np.meshgrid(range(patch.shape[1]), range(patch.shape[0]))
    axins.plot_surface(X,Y,patch, cmap='viridis', rstride=1, cstride=1,
                       linewidth=0)
    axins.set_xticks([]); axins.set_yticks([]); axins.set_zticks([])
    axins.set_title(labels[key], fontsize=10)

plt.show()
