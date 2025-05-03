"""
harris_with_eigen_plot_fixed.py
--------------------------------
* Adds Gaussian smoothing before Harris
* Uses manual min-max normalization (keeps negative R values)
* Lowers threshold ratio (TH_RATIO = 0.008)
Everything else identical: 4-panel view + λ1-λ2 map + 3-D patches
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (imported for side-effect)

# ---------- 1. configuration ----------
IMG_PATH   = r"pic/lab15.png"   # ← update to your own path
BLOCK_SIZE = 4                    # window size for M
KSIZE      = 5                    # Sobel kernel size
K_HARRIS   = 0.06                 # Harris free parameter k
TH_RATIO   = 0.008                # R threshold ratio (0.8 %)
RATIO_EDGE = 6                    # λ multiple to classify edges
PATCH_R    = 6                    # half-patch radius for 3-D surfaces
FLAT_EPS   = 1e-4                 # flat region tolerance
rng        = np.random.default_rng(0)
# --------------------------------------

# ---------- 2. read image ----------
bgr = cv2.imread(IMG_PATH)
if bgr is None:
    raise FileNotFoundError(f"Unable to read image: {IMG_PATH}")
rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

# ---------- 3. pre-blur & Harris ----------
gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
R   = cv2.cornerHarris(gray_blur, BLOCK_SIZE, KSIZE, K_HARRIS)
eig = cv2.cornerEigenValsAndVecs(gray_blur, BLOCK_SIZE, KSIZE)  # (...,6)
lam1 = eig[..., 0]      # smaller eigen-value
lam2 = eig[..., 1]      # larger  eigen-value

# ---------- 4. region masks ----------
flat_mask = (np.maximum(np.abs(lam1), np.abs(lam2)) < FLAT_EPS)
edge_h    = (np.abs(lam2) > np.abs(lam1) * RATIO_EDGE) & (~flat_mask)
edge_v    = (np.abs(lam1) > np.abs(lam2) * RATIO_EDGE) & (~flat_mask)
corner_m  = ((~flat_mask) & (~edge_h) & (~edge_v) &
             (R > TH_RATIO * R.max()))

def pick(mask, name):
    """randomly pick one pixel; if empty fall back to any non-flat pixel"""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        print(f"[warning] '{name}' mask empty, falling back to non-flat pixels")
        ys, xs = np.where(~flat_mask)
    idx = rng.integers(0, len(xs))
    return xs[idx], ys[idx]

xs_f, ys_f = pick(flat_mask, "flat")
xs_h, ys_h = pick(edge_h,   "h-edge")
xs_v, ys_v = pick(edge_v,   "v-edge")
xs_c, ys_c = pick(corner_m, "corner")
samples = {'flat':(xs_f,ys_f), 'h-edge':(xs_h,ys_h),
           'v-edge':(xs_v,ys_v), 'corner':(xs_c,ys_c)}

# ---------- 5. sparse scatter for λ-space ----------
step = 5
scatter_l1 = lam1[::step, ::step].ravel()
scatter_l2 = lam2[::step, ::step].ravel()

# ---------- 6. four-panel visualization ----------
fig1, ax = plt.subplots(1, 4, figsize=(18, 5))

ax[0].imshow(rgb)
ax[0].set_title("Original")
ax[0].axis("off")

# manual min-max normalization (keeps sign)
R_vis = (R - R.min()) / (R.max() - R.min())
ax[1].imshow(R_vis, cmap="jet")
ax[1].set_title("Corner response")
ax[1].axis("off")

mask = R > TH_RATIO * R.max()
ax[2].imshow(mask, cmap="gray")
ax[2].set_title("Thresholded R")
ax[2].axis("off")

overlay = rgb.copy()
ys_corn, xs_corn = np.where(mask)
for (x, y) in zip(xs_corn, ys_corn):
    cv2.circle(overlay, (x, y), 2, (255, 0, 0), 1)
ax[3].imshow(overlay)
ax[3].set_title("Corners overlay")
ax[3].axis("off")

fig1.tight_layout()

# ---------- 7. λ1-λ2 feature-space map ----------
fig2 = plt.figure(figsize=(8, 8))
ax2  = fig2.add_subplot(111)
ax2.scatter(scatter_l1, scatter_l2, s=3, alpha=.3)

styles = dict(flat='o', **{'h-edge':'^', 'v-edge':'s', 'corner':'*'})
colors = dict(flat='k', **{'h-edge':'b', 'v-edge':'g', 'corner':'r'})
labels = dict(flat='Flat',
              **{'h-edge':'Horizontal edge', 'v-edge':'Vertical edge', 'corner':'Corner'})

for key, (x, y) in samples.items():
    ax2.scatter(lam1[y, x], lam2[y, x], s=120,
                marker=styles[key], c=colors[key],
                label=f"{labels[key]}  λ1={lam1[y, x]:.1e}, λ2={lam2[y, x]:.1e}")
    ax2.annotate(labels[key], (lam1[y, x], lam2[y, x]),
                 textcoords="offset points", xytext=(5, 5), fontsize=12)

ax2.set_xlabel("λ1")
ax2.set_ylabel("λ2")
ax2.set_title("λ1-λ2 feature-space map")
ax2.legend()
ax2.grid(True)

# ---------- 8. 3-D grayscale surface patches ----------
def safe_patch(y, x):
    y0, y1 = max(y - PATCH_R, 0), min(y + PATCH_R + 1, gray.shape[0])
    x0, x1 = max(x - PATCH_R, 0), min(x + PATCH_R + 1, gray.shape[1])
    return gray[y0:y1, x0:x1]

fig3 = plt.figure(figsize=(10, 10))
for i, (name, (x, y)) in enumerate(samples.items(), 1):
    patch = safe_patch(y, x)
    ax3d  = fig3.add_subplot(2, 2, i, projection="3d")
    X, Y  = np.meshgrid(range(patch.shape[1]), range(patch.shape[0]))
    ax3d.plot_surface(X, Y, patch, rstride=1, cstride=1,
                      linewidth=0, antialiased=False, cmap="viridis")
    ax3d.set_title(f"{labels[name]} patch")
    ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([])

fig3.suptitle("3-D grayscale surfaces of typical patches")
plt.show()
