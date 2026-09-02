"""文字擦除（规范 §5.3/§5.4/§4.6）：逐行掩膜 + 硬裁剪 + 自适应膨胀；LaMa 修补或色块中位色重涂。"""
import numpy as np, cv2
from .common import S, odd
from .inpaint import inpaint_region

def line_erase_mask(d_full, mask_full, line_bbox, polarity, span, glow_r=0.0, uniform=False):
    """返回该行的擦除掩膜（全图尺寸 bool）。"""
    H, W = mask_full.shape
    x0, y0, x1, y1 = line_bbox
    lh = max(y1 - y0, 6)
    pad = int(round(S(16)))
    wx0, wy0, wx1, wy1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    m = mask_full[wy0:wy1, wx0:wx1]
    d = d_full[wy0:wy1, wx0:wx1]
    # 硬裁剪边界：高阈值掩膜的包围盒 + 8px（缩放）
    hi = (d / max(span, 1) > 0.45) & m
    if hi.sum() == 0: hi = m
    ys, xs = np.where(hi)
    if len(ys) == 0:
        return np.zeros((H, W), bool)
    p8 = int(round(S(8)))
    cx0, cy0, cx1, cy1 = max(0, xs.min() - p8), max(0, ys.min() - p8), min(m.shape[1], xs.max() + 1 + p8), min(m.shape[0], ys.max() + 1 + p8)
    # 膨胀半径自适应（4K 下 7~21 → 缩放）
    r = int(np.clip(round(0.16 * lh), round(S(7)), round(S(21))))
    r = max(r, 2)
    if glow_r > 0: r = int(r + glow_r)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    dil = cv2.dilate(m.astype(np.uint8), ker).astype(bool)
    clip = np.zeros_like(dil); clip[cy0:cy1, cx0:cx1] = True
    dil &= clip
    out = np.zeros((H, W), bool); out[wy0:wy1, wx0:wx1] = dil
    return out

def erase_zone(img_rgb, zinfo, uniform_fill=False, margin=None, dry_run=False):
    """zinfo 来自 ink.analyze_zone（含 mask,d,span,lines,polarity,glow）。返回 (新图, 擦除掩膜)。"""
    H, W = img_rgb.shape[:2]
    total = np.zeros((H, W), bool)
    glow_r = zinfo["glow"]["radius"] if zinfo["glow"]["has"] else 0.0
    for L in zinfo["lines"]:
        total |= line_erase_mask(zinfo["d"], zinfo["mask"], L["bbox"], zinfo["polarity"], zinfo["span"], glow_r)
    if total.sum() == 0 or dry_run:
        return img_rgb, total
    if uniform_fill:
        # 色块中位色重涂：取掩膜外扩环带里的像素中位色
        ring = cv2.dilate(total.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~total
        med = np.median(img_rgb[ring].reshape(-1, 3), axis=0)
        out = img_rgb.copy(); out[total] = med
        return out, total
    if margin is None:
        lh = np.median([L["bbox"][3] - L["bbox"][1] for L in zinfo["lines"]])
        margin = int(np.clip(6 * lh, 160, 320))
    out = inpaint_region(img_rgb, total.astype(np.uint8), margin=margin)
    return out, total
