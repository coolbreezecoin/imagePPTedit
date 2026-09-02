"""图层抠取：书法艺术字(§6)、底板/卡片/胶囊、人物/物品(§8)。"""
import numpy as np, cv2
from .common import S, odd, clip_box, gray_of
from .inpaint import inpaint_region

def _stroke_width(mask, q=95):
    """笔画宽度：取距离变换的高分位×2（≈最粗笔画宽），毛笔字粗细差异大时以粗笔为准。"""
    m = mask.astype(np.uint8)
    if m.sum() < 10: return 3.0
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    return float(max(2.0, 2.0 * np.percentile(dist[m > 0], q)))

def _fill_enclosed_holes(alpha_bin, max_frac=0.02):
    """只补"真正封闭"的孔：不接触边界且面积 < 区域 2%。"""
    inv = (~alpha_bin).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 4)
    H, W = alpha_bin.shape; out = alpha_bin.copy(); area_all = H * W
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        touches = x == 0 or y == 0 or x + w >= W or y + h >= H
        if not touches and a < max_frac * area_all:
            out[lab == i] = True
    return out

def calligraphy_matte(img_rgb, bbox, polarity, pad=None):
    """返回 dict: rgba(HxWx4 uint8, 紧裁剪), box(紧包围盒 全图坐标), alpha_full(HxW float)"""
    H, W = img_rgb.shape[:2]
    x0, y0, x1, y1 = clip_box(bbox, W, H)
    if pad is None: pad = int(round(0.08 * (y1 - y0)))
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    crop = img_rgb[Y0:Y1, X0:X1]; g = gray_of(crop).astype(np.float32)
    sgn = 1.0 if polarity == "light" else -1.0
    # 第一轮：估计笔画宽度（核取大一些，保证最粗笔画也能整体进入掩膜）
    k0 = odd(np.clip(0.35 * (y1 - y0), 9, 61))
    ker0 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k0, k0))
    env0 = cv2.morphologyEx(g, cv2.MORPH_OPEN if sgn > 0 else cv2.MORPH_CLOSE, ker0)
    d0 = np.clip((g - env0) * sgn, 0, None); m0 = d0 > 0.35 * max(np.percentile(d0, 99.5), 1)
    sw = min(_stroke_width(m0), 0.35 * (y1 - y0))
    # 第二轮：核 ≈ 1.3×最粗笔画宽（必须大于最粗笔画，否则粗笔内部会出现空洞）
    k = odd(np.clip(1.3 * sw + 2, 7, 0.7 * (y1 - y0)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    env = cv2.morphologyEx(g, cv2.MORPH_OPEN if sgn > 0 else cv2.MORPH_CLOSE, ker)
    d = np.clip((g - env) * sgn, 0, None)
    span = max(np.percentile(d, 99.5), 1)
    # 相对"局部笔画峰值"归一化：光晕相对邻近笔画很弱，会被地板压掉；纯笔画内部仍接近 1
    peak = cv2.dilate(d, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(2.5 * sw),) * 2))
    peak = cv2.GaussianBlur(peak, (0, 0), sw)
    alpha = np.clip(d / np.maximum(peak, 0.6 * span), 0, 1)
    alpha = np.clip((alpha - 0.22) / 0.78, 0, 1)                       # 抬地板
    # 丢掉"整体偏暗"的孤立连通域（装饰线、星点、底板边缘漏光）：峰值 < 0.45 span
    lab_n, lab, stats, _ = cv2.connectedComponentsWithStats((alpha > 0.3).astype(np.uint8), 8)
    if lab_n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]; amax = areas.max()
        for i in range(1, lab_n):
            comp = lab == i
            if d[comp].max() < 0.45 * span or stats[i, cv2.CC_STAT_AREA] < 0.004 * amax:
                alpha[comp] = 0
    # 去细线：只滤掉比"细笔画"还细的结构（田字格线/描边），保住飞白与撇捺细尾
    thin_sw = max(2.0, _stroke_width(alpha > 0.5, q=40))
    core = cv2.morphologyEx((alpha > 0.5).astype(np.uint8), cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(max(3, thin_sw * 0.6)),) * 2))
    keep = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(2.0 * sw),) * 2)).astype(np.float32)
    keep = cv2.GaussianBlur(keep, (0, 0), max(1.0, sw * 0.5))
    alpha = alpha * np.clip(keep, 0, 1)
    alpha = alpha ** 1.3
    alpha[alpha < 0.06] = 0          # 硬地板：星空/纹理背景漏进来的微弱 alpha 一律清零
    binm = _fill_enclosed_holes(alpha > 0.5, max_frac=0.05)
    alpha = np.maximum(alpha, binm.astype(np.float32) * 0.999 * (alpha > 0.5).astype(np.float32) + binm.astype(np.float32) * (alpha <= 0.5) * 0.9)
    # 墨色：alpha>0.6 取真实颜色，其余大尺度加权模糊外推
    w = (alpha > 0.6).astype(np.float32)
    rgbf = crop.astype(np.float32)
    sig = max(3.0, sw * 2)
    num = np.stack([cv2.GaussianBlur(rgbf[..., c] * w, (0, 0), sig) for c in range(3)], -1)
    den = cv2.GaussianBlur(w, (0, 0), sig)[..., None] + 1e-6
    prop = num / den
    # 若加权模糊仍覆盖不到（den 很小），再用更大尺度
    far = den[..., 0] < 0.02
    if far.any():
        num2 = np.stack([cv2.GaussianBlur(rgbf[..., c] * w, (0, 0), sig * 4) for c in range(3)], -1)
        den2 = cv2.GaussianBlur(w, (0, 0), sig * 4)[..., None] + 1e-6
        prop[far] = (num2 / den2)[far]
    color = np.where(w[..., None] > 0, rgbf, prop)
    # 紧包围盒
    ys, xs = np.where(alpha > 0.05)
    if len(ys) == 0:
        return None
    ty0, ty1, tx0, tx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    rgba = np.dstack([np.clip(color, 0, 255), alpha[..., None] * 255]).astype(np.uint8)[ty0:ty1, tx0:tx1]
    alpha_full = np.zeros((H, W), np.float32); alpha_full[Y0:Y1, X0:X1] = alpha
    return {"rgba": rgba, "box": [X0 + tx0, Y0 + ty0, X0 + tx1, Y0 + ty1], "alpha_full": alpha_full, "stroke_w": sw}

def panel_mask(img_rgb, bbox, soft=True, thr=None):
    """卡片/色块/徽章：按与外围环带中位色的色差分割。返回 alpha_full(float 0..1) 与紧包围盒。"""
    H, W = img_rgb.shape[:2]
    x0, y0, x1, y1 = clip_box(bbox, W, H)
    p = int(round(S(10)))
    X0, Y0, X1, Y1 = max(0, x0 - p), max(0, y0 - p), min(W, x1 + p), min(H, y1 + p)
    crop = img_rgb[Y0:Y1, X0:X1].astype(np.float32)
    ring = np.zeros(crop.shape[:2], bool); ring[:] = True
    ring[(y0 - Y0):(y1 - Y0), (x0 - X0):(x1 - X0)] = False
    if ring.sum() < 50:
        ring[:] = False; ring[:2] = True; ring[-2:] = True; ring[:, :2] = True; ring[:, -2:] = True
    bgc = np.median(crop[ring].reshape(-1, 3), axis=0)
    diff = np.abs(crop - bgc).max(axis=2)
    if thr is None:
        # 最近颜色分割：取 bbox 中心 50% 区域的中位色作为底板色，阈值 = 与外围色差的一半（6~24）
        cy0, cy1 = (y0 - Y0) + (y1 - y0) // 4, (y0 - Y0) + 3 * (y1 - y0) // 4
        cx0, cx1 = (x0 - X0) + (x1 - x0) // 4, (x0 - X0) + 3 * (x1 - x0) // 4
        pc = np.median(crop[cy0:cy1, cx0:cx1].reshape(-1, 3), axis=0)
        contrast = float(np.abs(pc - bgc).max())
        thr = float(np.clip(0.5 * contrast, 6, 18))
    m = (diff > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    # 填孔 + 取最大连通域（含触边）
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == i)
    m = _fill_enclosed_holes(m, max_frac=0.9)
    alpha = m.astype(np.float32)
    if soft:
        # 软边：色差在 [thr*0.5, thr*1.5] 之间线性，限制在掩膜 1px 邻域内
        band = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~m
        alpha[band] = np.clip((diff[band] - 0.5 * thr) / (thr + 1e-6), 0, 1) * 0.8
    ys, xs = np.where(alpha > 0.02)
    if len(ys) == 0: return None
    box = [X0 + xs.min(), Y0 + ys.min(), X0 + xs.max() + 1, Y0 + ys.max() + 1]
    full = np.zeros((H, W), np.float32); full[Y0:Y1, X0:X1] = alpha
    return {"alpha_full": full, "box": box, "bg_color": bgc, "thr": thr}

_SESS = {}
def _session(name):
    from rembg import new_session
    if name not in _SESS: _SESS[name] = new_session(name)
    return _SESS[name]

def rembg_alpha(img_rgb, bbox, model="u2net_human_seg", margin_frac=0.12):
    """在 bbox 外扩后的裁剪上跑 rembg，返回全图 alpha(float 0..1)。"""
    from rembg import remove
    from PIL import Image
    H, W = img_rgb.shape[:2]
    x0, y0, x1, y1 = clip_box(bbox, W, H)
    mx, my = int((x1 - x0) * margin_frac), int((y1 - y0) * margin_frac)
    X0, Y0, X1, Y1 = max(0, x0 - mx), max(0, y0 - my), min(W, x1 + mx), min(H, y1 + my)
    crop = Image.fromarray(img_rgb[Y0:Y1, X0:X1])
    m = remove(crop, session=_session(model), only_mask=True)
    a = np.asarray(m).astype(np.float32) / 255.0
    full = np.zeros((H, W), np.float32); full[Y0:Y1, X0:X1] = a
    # 限制在原 bbox 略外扩范围内，避免模型抓到远处的东西
    lim = np.zeros((H, W), np.float32); lim[max(0, y0 - my // 2):min(H, y1 + my // 2), max(0, x0 - mx // 2):min(W, x1 + mx // 2)] = 1
    return full * lim

def alpha_to_rgba(img_rgb, alpha_full, thresh=0.01):
    ys, xs = np.where(alpha_full > thresh)
    if len(ys) == 0: return None, None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    a = alpha_full[y0:y1, x0:x1]
    rgba = np.dstack([img_rgb[y0:y1, x0:x1], (a * 255).astype(np.uint8)])
    return rgba, [int(x0), int(y0), int(x1), int(y1)]

def remove_and_fill(img_rgb, alpha_full, dilate_px=4, margin=256):
    m = (alpha_full > 0.05).astype(np.uint8)
    if dilate_px > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2))
    if m.sum() == 0: return img_rgb.copy()
    return inpaint_region(img_rgb, m, margin=margin)


def grabcut_alpha(img_rgb, bbox, exclude_alpha=None, seed_alpha=None, iters=5, margin_frac=0.06):
    """§8 平直大件（桌子等）：GrabCut，bbox 内为可能前景、bbox 外为背景；人物区域强制背景；显著性高置信处作为前景种子。"""
    H, W = img_rgb.shape[:2]
    x0, y0, x1, y1 = clip_box(bbox, W, H)
    mx, my = int((x1 - x0) * margin_frac) + 8, int((y1 - y0) * margin_frac) + 8
    X0, Y0, X1, Y1 = max(0, x0 - mx), max(0, y0 - my), min(W, x1 + mx), min(H, y1 + my)
    crop = cv2.cvtColor(img_rgb[Y0:Y1, X0:X1], cv2.COLOR_RGB2BGR)
    mask = np.full(crop.shape[:2], cv2.GC_BGD, np.uint8)
    mask[(y0 - Y0):(y1 - Y0), (x0 - X0):(x1 - X0)] = cv2.GC_PR_FGD
    # bbox 触及画面边缘的一侧：边缘不算背景（桌子常被画面裁切）
    if seed_alpha is not None:
        sa = seed_alpha[Y0:Y1, X0:X1]
        mask[(sa > 0.8) & (mask == cv2.GC_PR_FGD)] = cv2.GC_FGD
    if exclude_alpha is not None:
        ea = exclude_alpha[Y0:Y1, X0:X1]
        mask[ea > 0.3] = cv2.GC_BGD
    bgd = np.zeros((1, 65), np.float64); fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(crop, mask, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except Exception as e:
        print("grabcut failed", e); return None
    fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    # 清理：开运算去毛刺，保留大块
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n > 1:
        big = stats[1:, cv2.CC_STAT_AREA].max()
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 0.03 * big: fg[lab == i] = 0
    fg = _fill_enclosed_holes(fg.astype(bool), max_frac=0.05).astype(np.float32)
    # 软边 1px
    a = cv2.GaussianBlur(fg, (0, 0), 0.8)
    full = np.zeros((H, W), np.float32); full[Y0:Y1, X0:X1] = a
    return full


def _table_edge_line(g):
    """返回 (xa,ya,xb,yb) 或 None：近水平、较长、梯度强、位于 bbox 纵向 20%~85% 的线段。"""
    ch, cw = g.shape
    gb = cv2.GaussianBlur(g, (5, 5), 0)
    sob = np.abs(cv2.Sobel(gb, cv2.CV_32F, 0, 1, ksize=3))
    edges = cv2.Canny(gb, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=int(0.3 * cw), maxLineGap=30)
    best = None
    if lines is None: return None
    for l in np.asarray(lines).reshape(-1, 4):
        xa, ya, xb, yb = [int(v) for v in l]
        ang = abs(np.degrees(np.arctan2(yb - ya, xb - xa))); ang = min(ang, 180 - ang)
        if ang > 25: continue
        cy = (ya + yb) / 2
        if cy < 0.18 * ch or cy > 0.88 * ch: continue
        length = np.hypot(xb - xa, yb - ya)
        # 沿线梯度均值
        nsamp = int(length); xs = np.linspace(xa, xb, nsamp).astype(int); ys = np.linspace(ya, yb, nsamp).astype(int)
        grad = sob[np.clip(ys, 0, ch - 1), np.clip(xs, 0, cw - 1)].mean()
        score = length * grad
        if best is None or score > best[0]: best = (score, xa, ya, xb, yb)
    return None if best is None else best[1:]

def table_alpha(img_rgb, bbox, exclude_alpha=None, seed_alpha=None, iters=5):
    """§8 桌面类大件：桌沿线以下 + 显著物件 作为 GrabCut 的确定前景，bbox 内其余为可能前景，精修边界。"""
    H, W = img_rgb.shape[:2]
    x0, y0, x1, y1 = clip_box(bbox, W, H)
    pad = 8
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    crop_rgb = img_rgb[Y0:Y1, X0:X1]; g = gray_of(crop_rgb)
    ch, cw = g.shape
    ox, oy = x0 - X0, y0 - Y0
    mask = np.full((ch, cw), cv2.GC_BGD, np.uint8)
    mask[oy:oy + (y1 - y0), ox:ox + (x1 - x0)] = cv2.GC_PR_FGD
    line = _table_edge_line(g[oy:oy + (y1 - y0), ox:ox + (x1 - x0)])
    below = np.zeros((ch, cw), bool)
    if line is not None:
        xa, ya, xb, yb = line
        if xb == xa: xb += 1
        k = (yb - ya) / (xb - xa)
        for xi in range(x1 - x0):
            yl = int(np.clip(ya + k * (xi - xa), 0, y1 - y0 - 1))
            below[oy + yl:oy + (y1 - y0), ox + xi] = True
        sure = cv2.erode(below.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
        mask[sure] = cv2.GC_FGD
    sa = seed_alpha[Y0:Y1, X0:X1] if seed_alpha is not None else np.zeros((ch, cw), np.float32)
    # 保守初始化：只有"桌沿线以下" ∪ "显著物件邻域(膨胀 25px)" 算可能前景，bbox 内其余先算可能背景
    near_items = cv2.dilate((sa > 0.3).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))).astype(bool)
    pr = (mask == cv2.GC_PR_FGD)
    mask[pr & ~(below | near_items)] = cv2.GC_PR_BGD
    # 很亮的像素（雾化/白底/天空）且不在显著物件上、也不在桌沿线以下 → 可能背景
    bright = (g > 200) & (sa < 0.3) & (~below) & (mask == cv2.GC_PR_FGD)
    mask[bright] = cv2.GC_PR_BGD
    mask[(sa > 0.8) & (mask != cv2.GC_BGD)] = cv2.GC_FGD
    if exclude_alpha is not None:
        ea = exclude_alpha[Y0:Y1, X0:X1]; mask[ea > 0.3] = cv2.GC_BGD
    bgd = np.zeros((1, 65), np.float64); fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR), mask, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
        fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    except Exception as e:
        print("grabcut failed", e); fg = below.astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n > 1:
        big = stats[1:, cv2.CC_STAT_AREA].max()
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 0.03 * big: fg[lab == i] = 0
    fg = _fill_enclosed_holes(fg.astype(bool), max_frac=0.05).astype(np.float32)
    if exclude_alpha is not None:
        fg *= (1 - np.clip(exclude_alpha[Y0:Y1, X0:X1] * 1.5, 0, 1))
    a = cv2.GaussianBlur(fg, (0, 0), 0.8)
    full = np.zeros((H, W), np.float32); full[Y0:Y1, X0:X1] = a
    return full


def fill_with_median(img_rgb, alpha_full, region_box, dilate_px=2):
    """把 alpha 区域用 region_box 内（排除 alpha）像素的中位色重涂（书法压在色块/徽章上时用）。"""
    H, W = img_rgb.shape[:2]
    x0, y0, x1, y1 = clip_box(region_box, W, H)
    m = (alpha_full > 0.02).astype(np.uint8)
    if dilate_px > 0: m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2))
    sub = img_rgb[y0:y1, x0:x1]; sm = m[y0:y1, x0:x1] > 0
    ring = cv2.dilate(sm.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~sm
    if ring.sum() < 20: ring = ~sm
    med = np.median(sub[ring].reshape(-1, 3), axis=0)
    out = img_rgb.copy(); o = out[y0:y1, x0:x1]; o[sm] = med; out[y0:y1, x0:x1] = o
    return out


def unmix_translucent(img_rgb, bbox, dark=False, keep_bg_boxes=None, patch_mask=None, keep_bg_mask=None):
    """半透明底板解混：LaMa 修补板区域得到"板后的背景"，按 I = a·C + (1-a)·B 反解
    板透明度 a 与板色 C（单色假设）。返回 {rgba, box, bg}——bg 为板已移除的全图。
    背景恒等式反解：bg = (I - a·C)/(1-a)，贴回板后逐像素等于原图，杜绝"信源切换"接缝；
    只有浓雾(a>0.85, 反解不稳)与文字擦除渣区（keep_bg_mask 笔画级掩膜，缺省退回 keep_bg_boxes 盒级）羽化换用修补底。"""
    import numpy as np, cv2
    H, W = img_rgb.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(W, x1), min(H, y1)
    if x1 - x0 < 20 or y1 - y0 < 20: return None
    mask = np.zeros((H, W), np.float32); mask[y0:y1, x0:x1] = 1.0
    def _boxes_mask(boxes):
        m_ = np.zeros((H, W), np.float32)
        for kb in (boxes or []):
            kx0, ky0 = max(0, int(kb[0]) - 4), max(0, int(kb[1]) - 4)
            kx1, ky1 = min(W, int(kb[2]) + 4), min(H, int(kb[3]) + 4)
            m_[ky0:ky1, kx0:kx1] = 1.0
        return m_
    if keep_bg_mask is not None:                         # 笔画级掩膜（清洁板 textmask）：接缝只沿细笔画走
        kbm = cv2.dilate((keep_bg_mask > 0.2).astype(np.float32), np.ones((5, 5), np.uint8))
    else:
        kbm = _boxes_mask(keep_bg_boxes)                 # 兜底：盒级（会有矩形接缝，仅当无 textmask）
    zbm = _boxes_mask(keep_bg_boxes)                     # 文字区范围（漏擦兜底用）
    pbm = patch_mask if patch_mask is not None else np.zeros((H, W), np.float32)   # 笔画级修补掩膜：alpha 也不能信
    bg = remove_and_fill(img_rgb, mask, dilate_px=0, margin=340)
    I = img_rgb.astype(np.float32); B = bg.astype(np.float32)
    if dark:
        a = np.clip(((B - I) / np.maximum(B, 8.0)).max(axis=2), 0, 1)
    else:
        a = np.clip(((I - B) / np.maximum(255.0 - B, 8.0)).max(axis=2), 0, 1)
    a *= mask
    a = cv2.GaussianBlur(a, (7, 7), 0)
    a[a < 0.04] = 0
    if (a > 0.15).sum() < 0.02 * (x1 - x0) * (y1 - y0): return None
    if pbm.any():   # 笔画级修补区（擦书法）：alpha 不能全信实测——擦除残留（亮球）会被当成浓雾。
        am = ((pbm > 0.05) & (mask > 0)).astype(np.uint8)
        if am.any():
            a_meas = a.copy()
            a = cv2.inpaint((a * 255).astype(np.uint8), am, 15, cv2.INPAINT_NS).astype(np.float32) / 255.0
            # 书法周围常画有白晕（艺术光晕），擦除区内的真实雾理应≈周边晕强——
            # 取"插值为底、实测为准、周边晕局部最大+余量封顶"：亮球被压成晕，晕不丢
            ring = (cv2.dilate(am, np.ones((25, 25), np.uint8)) - am).astype(bool)
            ceil = cv2.dilate(np.where(ring, a_meas, 0.0), np.ones((61, 61), np.uint8))
            a = np.where(am > 0, np.maximum(a, np.minimum(a_meas, ceil + 0.05)), a)
            a = cv2.GaussianBlur(a, (5, 5), 0)
            a *= mask; a[a < 0.04] = 0
    # 板色：a 较实的像素上反解 C = (I - (1-a)B)/a 的中位色（修补渣像素不参与——原像素是补丁不是板）
    sel = (a > max(0.3, np.percentile(a[a > 0], 70) * 0.8)) & (kbm == 0)
    if sel.sum() < 50: sel = (a > 0.15) & (kbm == 0)
    if sel.sum() < 50: sel = a > 0.15
    aa = a[..., None]
    C = np.clip((I - (1 - aa) * B) / np.maximum(aa, 0.2), 0, 255)
    color = np.median(C[sel], axis=0)
    # 异物门控：单色半透明模型装不下的像素（板上压着的树叶/石灯/彩色辉光，或修补失真处）
    # 判据 = 按当前 (a, C, B) 重建与原像素的通道最大误差；两轮求色（首轮色被异物带偏时二轮修正）
    def _foreign(cc):
        err = np.abs(I - (aa * cc + (1 - aa) * B)).max(axis=2)
        return ((err > 20) & (mask > 0)).astype(np.uint8)
    f = _foreign(color)
    sel2 = sel & (f == 0)
    if sel2.sum() > 50:
        color = np.median(C[sel2], axis=0)
        f = _foreign(color)
    f = cv2.morphologyEx(f, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))       # 去孤立噪点
    # 文字区漏擦兜底：OCR 漏检的字形（长破折号等标点）没进擦除掩膜，恒等式反解会把它原样复现、
    # 与重排文字叠成双份——按"与修补底的墨色极性差"识别，并入渣区换修补底
    if zbm.any():
        _pol = (B - I).max(axis=2) if not dark else (I - B).max(axis=2)
        miss = ((_pol > 45) & (zbm > 0) & (mask > 0)).astype(np.uint8)
        if miss.any():
            miss = cv2.dilate(miss, np.ones((7, 7), np.uint8))
            kbm = np.maximum(kbm, miss.astype(np.float32))
    f[kbm > 0] = 0   # 修补渣区不做保真恢复——残留是擦除补丁，不是画面内容
    fw = cv2.GaussianBlur(cv2.dilate(f, np.ones((9, 9), np.uint8)).astype(np.float32), (0, 0), 5) * mask
    a *= (1 - fw); a[a < 0.04] = 0                                           # 异物处板层让路
    rgba = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
    rgba[..., :3] = color.astype(np.uint8)
    rgba[..., 3] = (a[y0:y1, x0:x1] * 255).astype(np.uint8)
    # 背景 = 恒等式反解 (I - a·C)/(1-a)：贴回板后与原图逐像素一致，异物/雾浓淡/求解误差全被吸收，
    # 不存在"信源切换"边界。例外羽化换修补底：①浓雾 a>0.85（除数过小，反解噪声放大；板色主导，
    # 换底几乎无感）②文字擦除渣（渣是补丁不是内容，恒等式会把它原样复现）
    aa2 = a[..., None]
    Bs = np.clip((I - aa2 * color) / np.maximum(1 - aa2, 0.15), 0, 255)
    w_dense = np.clip((a - 0.70) / 0.15, 0, 1)
    w_B = np.maximum(w_dense, cv2.GaussianBlur(kbm, (0, 0), 2.5)) * mask
    w_B = w_B[..., None]
    out = np.where(mask[..., None] > 0, Bs * (1 - w_B) + B * w_B, I).astype(np.uint8)
    fr = float((f > 0).sum()) / max(1.0, float((mask > 0).sum()))
    return {"rgba": rgba, "box": [x0, y0, x1, y1], "bg": out, "color": [int(v) for v in color],
            "foreign_frac": round(fr, 3)}


def art_text_matte(img_rgb, bbox, margin=240, bg_full=None):
    """特效美术字（立体/描边/渐变/发光）：不做笔画建模，按"与修补底的差异"整块抠出——
    层内是原像素，贴回所见即所得；描边/阴影/光效全部随层走。返回同 calligraphy_matte 形状 + bg。
    bg_full：预先算好的修补底（多块艺术字应传"并集一次修补"的结果——各框单独修补时，
    邻框的字会被 LaMa 当上下文"续写"出白色残带）。"""
    H, W = img_rgb.shape[:2]
    x0, y0, x1, y1 = clip_box(bbox, W, H)
    if x1 - x0 < 8 or y1 - y0 < 8: return None
    mask = np.zeros((H, W), np.float32); mask[y0:y1, x0:x1] = 1.0
    bg = bg_full if bg_full is not None else remove_and_fill(img_rgb, mask, dilate_px=0, margin=margin)
    diff = np.abs(img_rgb.astype(np.float32) - bg.astype(np.float32)).max(axis=2) * mask
    a = np.clip((diff - 10.0) / 30.0, 0, 1)          # 差异 10~40 平滑上道：文字/特效留住，修补噪声出局
    a = cv2.GaussianBlur(a, (5, 5), 0)
    m8 = (a > 0.35).astype(np.uint8)
    nn, lab, stats, _ = cv2.connectedComponentsWithStats(m8, 8)
    if nn > 1:
        big = stats[1:, cv2.CC_STAT_AREA].max()
        for i in range(1, nn):
            if stats[i, cv2.CC_STAT_AREA] < 0.004 * big: a[lab == i] = 0
    if (a > 0.5).sum() < 40: return None
    a = solidify_alpha(a, lo=0.12, hi=0.6)
    rgba, box = alpha_to_rgba(img_rgb, a)
    if rgba is None: return None
    sw = _stroke_width(a > 0.5)
    return {"rgba": rgba, "box": box, "alpha_full": a, "stroke_w": float(max(3.0, sw)), "bg": bg}


def solidify_alpha(a, lo=0.15, hi=0.75):
    """分割模型的软掩膜内部常不足 1.0，会让底下修补背景透出来；用 smoothstep 把内部压实、保留边缘柔和。"""
    t = np.clip((a - lo) / (hi - lo), 0, 1)
    return (t * t * (3 - 2 * t)).astype(np.float32)
