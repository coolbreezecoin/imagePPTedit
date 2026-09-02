"""文字区 zone → 墨迹掩膜、行切分、墨迹盒测量（规范 §4.2/4.3/4.4/4.6, §5.1/5.2）。"""
import numpy as np, cv2
from .common import S, odd, clip_box, gray_of, char_count, is_cjk_char, rgb2hex, line_text, line_runs

def envelope_mask(gray, polarity, k, rel=0.11, abs_min=5):
    """形态学包络：亮字 OPEN，暗字 CLOSE；返回 (mask(bool), d(float), span)。自动抬阈值防掩膜过密。"""
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    g = gray.astype(np.float32)
    if polarity == "light":
        env = cv2.morphologyEx(g, cv2.MORPH_OPEN, ker); d = g - env
    else:
        env = cv2.morphologyEx(g, cv2.MORPH_CLOSE, ker); d = env - g
    d = np.clip(d, 0, None)
    span = max(np.percentile(d, 99.5), 1.0)
    r = rel
    while True:
        mask = (d / span > r) & (d > abs_min)
        if mask.mean() <= 0.40 or r >= 0.45: break
        r += 0.05
    return mask, d, span

def polarity_scores(gray, k):
    """§4.2：腐蚀细度判据。返回 {pol: (score, thin, frac)}"""
    out = {}
    for pol in ("light", "dark"):
        m, _, _ = envelope_mask(gray, pol, k)
        m8 = m.astype(np.uint8)
        area = m8.sum()
        if area < 20: out[pol] = (-1, 0, 0); continue
        er = cv2.erode(m8, np.ones((odd(S(9)), odd(S(9))), np.uint8))   # 4K 下 9x9 → 等比缩放
        thin = 1 - er.sum() / area
        frac = m.mean()
        out[pol] = (thin - 0.5 * frac, thin, frac)
    return out

def uniform_patch_mask(gray, polarity):
    """§5.2 色块上的文字：相对色片自身的百分位阈值。"""
    g = gray.astype(np.float32)
    if polarity == "light":
        p70, p995 = np.percentile(g, 70), np.percentile(g, 99.5)
        return g > p70 + 0.45 * (p995 - p70)
    else:
        p30, p05 = np.percentile(g, 30), np.percentile(g, 0.5)
        return g < p30 - 0.45 * (p30 - p05)

def _region_fill(reg):
    """把区域内部的洞（文字）填上：反相连通域中不接触边界的都并入。"""
    inv = (1 - reg).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 4)
    H, W = reg.shape; out = reg.copy()
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if not (x == 0 or y == 0 or x + w >= W or y + h >= H):
            out[lab == i] = 1
    return out

def uniform_patch_mask_region(gray, polarity, region=None):
    """§5.2 色片内百分位阈值（可选限定在 region 内统计）。"""
    g = gray.astype(np.float32)
    sel = g[region] if region is not None and region.sum() > 50 else g
    if polarity == "light":
        p70, p995 = np.percentile(sel, 70), np.percentile(sel, 99.5)
        return g > p70 + 0.45 * (p995 - p70)
    p30, p05 = np.percentile(sel, 30), np.percentile(sel, 0.5)
    return g < p30 - 0.45 * (p30 - p05)

def split_rows(mask, n_expect=None, single=False, merge_gap=None, min_row_override=None):
    """§4.3 行投影切分。返回 [(y0,y1), ...]（相对 mask 的行号，y1 不含）。"""
    H = mask.shape[0]
    rows = mask.sum(axis=1).astype(np.float32)
    if single:
        ys = np.where(rows > 0)[0]
        return [(int(ys.min()), int(ys.max()) + 1)] if len(ys) else []
    base = np.percentile(rows, 10)
    min_row = max(3, base + max(3, 0.08 * (rows.max() - base)))
    if min_row_override is not None: min_row = min_row_override
    on = rows >= min_row
    segs = []; y = 0
    while y < H:
        if on[y]:
            y0 = y
            while y < H and on[y]: y += 1
            segs.append([y0, y])
        else: y += 1
    if not segs: return []
    # 合并小间隙
    med_h = np.median([s[1]-s[0] for s in segs])
    if merge_gap is None: merge_gap = max(3, 0.25 * med_h)
    merged = [segs[0]]
    for s in segs[1:]:
        if s[0] - merged[-1][1] <= merge_gap: merged[-1][1] = s[1]
        else: merged.append(s)
    segs = merged
    # 按期望行数纠偏：多了 → 合并最小间隙；少了 → 在最高的段内部按投影最小值切分
    if n_expect:
        while len(segs) > n_expect:
            gaps = [(segs[i+1][0]-segs[i][1], i) for i in range(len(segs)-1)]
            g, i = min(gaps); segs[i][1] = segs[i+1][1]; del segs[i+1]
        # 去掉明显过小的噪声段（高度 < 0.3 中位）后若仍多于期望已处理；少于期望时尝试切分
        tries = 0
        while len(segs) < n_expect and tries < 10:
            tries += 1
            hs = [s[1]-s[0] for s in segs]; i = int(np.argmax(hs))
            y0, y1 = segs[i]
            if y1 - y0 < 8: break
            sub = rows[y0+3:y1-3]
            if len(sub) == 0: break
            j = int(np.argmin(sub)) + y0 + 3
            segs[i:i+1] = [[y0, j], [j, y1]]
    return [(int(a), int(b)) for a, b in segs]

def line_hextent(mask_rows, frac=0.04, line_h=None, prior=None):
    """行的水平墨迹范围。先按列间隙聚类、丢掉远离主团的小碎片（如邻近的亮星/装饰点）；
    prior=(x0,x1) 为 OCR 先验窗口（已含外扩），聚类结果优先取与先验重叠的。"""
    col = mask_rows.sum(axis=0).astype(np.float32)
    if col.max() == 0: return None
    xs = np.where(col >= frac * col.max())[0]
    if line_h is None: line_h = mask_rows.shape[0]
    gap_thr = max(1.2 * line_h, 6)
    # 聚类
    clusters = []; start = xs[0]; prev = xs[0]
    for x in xs[1:]:
        if x - prev > gap_thr:
            clusters.append((start, prev + 1)); start = x
        prev = x
    clusters.append((start, prev + 1))
    mass = [col[a:b].sum() for a, b in clusters]
    if prior is not None:
        ov = [min(b, prior[1]) - max(a, prior[0]) for a, b in clusters]
        keep = [i for i, o in enumerate(ov) if o > 0]
        if keep:
            clusters = [clusters[i] for i in keep]; mass = [mass[i] for i in keep]
    mmax = max(mass)
    # 只丢"又小又窄"的碎片（亮星/噪点）：质量 < 3% 且宽度 < 0.6 行高；有 OCR 先验时窗口内的一律保留
    if prior is not None:
        keep = list(range(len(clusters)))
    else:
        keep = [i for i, m in enumerate(mass) if not (m < 0.03 * mmax and (clusters[i][1] - clusters[i][0]) < 0.6 * line_h)]
    # 连续性：只保留包含最大团的连续块（中间不允许隔着被丢弃的团）
    imax = int(np.argmax(mass)); lo = imax; hi = imax
    while lo - 1 in keep: lo -= 1
    while hi + 1 in keep: hi += 1
    return int(clusters[lo][0]), int(clusters[hi][1])

def detect_glow(gray, mask, polarity):
    """§4.6 外发光检测。返回 (has_glow, radius_px, excess)"""
    core = cv2.erode(mask.astype(np.uint8), np.ones((3,3),np.uint8))
    if core.sum() < 10: return False, 0, 0
    dist = cv2.distanceTransform((1 - core).astype(np.uint8), cv2.DIST_L2, 3)
    g = gray.astype(np.float32)
    sgn = 1 if polarity == "light" else -1
    far = g[dist > S(90)] if (dist > S(90)).any() else g[dist > dist.max()*0.8]
    if far.size == 0: return False, 0, 0
    base = np.median(far)
    def ring(a, b):
        sel = (dist >= a) & (dist < b) & (~mask)
        return (np.median(g[sel]) - base) * sgn if sel.sum() > 20 else 0
    near = ring(S(4), S(8))
    if near <= 15: return False, 0, 0
    peak = near; r = S(6)
    while r < S(200):
        v = ring(r, r + S(4))
        if v < 0.15 * peak: break
        r += S(4)
    span = r - S(6)
    return (span > S(20)), float(r), float(near)

def text_color(rgb, d, span, mask):
    core = (d / span > 0.6) & mask
    if core.sum() < 10: core = mask
    if core.sum() == 0: return None
    med = np.median(rgb[core].reshape(-1, 3), axis=0)
    return rgb2hex(med)

def analyze_zone(img_rgb, zone, container_kind="bg"):
    """返回 dict：polarity, lines[{bbox,text,n,advance,ratio,ok,mask_box}], mask(全图bool), color, glow"""
    W, H = img_rgb.shape[1], img_rgb.shape[0]
    x0, y0, x1, y1 = clip_box(zone["bbox"], W, H)
    n_lines = len(zone["lines"]); single = zone.get("line_mode") == "single"
    # OCR 锚定齐全时，行几何以 OCR 行框为准：密排排版下标注框常装着"一行半"，
    # 形态学切行会切错带、行心落到行缝上（多块文字坍缩挤行之祸）。裁剪窗扩到框∪OCR带。
    _priors = zone.get("_line_priors")
    _priors_ok = bool(_priors) and len(_priors) == n_lines and all(p is not None for p in _priors)
    if _priors_ok:
        x0 = max(0, min(x0, int(min(p[0] for p in _priors)) - 4))
        y0 = max(0, min(y0, int(min(p[1] for p in _priors)) - 4))
        x1 = min(W, max(x1, int(max(p[2] for p in _priors)) + 4))
        y1 = min(H, max(y1, int(max(p[3] for p in _priors)) + 4))
    crop = img_rgb[y0:y1, x0:x1]; gray = gray_of(crop)
    line_h_est = (y1 - y0) / max(n_lines, 1)
    if _priors_ok:
        line_h_est = float(np.median([p[3] - p[1] for p in _priors]))
    k = odd(np.clip(0.28 * line_h_est, 5, 15))
    # 极性
    sc = polarity_scores(gray, k)
    pol_auto = max(sc, key=lambda p: sc[p][0])
    pol = zone.get("polarity", pol_auto)
    pol_warn = (pol != pol_auto) and abs(sc["light"][0] - sc["dark"][0]) > 0.15
    if zone.get("polarity") is None: pol = pol_auto
    # 底色亮度是极性的主判据：底暗字必亮、底亮字必暗；中间地带才信 标注/评分
    # （AI 标注和 polarity_scores 都在"深色卡上的白字"上一致翻过车——测出 #0B0C0F 的"黑墨"）。
    # 判底色用"多数派"：文字永远是少数派像素，Otsu 二分后占多数的一侧就是底
    # （曾用边缘环中值，被胶囊里的白图标+出格笔画污染判反——"欢迎咨询"白字被当黑墨，擦除留下重影）。
    # 只对足够大的块用：页码小框样本太少不可靠
    if gray.shape[0] >= 20 and gray.shape[1] >= 20 and gray.size >= 1500:
        g8 = gray.astype(np.uint8)
        thr_o, _ = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        frac_hi = float((g8 > thr_o).mean())
        if frac_hi < 0.42: pol = "light"      # 亮像素是少数派 → 底暗 → 亮字
        elif frac_hi > 0.58: pol = "dark"     # 暗像素是少数派 → 底亮 → 暗字

    uniform = container_kind in ("capsule", "cell", "box", "badge") or zone.get("mask_mode") == "uniform"
    patch_region = None
    if uniform:
        # §5.2：先把 zone 收缩到色片内部（与中位色差 < 42 的最大连通域，内缩 10%），再用色片百分位阈值
        med = np.median(crop.reshape(-1, 3), axis=0)
        near = (np.abs(crop.astype(np.float32) - med).max(axis=2) < 42).astype(np.uint8)
        nn, lab, stats, _ = cv2.connectedComponentsWithStats(near, 8)
        if nn > 1:
            i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            reg = (lab == i).astype(np.uint8)
            er = max(2, int(0.10 * min(reg.shape)))
            reg = cv2.erode(reg, np.ones((er, er), np.uint8))
            # 把文字挖掉的洞补回来（文字像素不属于 near，但在色片内部）
            reg = _region_fill(reg)
            patch_region = reg.astype(bool)
        g_in = gray[patch_region] if patch_region is not None and patch_region.sum() > 50 else gray
        gm = np.full(gray.shape, np.median(g_in), np.float32); gm[patch_region] = gray[patch_region]
        mask = uniform_patch_mask_region(gray, pol, patch_region)
        _, d, span = envelope_mask(gray, pol, k)
        if patch_region is not None: mask &= patch_region
    else:
        mask, d, span = envelope_mask(gray, pol, k)
    # 去掉极小噪点
    m8 = mask.astype(np.uint8)
    ncc, lab, stats, _ = cv2.connectedComponentsWithStats(m8, 8)
    small = stats[:, cv2.CC_STAT_AREA] < max(2, S(6))
    small[0] = False
    mask[np.isin(lab, np.where(small)[0])] = False
    # 行切分
    merge_gap = None
    if zone.get("line_mode") == "pinyin": merge_gap = S(22)
    if _priors_ok:   # 每行在自己的 OCR 带内测量，行序与 lines 一一对应
        segs = [(max(0, int(p[1]) - y0 - 2), min(y1 - y0, int(p[3]) - y0 + 2)) for p in _priors]
    else:
        segs = split_rows(mask, n_expect=n_lines, single=single, merge_gap=merge_gap,
                      min_row_override=(3 if single else None))
    lines = []
    priors = zone.get("_line_priors")   # 可选：每行 OCR 框 [x0,y0,x1,y1]（全图坐标）
    for i, (ya, yb) in enumerate(segs):
        pr = None
        if priors and i < len(priors) and priors[i] is not None:
            em = max(yb - ya, 8)
            pr = (priors[i][0] - x0 - 0.8 * em, priors[i][2] - x0 + 0.8 * em)
        ext = line_hextent(mask[ya:yb], line_h=(yb - ya), prior=pr)
        if ext is None: continue
        xa, xb = ext
        if _priors_ok:   # OCR 带只负责切行；字高按带内实际墨迹行收紧（OCR 框自带 ~30% 松边，直接抄会虚高）
            _ri = mask[ya:yb, xa:xb].sum(axis=1)
            _nz = np.where(_ri > 0)[0]
            if len(_nz) >= 3: ya, yb = ya + int(_nz.min()), ya + int(_nz.max()) + 1
        text = line_text(zone["lines"][i]) if i < n_lines else ""
        n = char_count(text)
        cjk = sum(is_cjk_char(c) for c in text) >= max(2, 0.6 * len(text.replace(" ", "")))
        ink_w, ink_h = xb - xa, yb - ya
        adv = ink_w / n if n > 0 else 0
        ratio = ink_h / adv if adv > 0 else 0
        ok = (0.62 <= ratio <= 1.15) if (cjk and n >= 2) else True
        entry = {"idx": i, "text": text, "bbox": [x0 + xa, y0 + ya, x0 + xb, y0 + yb],
                 "n": n, "advance": round(adv, 2), "ratio": round(ratio, 3), "ok": bool(ok), "cjk": bool(cjk)}
        if _priors_ok: entry["geom_src"] = "ocr_band"   # 行几何来自 OCR 带：字号可按行高直配
        # 暗底小灰字掩膜常残缺：墨迹行框明显小于 OCR 行框 → 几何回退 OCR 框（防标定
        # "自洽收敛到小目标"、成品字变小），并在 OCR 框内放宽阈值补掩膜（防擦不净留残影）
        if priors and i < len(priors) and priors[i] is not None:
            pb = priors[i]; ph, pw = pb[3] - pb[1], pb[2] - pb[0]
            if ph > 6 and pw > 6 and (ink_h < 0.70 * ph or ink_w < 0.70 * pw):
                entry["bbox"] = [int(pb[0]), int(pb[1]), int(pb[2]), int(pb[3])]
                entry["advance"] = round(pw / n, 2) if n else entry["advance"]
                entry["ratio"] = round(ph / max(1.0, pw / n), 3) if n else entry["ratio"]
                entry["ok"] = True; entry["geom_src"] = "ocr"
                pya, pyb = max(0, int(pb[1] - y0)), min(mask.shape[0], int(pb[3] - y0))
                pxa, pxb = max(0, int(pb[0] - x0)), min(mask.shape[1], int(pb[2] - x0))
                if pyb > pya and pxb > pxa:
                    mask[pya:pyb, pxa:pxb] |= (d[pya:pyb, pxa:pxb] > 0.30 * span)
        runs = line_runs(zone["lines"][i]) if i < n_lines else [{"t": text}]
        if len(runs) > 1 and adv > 0:
            cols = []; cum = 0.0
            for r in runs:
                rn = char_count(r["t"]); rx0 = xa + cum * adv; rx1 = xa + (cum + rn) * adv; cum += rn
                sub = mask[ya:yb, int(rx0):int(max(rx0 + 1, rx1))]; dsub = d[ya:yb, int(rx0):int(max(rx0 + 1, rx1))]
                core = (dsub / span > 0.6) & sub
                if core.sum() < 5: core = sub
                c = np.median(crop[ya:yb, int(rx0):int(max(rx0 + 1, rx1))][core].reshape(-1, 3), axis=0) if core.sum() else None
                cols.append(rgb2hex(c) if c is not None else None)
            entry["run_colors"] = cols
        lines.append(entry)
    full = np.zeros((H, W), bool); full[y0:y1, x0:x1] = mask
    dfull = np.zeros((H, W), np.float32); dfull[y0:y1, x0:x1] = d
    if uniform or container_kind in ("capsule", "cell"):
        glow = (False, 0, 0); outline = None
    else:
        glow = detect_glow(gray, mask, pol)
        outline = detect_outline(crop, gray, mask, pol)
    return {"outline": outline, "polarity": pol, "polarity_auto": pol_auto, "polarity_scores": {p: [round(v, 3) for v in sc[p]] for p in sc},
            "polarity_warn": bool(pol_warn), "k": k, "span": float(span), "mask_frac": float(mask.mean()),
            "lines": lines, "n_expected": n_lines, "n_found": len(lines),
            "color": text_color(crop, d, span, mask), "glow": {"has": bool(glow[0]), "radius": round(glow[1], 1), "excess": round(glow[2], 1)},
            "mask": full, "d": dfull}


def detect_outline(rgb, gray, mask, polarity):
    """检测与文字极性相反的窄描边/硬阴影（如白字带深色细边）。返回 None 或 {"color":hex,"width_px":w,"strength":0..1}"""
    core = mask.astype(np.uint8)
    if core.sum() < 20: return None
    dist = cv2.distanceTransform((1 - core).astype(np.uint8), cv2.DIST_L2, 3)
    g = gray.astype(np.float32)
    far = (dist > 10) & (dist < 60)
    if far.sum() < 50: return None
    base = np.median(g[far])
    sgn = 1 if polarity == "light" else -1      # 亮字：描边应更暗 → (ring-base)*sgn < 0
    def ring(a, b):
        sel = (dist >= a) & (dist < b)
        return (np.median(g[sel]) - base) * sgn if sel.sum() > 20 else 0, sel
    v1, sel1 = ring(1, 2.5)
    # 相对判据：环带比远处背景更"反向"至少 8 级灰度且 ≥ 25% 的背景余量（深底上的暗描边绝对差很小）
    headroom = base if polarity == "light" else (255 - base)
    if v1 > -max(8, 0.25 * headroom): return None
    w = 2.5
    for b in (3.5, 4.5, 5.5, 6.5):
        v, _ = ring(b - 1, b)
        if v < 0.5 * v1: w = b       # 半峰宽
        else: break
    col = np.median(rgb[sel1].reshape(-1, 3), axis=0)
    strength = float(np.clip(-v1 / 40.0, 0, 1))
    return {"color": rgb2hex(col), "width_px": float(w), "strength": strength}
