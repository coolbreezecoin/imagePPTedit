"""文字几何标定（规范 §4.5）：白底黑字标定稿 → PowerPoint 渲染 → 测墨迹盒 → 迭代 size/dx/dy/pitch。"""
import os, json, numpy as np, cv2
from .common import W0, H0, char_count, is_cjk_char, line_text
from .pptx_build import new_prs, blank_slide, add_textbox, export_pdf, render_pdf, PX2PT, PT2PX

def _cjk_line(text):
    t = text.strip(" 　")
    return sum(is_cjk_char(c) for c in t) >= max(1, 0.6 * len(t))

def init_params(zone_spec, zinfo):
    """从测得的墨迹行框给出初始参数（像素/磅）。"""
    lines = zinfo["lines"]; texts = [line_text(t) for t in zone_spec["lines"]]
    adv = []; hs = []
    for L in lines:
        t = texts[L["idx"]] if L["idx"] < len(texts) else ""
        n = char_count(t.strip(" 　"))
        w = L["bbox"][2] - L["bbox"][0]; h = L["bbox"][3] - L["bbox"][1]
        if _cjk_line(t) and n >= 2: adv.append(w / n)
        hs.append(h)
    ocrh = [L["bbox"][3] - L["bbox"][1] for L in lines if L.get("geom_src") == "ocr_band"]
    if ocrh:
        size_px = float(np.median(ocrh)) * 1.02   # 紧缩墨迹行高→字号直配（CJK 墨高≈字号×0.95，留少许余量）
    elif adv: size_px = float(np.median(adv))
    else:
        # 纯数字/字母：按高度估（大写/数字高 ≈ 0.72em）
        size_px = float(np.median(hs)) / 0.72
    cys = [(L["bbox"][1] + L["bbox"][3]) / 2 for L in lines]
    pitch_px = float(np.median(np.diff(cys))) if len(cys) >= 2 else size_px * 1.3
    pitch_pts = [float(g) * PX2PT for g in np.diff(cys)] if len(cys) >= 2 else []
    align = zone_spec.get("align", "left")
    x0s = [L["bbox"][0] for L in lines]; x1s = [L["bbox"][2] for L in lines]; cxs = [(a + b) / 2 for a, b in zip(x0s, x1s)]
    if align == "left": x_anchor = float(min(x0s)) - 0.06 * size_px
    elif align == "right": x_anchor = float(max(x1s)) + 0.06 * size_px
    else: x_anchor = float(np.mean(cxs))
    # 首行中心 y；文本框顶 = 首行中心 - pitch/2（迭代会修正）
    y_top = cys[0] - pitch_px / 2
    return {"size_pt": size_px * PX2PT, "pitch_pt": pitch_px * PX2PT, "pitch_pts": pitch_pts,
            "x_anchor_px": x_anchor, "y_top_px": float(y_top),
            "spc_pt": 0.0, "align": align, "n_lines": len(texts)}

def textbox_geom(params, lines):
    """由参数得到文本框 (x,y,w)。w 取足够宽。"""
    size_px = params["size_pt"] * PT2PX
    maxn = max(char_count(line_text(t)) for t in lines) if lines else 1
    w = maxn * size_px * 1.15 + 2 * size_px
    if params["align"] == "left": x = params["x_anchor_px"]
    elif params["align"] == "right": x = params["x_anchor_px"] - w
    else: x = params["x_anchor_px"] - w / 2
    return x, params["y_top_px"], w

def measure_lines(render_rgb, targets, size_px, pitch_px, others=None):
    """在渲染图上按目标行框附近测墨迹盒。targets: [[x0,y0,x1,y1],...]（同一 zone 的各行，按序）。
    others: 全页其它行的目标框——跨字块纵向裁剪用（卡片区行距极密、标注框互相重叠，
    邻行墨被量进来会把"字太高"误传给字号，出现级联缩字）。返回同长列表（可含 None）。"""
    g = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2GRAY)
    H, W = g.shape; res = []
    cys = [(t[1] + t[3]) / 2 for t in targets]
    for i, t in enumerate(targets):
        mx = 0.35 * size_px                        # 横向余量收紧：相邻块的渲染尾巴曾伸进测量窗把锚点推着走
        my = max(4.0, 0.22 * size_px)
        x0, x1 = int(max(0, t[0] - mx)), int(min(W, t[2] + mx))
        y0, y1 = t[1] - my, t[3] + my
        if i > 0: y0 = max(y0, (cys[i - 1] + cys[i]) / 2)
        if i < len(targets) - 1: y1 = min(y1, (cys[i] + cys[i + 1]) / 2)
        tcy = (t[1] + t[3]) / 2
        for ob in (others or []):                  # 跨字块：与本行横向重叠的邻行，窗口裁到两行中线
            ox = min(t[2], ob[2]) - max(t[0], ob[0])
            if ox < 0.3 * max(1.0, t[2] - t[0]): continue
            ocy = (ob[1] + ob[3]) / 2
            if abs(ocy - tcy) < 2: continue        # 同一行（自己）
            if ocy < tcy: y0 = max(y0, (ocy + tcy) / 2)
            else: y1 = min(y1, (ocy + tcy) / 2)
        y0, y1 = int(max(0, y0)), int(min(H, y1))
        if y1 - y0 < max(4, 0.75 * (t[3] - t[1])):   # 窗被邻行裁成碎片 → 视为测不到（量碎片会驱动字号失控）
            res.append(None); continue
        win = g[y0:y1, x0:x1] < 128
        # 只认与目标行框相交的墨列连通段（防串门：邻 zone 的墨落在窗内但在目标框外）
        cols = win.any(axis=0)
        if cols.any():
            runs = []; s_ = None
            for cx in range(len(cols) + 1):
                on = cx < len(cols) and cols[cx]
                if on and s_ is None: s_ = cx
                elif not on and s_ is not None: runs.append((s_, cx)); s_ = None
            gap = max(4, int(1.0 * size_px))
            merged = []
            for r_ in runs:   # 字间空隙合并
                if merged and r_[0] - merged[-1][1] <= gap: merged[-1] = (merged[-1][0], r_[1])
                else: merged.append(list(r_) if isinstance(r_, tuple) else r_)
            merged = [list(m_) for m_ in merged]
            tt0, tt1 = t[0] - x0 - 4, t[2] - x0 + 4
            keep = [m_ for m_ in merged if m_[1] > tt0 and m_[0] < tt1]
            if keep:
                kx0, kx1 = min(m_[0] for m_ in keep), max(m_[1] for m_ in keep)
                sel = np.zeros_like(cols); sel[kx0:kx1] = True
                win = win & sel[None, :]
        ys, xs = np.where(win)
        if len(ys) == 0: res.append(None); continue
        res.append([x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1])
    return res

def update_params(params, zone_spec, zinfo, measured):
    """一次迭代更新。返回 (new_params, err dict)。"""
    size_px = params["size_pt"] * PT2PX
    texts = [line_text(t) for t in zone_spec["lines"]]; lines = zinfo["lines"]
    wr, dxs, dys, spc_res = [], [], [], []
    n_ln = len(lines)
    tcy_i = [None] * n_ln; mcy_i = [None] * n_ln      # 按行序对齐（漏测行留 None），供逐间隙行距校准
    for i, (L, m) in enumerate(zip(lines, measured)):
        if m is None: continue
        t = texts[L["idx"]] if L["idx"] < len(texts) else ""
        tb = L["bbox"]; tw, th = tb[2] - tb[0], tb[3] - tb[1]; mw, mh = m[2] - m[0], m[3] - m[1]
        n = char_count(t.strip(" 　"))
        if mw > 2 and tw > 2 and mh > 2:
            # 正交设计：字号只管高度（各类行通用），宽度全部交给字间距——
            # 原图标题常带 letter-spacing，中文行靠字号撑宽度会把字撑大（高度就错了）
            wr.append(th / mh)
            res_pc = (tw - mw) / max(1, n - 1)
            if n >= 2 and abs(res_pc) < 0.3 * size_px: spc_res.append(res_pc)   # 残差离谱=测歪，不喂
        _dx = (tb[0] - m[0]) if params["align"] == "left" else \
              (tb[2] - m[2]) if params["align"] == "right" else \
              ((tb[0] + tb[2]) / 2 - (m[0] + m[2]) / 2)
        if abs(_dx) < 2.0 * size_px: dxs.append(_dx)      # 偏出两个字宽=测到了别的东西，不动
        _dy = (tb[1] + tb[3]) / 2 - (m[1] + m[3]) / 2
        if abs(_dy) < 1.5 * size_px: dys.append(_dy)
        tcy_i[i] = (tb[1] + tb[3]) / 2; mcy_i[i] = (m[1] + m[3]) / 2
    p = dict(params)
    err = {"w_rel": None, "dx": None, "dy": None, "pitch_rel": None, "spc": None}
    if wr:
        r = float(np.clip(np.median(wr), 0.85, 1.18)) ** 0.65
        p["size_pt"] = params["size_pt"] * r; err["w_rel"] = float(np.median(wr)) - 1
    # （曾以实测字距封顶字号防"截断拉伸"——字号改由高度驱动后拉伸不再发生，
    #   而误算的字距会把字号按死、宽度差全靠字距硬凑出"小字大间距"，已拆除）
    if spc_res:
        add = float(np.clip(np.median(spc_res) * PX2PT * 0.7, -1.2, 1.2))
        p["spc_pt"] = float(np.clip(params.get("spc_pt", 0.0) + add, -3.0, 6.0)); err["spc"] = add
    if dxs:
        dx = float(np.median(dxs)); p["x_anchor_px"] = params["x_anchor_px"] + dx; err["dx"] = dx
    # 行距：逐间隙独立校准（原图行间隙本就不均，一刀切中位数会让后排行越漂越远）
    if n_ln >= 2:
        pts = list(params.get("pitch_pts") or []) or [params["pitch_pt"]] * (n_ln - 1)
        while len(pts) < n_ln - 1: pts.append(params["pitch_pt"])
        p0 = list(params.get("pitch0_pts") or pts); p["pitch0_pts"] = p0   # 初值笼子：防误测样本把间隙带飞
        moved = []
        for i in range(n_ln - 1):
            if tcy_i[i] is None or tcy_i[i + 1] is None or mcy_i[i] is None or mcy_i[i + 1] is None: continue
            g_err = (tcy_i[i + 1] - tcy_i[i]) - (mcy_i[i + 1] - mcy_i[i])
            if abs(g_err) > 0.5 * size_px: continue                        # 测歪不更新
            v = float(pts[i] + np.clip(g_err, -8, 8) * PX2PT * 0.8)
            lo, hi = p0[min(i, len(p0) - 1)] - 3.0, p0[min(i, len(p0) - 1)] + 3.0
            pts[i] = float(np.clip(v, lo, hi))
            moved.append(abs(g_err))
        p["pitch_pts"] = pts
        p["pitch_pt"] = float(np.median(pts))
        if moved: err["pitch_rel"] = float(np.max(moved))
    if dys:
        # 用首行对齐（行距已逐间隙修正）
        dy = float(dys[0]) if len(dys) == 1 else float(np.median(dys[:2]))
        p["y_top_px"] = params["y_top_px"] + dy; err["dy"] = dy
    return p, err
