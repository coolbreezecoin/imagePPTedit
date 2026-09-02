# -*- coding: utf-8 -*-
"""阶段：文字分级门控（erase 之前）。对每个 plain 文字块做"转成可编辑后还像不像"自检，
不像的自动降级为艺术字图层（method=art，改道 calligraphy 通道整块抠图保真）。
判据：特大字号+粗描边 / 渲染形状对不上 / 行内多色或上下渐变。
用法: art_gate.py <deck> <pages...>   （改写 spec，幂等；style_user 的块不动）"""
import sys, os, json
import numpy as np, cv2
from ppt2layers.common import load_spec, deck_dir

FONT_CANDIDATES = [
    "/System/Library/Fonts/Hiragino Sans GB.ttc",                  # mac（新系统 PingFang 不在此路径）
    "/System/Library/Fonts/STHeiti Medium.ttc",                    # mac 兜底
    "/usr/share/fonts/slidelift/SourceHanSansCN-Regular.otf",      # 服务器
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
_font_path = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)


def _zone_text(z):
    out = []
    for ln in z.get("lines") or []:
        out.append(ln if isinstance(ln, str) else "".join(r.get("t", "") for r in (ln.get("runs") or [])))
    return out


def _ink_mask(crop):
    """行内墨迹：与边缘环中值色的通道最大差 > 自适应阈值。返回 (mask bool, diff)。"""
    h, w = crop.shape[:2]
    ring = np.ones((h, w), bool); ring[2:-2, 2:-2] = False
    bgc = np.median(crop[ring].reshape(-1, 3), axis=0)
    d = np.abs(crop.astype(np.float32) - bgc).max(axis=2)
    thr = max(35.0, 0.35 * float(d.max()))
    return d > thr, d


def _shape_iou(text, ink, lineH):
    """PIL 用替身字体渲染同文本，与实际墨迹比形状 IoU（艺术变形/立体字 IoU 会塌）。"""
    if not _font_path or not text.strip(): return None
    try:
        from PIL import Image, ImageDraw, ImageFont
        f = ImageFont.truetype(_font_path, size=max(12, int(lineH * 0.82)))
        im = Image.new("L", (max(8, int(lineH * 0.9) * max(1, len(text))), int(lineH * 1.6)), 0)
        dr = ImageDraw.Draw(im); dr.text((4, int(lineH * 0.2)), text, fill=255, font=f)
        r = np.array(im) > 96
        ys, xs = np.where(r)
        if len(ys) < 20: return None
        r = r[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        ys, xs = np.where(ink)
        if len(ys) < 20: return None
        k = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        H_, W_ = 48, 240
        a = cv2.resize(r.astype(np.uint8), (W_, H_)) > 0
        b = cv2.resize(k.astype(np.uint8), (W_, H_)) > 0
        inter = (a & b).sum(); union = (a | b).sum()
        return inter / max(1, union)
    except Exception:
        return None


def _judge(zone, boxes, img, H):
    """返回 (降级?, 原因列表)。boxes=该 zone 的行框列表。"""
    texts = _zone_text(zone)
    pol = zone.get("polarity") if zone.get("polarity") in ("light", "dark") else "light"
    inks, tops, bots, sws, ious, rings, covs = [], [], [], [], [], [], []
    lineHs = []
    for bi, b in enumerate(boxes):
        x0, y0, x1, y1 = [int(v) for v in b]
        x0, y0 = max(0, x0 - 2), max(0, y0 - 2); x1, y1 = min(img.shape[1], x1 + 2), min(img.shape[0], y1 + 2)
        if x1 - x0 < 8 or y1 - y0 < 8: continue
        crop = img[y0:y1, x0:x1]
        m, d = _ink_mask(crop)
        if m.sum() < 40: continue
        lineHs.append(y1 - y0)
        # 取色只采"实心核心"像素——抗锯齿过渡像素会被误算成第二种颜色（小字过杀之祸）
        core = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & (d > max(60.0, 0.55 * float(d.max())))
        sel = core if core.sum() >= 40 else m
        inks.append(crop[sel].astype(np.float32))
        mid = (y1 - y0) // 2
        tm, bm = sel[:mid], sel[mid:]
        if tm.sum() > 20 and bm.sum() > 20:
            tops.append(crop[:mid][tm].astype(np.float32).mean(axis=0))
            bots.append(crop[mid:][bm].astype(np.float32).mean(axis=0))
        dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
        sws.append(2.0 * float(dist.max()))
        # 描边/立体壳：紧贴墨迹外一圈存在"既不是填充色也不是底色"的**一致颜色**（勾边/挤出侧面）。
        # 不能只比亮度——深底黑描边亮度差不足；壳层色彩一致性可把照片纹理背景排除在外
        ring = (cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0) & \
               ~(cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        if ring.sum() > 30:
            hh, ww = crop.shape[:2]
            rr = np.ones((hh, ww), bool); rr[2:-2, 2:-2] = False
            bgc2 = np.median(crop[rr].reshape(-1, 3), axis=0)
            fillc = np.median(crop[sel].reshape(-1, 3), axis=0)
            rp = crop[ring].astype(np.float32)
            shell = (np.abs(rp - fillc).max(axis=1) > 40) & (np.abs(rp - bgc2).max(axis=1) > 25)
            frac = float(shell.mean())
            if shell.sum() > 30 and float(rp[shell].std(axis=0).mean()) < 35:
                rings.append(frac)
            else:
                rings.append(0.0)
        # 墨量覆盖率：紧包围盒内墨迹占比（超粗装饰字 0.45+，正常粗体 ~0.3）
        ys_, xs_ = np.where(m)
        covs.append(float(m.sum()) / max(1.0, (ys_.max() - ys_.min() + 1) * (xs_.max() - xs_.min() + 1)))
        if bi < 3 and bi < len(texts):
            iou = _shape_iou(texts[bi].replace(" ", ""), m, y1 - y0)
            if iou is not None: ious.append(iou)
    if not lineHs: return False, []
    lineH = float(np.median(lineHs))
    allpx = np.concatenate(inks, axis=0)
    spread = 0.0   # 行内多色：主簇（权重≥15%）两两的**色度**距离——白字抗锯齿是灰阶亮度差，
    if len(allpx) >= 60:   # 去掉亮度分量后趋零；绿字 vs 白字的色度差才是真多色
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
        _c, lab, cen = cv2.kmeans(allpx, 3, None, crit, 2, cv2.KMEANS_PP_CENTERS)
        wgt = np.bincount(lab.ravel(), minlength=3) / len(lab)
        major = [cen[i] - float(cen[i].mean()) for i in range(3) if wgt[i] >= 0.15]   # 去亮度→纯色度
        for i in range(len(major)):
            for j in range(i + 1, len(major)):
                spread = max(spread, float(np.abs(major[i] - major[j]).max()))
    gt, gb = (np.mean(tops, axis=0), np.mean(bots, axis=0)) if tops and bots else (None, None)
    grad = float(np.abs((gt - gt.mean()) - (gb - gb.mean())).max()) if gt is not None else 0.0   # 渐变也按色度算
    swr = float(np.median(sws)) / max(1.0, lineH)
    iou = float(np.mean(ious)) if ious else None
    ringf = float(np.mean(rings)) if rings else 0.0
    cov = float(np.median(covs)) if covs else 0.0
    giant, big = lineH > 0.16 * H, lineH > 0.075 * H
    print(f"    [gate] {zone.get('id','?'):>4} h{lineH:.0f} 描边比{swr:.2f} IoU{('%.2f' % iou) if iou is not None else '—'} "
          f"渐变{grad:.0f} 多色{spread:.0f} 勾边{ringf:.2f} 墨量{cov:.2f}", flush=True)
    reasons = []
    if giant:
        if swr > 0.22: reasons.append(f"特大字+描边比{swr:.2f}")
        if iou is not None and iou < 0.35: reasons.append(f"特大字+形状IoU{iou:.2f}")
        if grad > 24: reasons.append(f"特大字+上下渐变{grad:.0f}")
        if spread > 45: reasons.append(f"特大字+多色{spread:.0f}")
        if ringf > 0.20: reasons.append(f"特大字+勾边环{ringf:.2f}")
        if cov > 0.45: reasons.append(f"特大字+墨量{cov:.2f}")
    elif big:
        hits = []
        if swr > 0.26: hits.append(f"描边比{swr:.2f}")
        if iou is not None and iou < 0.30: hits.append(f"形状IoU{iou:.2f}")
        if grad > 28: hits.append(f"上下渐变{grad:.0f}")
        if spread > 50: hits.append(f"多色{spread:.0f}")
        if ringf > 0.28: hits.append(f"勾边环{ringf:.2f}")
        if cov > 0.50: hits.append(f"墨量{cov:.2f}")
        if len(hits) >= 2: reasons.append("大字+" + "+".join(hits))
    if spread > 60: reasons.append(f"行内多色{spread:.0f}")
    return bool(reasons), reasons


def gate_page(deck, n):
    spec = load_spec(deck, n)
    img_p = f"{deck_dir(deck)}/src/p{n:02d}.png"
    img = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB)
    H = img.shape[0]
    try:
        ocr = json.load(open(f"{deck_dir(deck)}/ocr/p{n:02d}.json"))
    except Exception:
        ocr = []
    keep, moved = [], []
    exist_ids = {k.get("id") for k in spec.get("calligraphy", [])}
    an = 0
    zs = spec.get("text_zones", [])
    def _overlapped(z):
        b = z["bbox"]
        for o in zs:
            if o is z: continue
            ob = o["bbox"]
            ox = min(b[2], ob[2]) - max(b[0], ob[0])
            oy = min(b[3], ob[3]) - max(b[1], ob[1])
            if oy > 0 and ox > 0.3 * min(b[2] - b[0], ob[2] - ob[0]): return True
        return False
    for z in zs:
        if z.get("style_user"):            # 用户手动定档的不动
            keep.append(z); continue
        boxes = [ocr[j]["box"] for j in (z.get("ocr_ids") or []) if isinstance(j, int) and 0 <= j < len(ocr)]
        if not boxes:   # 无行锚定：多行块按行数把 bbox 均分成行条——整块当"一行大字"会把正文误判成艺术字
            nl = max(1, len(z.get("lines") or []))
            x0_, y0_, x1_, y1_ = z["bbox"]; hh_ = (y1_ - y0_) / nl
            boxes = [[x0_, y0_ + i_ * hh_, x1_, y0_ + (i_ + 1) * hh_] for i_ in range(nl)]
        art = str(z.get("style") or "").lower() == "art"
        reasons = ["AI 分级 art"] if art else []
        if not art:
            # 微缩/密排：行高太小或标注框与邻块重叠的密排小字——转文字是跟物理打架
            # （测不准、擦不净、编辑价值低；要编辑随时可在审校台"转为可编辑文字"）
            lh = float(np.median([b[3] - b[1] for b in boxes]))
            # 密排规则曾撤回试验"全部文字化"：三次实证（手工回填/全OCR锚定重起草）排版链在
            # 行高~24px 多列密排下都会错乱——恢复成层为默认；要编辑哪块，审校台单块改判即可
            if lh < 20: art, reasons = True, [f"微缩文字（行高{lh:.0f}px）"]
            elif lh < 30 and _overlapped(z): art, reasons = True, [f"密排微字（行高{lh:.0f}px+邻块框重叠）"]
        if not art:
            art, reasons = _judge(z, boxes, img, H)
        if art:
            an += 1
            aid = f"a{an}"
            while aid in exist_ids: an += 1; aid = f"a{an}"
            exist_ids.add(aid)
            moved.append({"id": aid, "text": "".join(_zone_text(z))[:20] or "艺术字",
                          "bbox": z["bbox"], "ocr_ids": z.get("ocr_ids") or [],
                          "color": z.get("color") or "#FFFFFF",
                          "polarity": z.get("polarity") or "light", "method": "art",
                          "notes": "引擎降级为图层：" + "；".join(reasons)})
            print(f"p{n:02d} 分级 {z.get('id')}「{''.join(_zone_text(z))[:10]}」→ 艺术字图层（{'；'.join(reasons)}）", flush=True)
        else:
            keep.append(z)
    if moved:
        spec["text_zones"] = keep
        spec.setdefault("calligraphy", []).extend(moved)
        json.dump(spec, open(f"{deck_dir(deck)}/spec/p{n:02d}.json", "w"), ensure_ascii=False, indent=1)
    print(f"p{n:02d} gate done 降级{len(moved)}块", flush=True)


if __name__ == "__main__":
    deck = sys.argv[1]
    for p in [int(x) for x in sys.argv[2:]]:
        gate_page(deck, p)
