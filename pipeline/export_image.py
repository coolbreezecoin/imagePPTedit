# -*- coding: utf-8 -*-
"""差量图片导出：原图为底，只把"用户修改波及的区域"换成重建结果。
用法: export_image.py <deck> [页码...]（缺省全部页）
产物: output/<deck>_成品图_pNN.png

原则：没动过的地方任何重建都只会更差——掩膜外逐像素取原图；
掩膜 = 每个被改元素的旧位置 ∪ 新位置（挪层/隐藏/挪字/改字/改判/改色），边界羽化。
有 orig/pNN.png（图片直传的原件副本）时在**原始分辨率**合成；否则退回画布空间，
padded 页裁掉舞台衬底只出内容区。一处没改 → 导出就是原图本身。"""
import sys, os, json, glob
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TW, TH = 1672, 941

# 这些标记任一存在即视为"用户改过这个文字块"（改字/改色/改判/改字体等前端都会打 user_touched；
# color_user/style_user 是更早就有的同类标记，一并认）
ZONE_TOUCH_KEYS = ("user_touched", "color_user", "style_user", "font_user", "text_user")


def _zone_pad_box(z):
    """文字块掩膜框：bbox 外扩（重排后可能变长变高）。"""
    b = [float(v) for v in z["bbox"]]
    n = max(1, len(z.get("lines") or [1]))
    lh = max(8.0, (b[3] - b[1]) / n)
    px, py = 0.8 * lh, 0.45 * lh
    return [b[0] - px, b[1] - py, b[2] + px, b[3] + py]


def collect_boxes(spec, manifest):
    """返回画布坐标下的波及框列表；空列表 = 没有任何修改。"""
    boxes = []
    ed = spec.get("edits") or {}
    layers = (manifest or {}).get("layers") or []

    def _find_layer(key):
        for L in layers:
            if L.get("id") == key or L.get("name") == key:
                return L
        return None

    for key, e in (ed.get("layers") or {}).items():
        L = _find_layer(key)
        if not L or L.get("file") == "背景.png":
            continue
        b = [float(v) for v in L["box"]]
        boxes.append(list(b))                      # 旧位置：露出的补底
        if not e.get("hidden"):
            dx, dy = float(e.get("dx", 0)), float(e.get("dy", 0))
            if dx or dy:
                boxes.append([b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy])
            elif not (e.get("dx") or e.get("dy")):
                boxes.pop()                        # 既没挪也没藏的空条目：不算修改
    zones = {z["id"]: z for z in (spec.get("text_zones") or [])}
    for tid, e in (ed.get("zones") or {}).items():
        z = zones.get(tid)
        if not z:
            continue
        b = _zone_pad_box(z)
        boxes.append(list(b))
        if not e.get("hidden"):
            dx, dy = float(e.get("dx", 0)), float(e.get("dy", 0))
            if dx or dy:
                boxes.append([b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy])
            elif not (e.get("dx") or e.get("dy")):
                boxes.pop()
    for z in (spec.get("text_zones") or []):
        if any(z.get(k) for k in ZONE_TOUCH_KEYS):
            boxes.append(_zone_pad_box(z))
    for k in (spec.get("calligraphy") or []):      # text→艺术字改判块：渲染来源变了，也算修改
        if k.get("style_user"):
            b = [float(v) for v in k["bbox"]]
            boxes.append([b[0] - 6, b[1] - 6, b[2] + 6, b[3] + 6])
    return boxes


def build_mask(boxes, pad=6, feather=9):
    m = np.zeros((TH, TW), np.float32)
    for b in boxes:
        x0 = max(0, int(b[0] - pad)); y0 = max(0, int(b[1] - pad))
        x1 = min(TW, int(b[2] + pad)); y1 = min(TH, int(b[3] + pad))
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = 1.0
    if feather:
        k = feather * 2 + 1
        m = cv2.GaussianBlur(m, (k, k), feather / 2.0)
    return m


def export_page(dd, out_base, n, item):
    p2 = f"p{n:02d}"
    src = cv2.imread(f"{dd}/src/{p2}.png")
    if src is None:
        print(f"{p2} 无源图，跳过"); return None
    spec = {}
    sp = f"{dd}/spec/{p2}.json"
    if os.path.exists(sp):
        spec = json.load(open(sp))
    man = None
    mp = f"{dd}/layers/{p2}/manifest.json"
    if os.path.exists(mp):
        man = json.load(open(mp))
    orig_p = f"{dd}/orig/{p2}.png"
    orig = cv2.imread(orig_p) if os.path.exists(orig_p) else None
    cb = [0, 0, TW, TH]
    if item and item.get("content_box"):
        cb = [int(v) for v in item["content_box"]]

    boxes = collect_boxes(spec, man)
    out_path = f"{out_base}_成品图_{p2}.png"

    if not boxes:                                   # 无修改：导出即原图
        if orig is not None:
            cv2.imwrite(out_path, orig)
        elif cb != [0, 0, TW, TH]:
            cv2.imwrite(out_path, src[cb[1]:cb[3], cb[0]:cb[2]])
        else:
            cv2.imwrite(out_path, src)
        print(f"{p2} 无修改 → 原图直出")
        return out_path

    reb = cv2.imread(f"{dd}/qa/{p2}_rebuilt.png")
    if reb is None:
        print(f"{p2} 无重建图（先跑组装/QA），跳过"); return None
    if reb.shape[:2] != (TH, TW):
        reb = cv2.resize(reb, (TW, TH), interpolation=cv2.INTER_LANCZOS4)
    mask = build_mask(boxes)
    cov = float((mask > 0.5).mean())

    if orig is not None:                            # 原始分辨率合成：掩膜外逐像素原件
        H0, W0 = orig.shape[:2]
        rc = reb[cb[1]:cb[3], cb[0]:cb[2]]
        mc = mask[cb[1]:cb[3], cb[0]:cb[2]]
        rc = cv2.resize(rc, (W0, H0), interpolation=cv2.INTER_LANCZOS4)
        mc = cv2.resize(mc, (W0, H0), interpolation=cv2.INTER_LINEAR)[..., None]
        out = (mc * rc.astype(np.float32) + (1 - mc) * orig.astype(np.float32))
        cv2.imwrite(out_path, np.clip(out, 0, 255).astype(np.uint8))
        print(f"{p2} 差量导出（原始 {W0}x{H0}，波及 {cov:.1%}，{len(boxes)} 框）")
    else:                                           # 无原件：画布空间，padded 页裁内容区
        m3 = mask[..., None]
        out = (m3 * reb.astype(np.float32) + (1 - m3) * src.astype(np.float32))
        out = np.clip(out, 0, 255).astype(np.uint8)
        if cb != [0, 0, TW, TH]:
            out = out[cb[1]:cb[3], cb[0]:cb[2]]
        cv2.imwrite(out_path, out)
        print(f"{p2} 差量导出（画布空间，波及 {cov:.1%}，{len(boxes)} 框）")
    return out_path


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print("用法: export_image.py <deck> [页码...]"); sys.exit(1)
    deck = args[0]
    dd = os.path.join(ROOT, "work", deck)
    meta = {}
    if os.path.exists(f"{dd}/deck.json"):
        meta = json.load(open(f"{dd}/deck.json"))
    items = meta.get("items") or []
    n_src = len(glob.glob(f"{dd}/src/p*.png"))
    pages = [int(p) for p in args[1:]] or list(range(1, n_src + 1))
    out_base = os.path.join(ROOT, "output", deck)
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    done = []
    for n in pages:
        item = items[n - 1] if n - 1 < len(items) else None
        r = export_page(dd, out_base, n, item)
        if r:
            done.append(r)
    if not done:
        print("ERROR: 没有导出任何页"); sys.exit(2)
    print(f"DONE {len(done)} 页")


if __name__ == "__main__":
    main()
