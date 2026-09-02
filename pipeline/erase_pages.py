"""阶段：文字擦除。输入 spec + src，输出 work/<deck>/clean/pNN_text.png（去文字）、pNN_textmask.png、ink/pNN.json"""
import sys, json, os, numpy as np, cv2, time
from ppt2layers.common import load_spec, load_src, deck_dir, save_rgb
from ppt2layers.ink import analyze_zone
from ppt2layers.erase import erase_zone
from ppt2layers.inpaint import inpaint_region
deck = sys.argv[1]; pages = [int(p) for p in sys.argv[2:]]
for n in pages:
    t = time.time()
    spec = load_spec(deck, n); img = load_src(deck, n)
    kinds = {p["id"]: p["kind"] for p in spec.get("panels", [])}
    fills = {p["id"]: p.get("fill", "") for p in spec.get("panels", [])}
    ocr = json.load(open(f"{deck_dir(deck)}/ocr/p{n:02d}.json"))
    cur = img.copy(); allmask = np.zeros(img.shape[:2], bool); rep = {"page": n, "zones": []}
    lama_mask = np.zeros(img.shape[:2], bool)   # 非色块文字：全页合并后一次 LaMa（避免邻近文字诱导模型"续写"笔画）
    for z in spec["text_zones"]:
        ck = "bg"; cid = None
        if ":" in z.get("container", "bg"):
            cid = z["container"].split(":")[1]; ck = kinds.get(cid, "card")
        if len(z.get("ocr_ids", [])) == len(z["lines"]) and z["ocr_ids"]:
            z["_line_priors"] = [ocr[i]["box"] if i < len(ocr) else None for i in z["ocr_ids"]]
        r = analyze_zone(img, z, ck)          # 在原图上测量（不受前面擦除影响）
        uniform = ck in ("cell", "box") or (ck == "card" and fills.get(cid, "") == "opaque_light")
        if uniform:
            cur, m = erase_zone(cur, r, uniform_fill=True)
        else:
            _, m = erase_zone(cur, r, uniform_fill=False, dry_run=True)
            lama_mask |= m
        allmask |= m
        zi = {k: v for k, v in r.items() if k not in ("mask", "d")}; zi["id"] = z["id"]; zi["font"] = z["font"]; zi["uniform_fill"] = uniform
        rep["zones"].append(zi)
        bad = [L for L in r["lines"] if not L["ok"]]
        print(f"p{n:02d} {z['id']:>4} lines {r['n_found']}/{r['n_expected']} pol={r['polarity']} {'POLWARN' if r['polarity_warn'] else ''} bad={len(bad)} uniform={uniform}", flush=True)
    if lama_mask.any():
        cur = inpaint_region(cur, lama_mask.astype(np.uint8), margin=260)
    # 残影清扫：擦过后文字区里仍与"局部底"高对比的像素 = 漏擦笔迹
    # （彩字测偏、白字黑底、OCR 框咬掉笔画尾），并入掩膜补擦一次——治重排后"双影"
    # 扫描窗 = 标注框 ∪ 实测行框：标注框咬字（比实际字位偏十几像素）时，框外的半截字也要清；
    # 黑底大白斑一轮 LaMa 补不净（会"续"出灰痕）→ 迭代清扫至收敛（最多 3 轮）
    sweep = [(z["bbox"], False) for z in spec["text_zones"]]
    sweep += [(L["bbox"], True) for zi in rep["zones"] for L in zi.get("lines", []) if L.get("bbox")]
    # 擦除掩膜的行级块也入清扫域（豁免保护门）：LaMa 在大掩膜内会"续写"出变形幻字，
    # 位置可离原行框很远、但必在掩膜内（p13 幻字案——行框窗与保护门都罩不住）
    _tmH = cv2.dilate((allmask * 255).astype(np.uint8), np.ones((1, 31), np.uint8))
    _nnT, _labT, _statT, _cT = cv2.connectedComponentsWithStats((_tmH > 128).astype(np.uint8), 8)
    for _ti in range(1, _nnT):
        _tx, _ty, _tw, _th, _ta = _statT[_ti]
        if _ta > 200: sweep.append(([int(_tx), int(_ty), int(_tx + _tw), int(_ty + _th)], True))
    for _round in range(3):
        resid = np.zeros(img.shape[:2], bool)
        P = cv2.medianBlur(cur, 21)
        dif = np.abs(cur.astype(np.int16) - P.astype(np.int16)).max(axis=2)
        _hsv = cv2.cvtColor(cur, cv2.COLOR_BGR2HSV)
        _sat = (_hsv[..., 1] > 90) & (_hsv[..., 2] > 120)   # 高饱和彩色=装饰图标/星星，不是文字残渣（残渣灰阶）——窗扩大后必须豁免
        grown = cv2.dilate(allmask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0 if _round == 0 else np.zeros(img.shape[:2], bool)
        for bb, _is_line in sweep:
            x0, y0, x1, y1 = [int(v) for v in bb]
            _mx = max(8, int((y1 - y0) * 0.7))    # 横向余量按行高自适应：行尾"——"长破折号出框（OCR 检不出长横线），±8px 兜不住
            _my = max(8, int((y1 - y0) * 0.35))   # 纵向余量同理：LaMa 大掩膜"续写"的变形幻字贴在原行下缘外（p13 案）
            x0, y0 = max(0, x0 - _mx), max(0, y0 - _my); x1, y1 = min(img.shape[1], x1 + _mx), min(img.shape[0], y1 + _my)
            if x1 - x0 < 4 or y1 - y0 < 4: continue
            sub = (dif[y0:y1, x0:x1] > 45) & ~grown[y0:y1, x0:x1] & ~_sat[y0:y1, x0:x1]
            # 保护门只保极端场景（几乎整窗高对比=真图形背景）；行框窗完全豁免。
            # 0.4 时代整行残字/幻字占比常超门限反被拦（p13 案），文字区误擦风险已有 _sat+dif 阈值兜底
            if sub.sum() < 30 or ((not _is_line) and sub.mean() > 0.85): continue
            resid[y0:y1, x0:x1] |= sub
        if not resid.any():
            if _round == 0: continue   # 首轮排除了掩膜内区（grown），幻字恰在掩膜内——强制进第二轮全量检测
            break
        resid = cv2.morphologyEx(resid.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        resid = cv2.dilate(resid, np.ones((5, 5), np.uint8)).astype(bool)
        if not resid.any(): break
        print(f"p{n:02d} 残影清扫 R{_round + 1} {int(resid.sum())}px", flush=True)
        # 回填用中值底而非 LaMa：残影的定义就是"与中值底高对比"，中值替换必收敛；
        # LaMa 回填会参考周围残字再次"续写"幻字——实测 3 轮 8000→3109px 猫抓老鼠不收敛
        _rw = cv2.GaussianBlur(cv2.dilate(resid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32), (7, 7), 0)[..., None]
        cur = (_rw * P.astype(np.float32) + (1 - _rw) * cur.astype(np.float32)).astype(np.uint8)
        allmask |= resid
    os.makedirs(f"{deck_dir(deck)}/clean", exist_ok=True); os.makedirs(f"{deck_dir(deck)}/ink", exist_ok=True)
    save_rgb(f"{deck_dir(deck)}/clean/p{n:02d}_text.png", cur)
    cv2.imwrite(f"{deck_dir(deck)}/clean/p{n:02d}_textmask.png", (allmask * 255).astype(np.uint8))
    json.dump(rep, open(f"{deck_dir(deck)}/ink/p{n:02d}.json", "w"), ensure_ascii=False, indent=1)
    print(f"p{n:02d} done {time.time()-t:.1f}s", flush=True)

os._exit(0)
