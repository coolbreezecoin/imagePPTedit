"""阶段：图层分离。输入 clean/pNN_text.png + spec → layers/pNN/*.png + layers/pNN/manifest.json + 背景"""
import sys, json, os, numpy as np, cv2, time
from ppt2layers.common import load_spec, deck_dir, save_rgb, save_rgba, load_src
from ppt2layers.matte import calligraphy_matte, art_text_matte, panel_mask, rembg_alpha, alpha_to_rgba, remove_and_fill, grabcut_alpha, table_alpha, fill_with_median, solidify_alpha, unmix_translucent
from ppt2layers.sam_matte import sam_alpha
from ppt2layers.detect import refine_box, assign_person_boxes, composite_boxes, find_unclaimed_objects, EN2CN, score_object_alpha
from ppt2layers.inpaint import inpaint_region
deck = sys.argv[1]; pages = [int(p) for p in sys.argv[2:]]
for n in pages:
    t = time.time(); spec = load_spec(deck, n)
    img = cv2.cvtColor(cv2.imread(f"{deck_dir(deck)}/clean/p{n:02d}_text.png"), cv2.COLOR_BGR2RGB)
    out_dir = f"{deck_dir(deck)}/layers/p{n:02d}"; os.makedirs(out_dir, exist_ok=True)
    manifest = {"page": n, "layers": []}   # 按叠放顺序从下到上记录
    cur = img.copy()
    # 1) 书法
    cal_layers = []
    cal_patch = np.zeros(img.shape[:2], np.float32)      # 书法擦除掩膜（精确到笔画）：解混 alpha 需绕开的修补渣
    cal_solve = []                                       # (alpha_full, box, rgba)：供"笔画背后真实底"反解
    brush_patch = np.zeros(img.shape[:2], np.float32)    # 仅毛笔书法的笔画掩膜（反解范围；艺术字不参与）
    _tm = cv2.imread(f"{deck_dir(deck)}/clean/p{n:02d}_textmask.png", cv2.IMREAD_GRAYSCALE)
    textmask = (_tm.astype(np.float32) / 255.0) if _tm is not None and _tm.shape == img.shape[:2] else None
    used_fn = set()   # 图层文件名去重：两个"人物_女孩"互相覆盖 PNG，另一个的框会拉伸错图
    def _ufn(base):
        fn_ = f"{base}.png"; k2 = 2
        while fn_ in used_fn: fn_ = f"{base}{k2}.png"; k2 += 1
        used_fn.add(fn_); return fn_
    # 艺术字：所有框并集一次修补作公共底——各框单独修补时邻框的字会被 LaMa "续写"出残带
    # 毛笔书法框也入并集：书法统一走差异抠图后同样需要纯净公共底
    _art_boxes = [k["bbox"] for k in spec.get("calligraphy", []) if k.get("bbox")]
    art_bg = None; cur0 = cur.copy()
    if _art_boxes:
        _am = np.zeros(img.shape[:2], np.float32)
        for b_ in _art_boxes:
            _am[max(0, int(b_[1])):int(b_[3]), max(0, int(b_[0])):int(b_[2])] = 1.0
        art_bg = remove_and_fill(cur0, _am, dilate_px=0, margin=300)
        # LaMa 在深色渐变大区必产"浮雕骨架"且迭代重填是猫抓老鼠——改**渐变续底**：
        # 掩膜区按列用上下边界色线性插值（数学延续渐变，不给模型想象空间）；
        # 边界纹理重的列（头发/复杂物）按纹理度回退 LaMa。
        _mb = _am > 0
        _out = art_bg.astype(np.float32)
        _Hh = img.shape[0]
        for _cx in np.where(_mb.any(axis=0))[0]:
            _ys = np.where(_mb[:, _cx])[0]
            _s0 = _prev = _ys[0]; _segs = []
            for _yv in _ys[1:]:
                if _yv != _prev + 1: _segs.append((_s0, _prev)); _s0 = _yv
                _prev = _yv
            _segs.append((_s0, _prev))
            for _ya, _yb in _segs:
                _top = cur0[max(0, _ya - 4):_ya, _cx].astype(np.float32)
                _bot = cur0[_yb + 1:min(_Hh, _yb + 5), _cx].astype(np.float32)
                if not len(_top) and not len(_bot): continue
                _ct = _top.mean(axis=0) if len(_top) else _bot.mean(axis=0)
                _cb = _bot.mean(axis=0) if len(_bot) else _ct
                # 门控废除：纹理列回退 LaMa 时，LaMa 会参考框外字迹光晕把字"续写"回补丁成幽灵字
                # （p05 背景幽灵标题案：变形标题被写回背景层，与 k1 层叠成重影）；掩膜内一律数学插值
                _w = 1.0
                _t = np.linspace(0.0, 1.0, _yb - _ya + 1)[:, None]
                _grad = _ct[None, :] * (1 - _t) + _cb[None, :] * _t
                _out[_ya:_yb + 1, _cx] = _w * _grad + (1 - _w) * _out[_ya:_yb + 1, _cx]
        _k = cv2.getGaussianKernel(31, 8).T                     # 掩膜内横向轻度平滑，融合列间
        _sm = cv2.filter2D(_out, -1, _k)
        _out[_mb] = _sm[_mb]
        art_bg = np.clip(_out, 0, 255).astype(np.uint8)
        print(f"p{n:02d} 艺术字底渐变续底完成（{len(_art_boxes)} 框）", flush=True)
    _obj_guard = [[int(v) for v in o["bbox"]] for o in (spec.get("objects") or []) if o.get("bbox")]
    for k in spec.get("calligraphy", []):
        is_art = True   # 毛笔/艺术字统一差异抠图（背景=纯净修补底，字缘环全随层走）；
                        # calligraphy_matte+2.5反解退役——反解把字缘按原图写回背景，拖走层即浮雕轮廓
        r = art_text_matte(cur0, k["bbox"], bg_full=art_bg) if is_art else calligraphy_matte(cur, k["bbox"], k.get("polarity", "light"))
        if r is None: print(f"p{n:02d} calligraphy {k['id']} empty"); continue
        if is_art:
            # 物件优先 + 小框优先：起草宽框常压住邻近小物件（图标/气泡）和贴身小字块，
            # 差异抠图会把框内邻居的字整排偷进大层——拖动标题时邻居的字跟着走。
            # 重叠像素归物件框与面积更小的书法框；人物框不参与（大字笔画常伸进人物框，清了会截肢）
            bx = r["box"]; _cut = 0
            _ka = max(1, (k["bbox"][2] - k["bbox"][0]) * (k["bbox"][3] - k["bbox"][1]))
            _sib = [c["bbox"] for c in spec.get("calligraphy", [])
                    if c["id"] != k["id"]
                    and (c["bbox"][2] - c["bbox"][0]) * (c["bbox"][3] - c["bbox"][1]) < _ka]
            for _g in _obj_guard + _sib:
                ix0, iy0 = max(bx[0], _g[0]), max(bx[1], _g[1])
                ix1, iy1 = min(bx[2], _g[2]), min(bx[3], _g[3])
                if ix1 > ix0 and iy1 > iy0:
                    _cut += int((r["rgba"][iy0 - bx[1]:iy1 - bx[1], ix0 - bx[0]:ix1 - bx[0], 3] > 0).sum())
                    r["rgba"][iy0 - bx[1]:iy1 - bx[1], ix0 - bx[0]:ix1 - bx[0], 3] = 0
                    r["alpha_full"][iy0:iy1, ix0:ix1] = 0
            if _cut: print(f"p{n:02d} 艺术字 {k['id']} 让位物件框 {_cut}px", flush=True)
        pre = "艺术字" if is_art else "书法"
        fn = _ufn(f"{pre}_{k['text'][:8]}"); save_rgba(f"{out_dir}/{fn}", r["rgba"])
        cal_layers.append({"type": "calligraphy", "id": k["id"], "name": f"{pre}_{k['text'][:8]}", "file": fn, "box": [int(v) for v in r["box"]]})
        if not is_art:   # 反解只服务毛笔书法；艺术字 blend 已恒等，反解会把字缘环写回背景（浮雕骨架之祸）
            cal_solve.append((r["alpha_full"], [int(v) for v in r["box"]], r["rgba"]))
            brush_patch = np.maximum(brush_patch, cv2.dilate((r["alpha_full"] > 0.03).astype(np.float32),
                                                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(round(max(2, r["stroke_w"] * 0.6))) * 2 + 5,) * 2)))
        if is_art:
            # 擦除：复用抠层时算好的修补底，按 alpha 羽化替换（不二次跑 LaMa；层贴回后逐像素还原）
            aw = cv2.GaussianBlur(cv2.dilate((r["alpha_full"] > 0.02).astype(np.float32),
                                             np.ones((5, 5), np.uint8)), (7, 7), 0)
            cur = (aw[..., None] * r["bg"].astype(np.float32) + (1 - aw[..., None]) * cur.astype(np.float32)).astype(np.uint8)
        else:
            host = None
            kb = r["box"]
            for pp in spec.get("panels", []):
                bx = pp["bbox"]; ix = max(0, min(kb[2], bx[2]) - max(kb[0], bx[0])); iy = max(0, min(kb[3], bx[3]) - max(kb[1], bx[1]))
                if ix * iy >= 0.8 * (kb[2] - kb[0]) * (kb[3] - kb[1]): host = pp; break
            if host is not None and (host.get("kind") in ("cell", "box") or (host.get("kind") == "card" and host.get("fill") == "opaque_light")):
                cur = fill_with_median(cur, r["alpha_full"], host["bbox"], dilate_px=int(round(max(3, r["stroke_w"] * 0.9))))
            else:
                cur = remove_and_fill(cur, r["alpha_full"], dilate_px=int(round(max(2, r["stroke_w"] * 0.6))), margin=200)
        _dk = 7 if is_art else int(round(max(2, r["stroke_w"] * 0.6))) * 2 + 5   # 艺术字 sw 大，膨胀过宽会撑爆插值区
        cal_patch = np.maximum(cal_patch, cv2.dilate((r["alpha_full"] > 0.03).astype(np.float32),
                                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_dk, _dk))))
        print(f"p{n:02d} {pre} {k['id']} box={r['box']} sw={r['stroke_w']:.1f}", flush=True)
    # 1.5) 书法/艺术字擦除区迭代清扫：LaMa 在深底上一轮补不净会留灰痕（挪走图层就露馅），
    #      与文字擦除同款：按与局部底的高对比找残迹，补擦至收敛（≤2 轮）
    _sweep_boxes = [[k["bbox"][0] - 10, k["bbox"][1] - 10, k["bbox"][2] + 10, k["bbox"][3] + 10]
                    for k in spec.get("calligraphy", []) if k.get("bbox")]
    if _sweep_boxes:
        for _r in range(2):
            P_ = cv2.medianBlur(cur, 21)
            dif_ = np.abs(cur.astype(np.int16) - P_.astype(np.int16)).max(axis=2)
            res_ = np.zeros(img.shape[:2], bool)
            for bb in _sweep_boxes:
                x0_, y0_ = max(0, int(bb[0])), max(0, int(bb[1]))
                x1_, y1_ = min(img.shape[1], int(bb[2])), min(img.shape[0], int(bb[3]))
                if x1_ - x0_ < 4 or y1_ - y0_ < 4: continue
                sub_ = (dif_[y0_:y1_, x0_:x1_] > 30) & (cal_patch[y0_:y1_, x0_:x1_] > 0.05)
                if sub_.sum() < 30 or sub_.mean() > 0.5: continue
                res_[y0_:y1_, x0_:x1_] |= sub_
            if not res_.any(): break
            res_ = cv2.dilate(cv2.morphologyEx(res_.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)),
                              np.ones((5, 5), np.uint8)).astype(bool)
            if not res_.any(): break
            print(f"p{n:02d} 书法区清扫 R{_r + 1} {int(res_.sum())}px", flush=True)
            cur = remove_and_fill(cur, res_.astype(np.float32), dilate_px=0, margin=200)
    # 2) 底板（卡片/色块/徽章）
    panel_layers = []
    pan_S = np.zeros_like(img, np.float32); pan_T = np.ones(img.shape[:2], np.float32)  # 半透明板叠加的色/透过率
    for p in spec.get("panels", []):
        # 半透明衬板（柔光/渐变板）：LaMa 解混 → 单色半透明层 + 干净背景（README §9）
        if str(p.get("fill", "")).startswith("translucent") or p.get("kind") == "panel":
            r2 = unmix_translucent(cur, p["bbox"], dark=str(p.get("fill", "")).endswith("dark"),
                                   keep_bg_boxes=[z["bbox"] for z in spec.get("text_zones", []) if z.get("bbox")],
                                   patch_mask=cal_patch, keep_bg_mask=textmask)
            if r2 is not None:
                fn = _ufn(p["name"]); save_rgba(f"{out_dir}/{fn}", r2["rgba"])
                panel_layers.append({"type": "panel", "id": p["id"], "name": p["name"],
                                     "kind": p.get("kind", "panel"), "file": fn, "box": r2["box"]})
                cur = r2["bg"]
                bx2 = r2["box"]; ap = np.zeros(img.shape[:2], np.float32)
                ap[bx2[1]:bx2[3], bx2[0]:bx2[2]] = r2["rgba"][..., 3].astype(np.float32) / 255.0
                Cp = np.array(r2["color"], np.float32)
                pan_S = ap[..., None] * Cp + (1 - ap[..., None]) * pan_S
                pan_T = (1 - ap) * pan_T
                print(f"p{n:02d} 底板 {p['id']} 半透明解混 色={r2['color']}", flush=True)
                continue
            print(f"p{n:02d} 底板 {p['id']} 解混失败，转不透明路径", flush=True)
        r = panel_mask(cur, p["bbox"])
        pb = p["bbox"]
        cov_p = float((r["alpha_full"] > 0.5).sum()) / max(1.0, (pb[2] - pb[0]) * (pb[3] - pb[1])) if r is not None else 0.0
        if cov_p < 0.25:   # 颜色分割失败/覆盖过低（笔刷徽章、渐变底板）→ SAM 兜底
            a_p = sam_alpha(cur, pb, expand=0.06)
            if a_p is not None and (a_p > 0.5).sum() > 0.2 * (pb[2] - pb[0]) * (pb[3] - pb[1]):
                r = {"alpha_full": solidify_alpha(a_p), "thr": -1.0}
                print(f"p{n:02d} 底板 {p['id']} 颜色分割弱（{cov_p:.2f}），SAM 兜底", flush=True)
        if r is None: print(f"p{n:02d} panel {p['id']} empty"); continue
        rgba, box = alpha_to_rgba(cur, r["alpha_full"])
        fn = _ufn(p["name"]); save_rgba(f"{out_dir}/{fn}", rgba)
        panel_layers.append({"type": "panel", "id": p["id"], "name": p["name"], "kind": p["kind"], "file": fn, "box": box})
        cur = remove_and_fill(cur, r["alpha_full"], dilate_px=2, margin=160)
        print(f"p{n:02d} 底板 {p['id']} box={box} thr={r['thr']:.1f}", flush=True)
    # 2.5) 笔画背后的真实底：按 合成 = 书法 over 半透明板 over 底 反解——
    #      擦书法的修补是"猜"，这里是"解"：贴回后逐像素还原原图，移开字露出的是艺术光晕而非脑补
    if cal_solve and brush_patch.any():
        X = img.astype(np.float32); rem = np.ones(img.shape[:2], np.float32)
        for _af, _kb, _kr in cal_solve:
            C1 = np.zeros_like(X); C1[_kb[1]:_kb[3], _kb[0]:_kb[2]] = _kr[..., :3].astype(np.float32)
            X = X - _af[..., None] * C1; rem = rem * (1 - _af)
        X = X / np.maximum(rem[..., None], 1e-3)
        Bs = np.clip((X - pan_S) / np.maximum(pan_T[..., None], 0.05), 0, 255)
        inpan = np.zeros(img.shape[:2], bool)
        for pp in spec.get("panels", []):
            _b = pp["bbox"]; inpan[max(0, _b[1]):_b[3], max(0, _b[0]):_b[2]] = True
        sel_ok = (brush_patch > 0) & (rem > 0.15) & ((pan_T < 0.999) | (~inpan))  # 不透明板后面不动；仅毛笔书法区
        wmix = np.clip((0.85 - (1 - rem)) / 0.35, 0, 1) * sel_ok                # 浓墨(>0.85)处反解不稳，保留修补；其余全信反解
        wmix = cv2.GaussianBlur(wmix.astype(np.float32), (5, 5), 0)
        cur = (cur.astype(np.float32) * (1 - wmix[..., None]) + Bs * wmix[..., None]).astype(np.uint8)
    # 3) 人物：SAM2 实例掩膜界定"这个人"的范围（治重叠人物粘连），u2net 软 alpha 在范围内保发丝
    person_layers = []; person_alpha = np.zeros(cur.shape[:2], np.float32)
    person_alpha_full = person_alpha.copy(); _punch_people = []   # 人物全 alpha（供最终擦除）与待冲孔记账
    _pboxes = assign_person_boxes(cur, spec.get("persons", [])) if spec.get("persons") else {}
    for h in spec.get("persons", []):
        hb, note = _pboxes.get(h["id"], (h["bbox"], None))
        if note: print(f"p{n:02d} 人物 {h['id']} {note}：{h['bbox']} → {hb}", flush=True)
        _prev = None   # (a, hb, area)
        for _try in range(3):   # 贴缘扩框重抠：起草/检测对躺姿等姿态常只框住半个人（p21 案），
            soft = rembg_alpha(cur, hb, "u2net_human_seg")   # alpha 大面积贴在框内缘=人被切断，向被切方向扩框重抠
            inst = sam_alpha(cur, hb)
            agree = inst is not None and (soft * (inst > 0.5)).sum() > 0.35 * max(1.0, (soft > 0.5).sum())
            if agree:   # SAM 实例与人像分割吻合 → 用实例门切开重叠人物，u2net 软 alpha 在门内保发丝
                gate = cv2.dilate((inst > 0.5).astype(np.float32), np.ones((9, 9), np.uint8))
                a = solidify_alpha(soft * gate)
                print(f"p{n:02d} 人物 {h['id']} SAM实例门 生效", flush=True)
            else:       # SAM 缺席或与人像分割打架 → 信人像分割（老行为）
                a = solidify_alpha(soft)
                if inst is not None: print(f"p{n:02d} 人物 {h['id']} SAM与人像分割不一致，回退 u2net", flush=True)
            _area = int((a > 0.5).sum())
            if _prev is not None and _area < _prev[2] * 1.15:
                # 扩框没带来 15% 以上增量=原框并没切住人（检测框天然贴身也会触发贴缘），回退原结果——
                # 大框重抠人占比变小、分割质量反降（p13 脸部斑点洞案）
                a, hb = _prev[0], _prev[1]
                print(f"p{n:02d} 人物 {h['id']} 扩框无增量，回退原框", flush=True)
                break
            _Hh, _Ww = a.shape[:2]
            _x0, _y0, _x1, _y1 = [int(v) for v in hb]
            _ys, _xs = np.where(a > 0.5)
            if not len(_xs): break
            _t = [int(_xs.min()), int(_ys.min()), int(_xs.max()), int(_ys.max())]   # alpha 实际紧框
            _exp = [0, 0, 0, 0]   # 紧框够到输入框缘（±5px）=实体可能被框切断；页面边界豁免（人立在页底属正常）
            if _x0 > 4 and _t[0] <= _x0 + 5: _exp[0] = int((_x1 - _x0) * 0.30)
            if _y0 > 4 and _t[1] <= _y0 + 5: _exp[1] = int((_y1 - _y0) * 0.30)
            if _x1 < _Ww - 4 and _t[2] >= _x1 - 5: _exp[2] = int((_x1 - _x0) * 0.30)
            if _y1 < _Hh - 4 and _t[3] >= _y1 - 5: _exp[3] = int((_y1 - _y0) * 0.30)
            if not any(_exp) or _try == 2: break
            _prev = (a, hb, _area)
            hb = [max(0, _x0 - _exp[0]), max(0, _y0 - _exp[1]), min(_Ww, _x1 + _exp[2]), min(_Hh, _y1 + _exp[3])]
            print(f"p{n:02d} 人物 {h['id']} 贴缘扩框重抠 → {hb}", flush=True)
        # 内部封闭小洞填实：人是实心的——脸部高光/衣纹被分割误判成背景会留斑点洞（星空底透出）。
        # 只填不连通外部的小洞（<1200px）；胳膊圈出的真实镂空是大洞或连通外部，不受影响
        _mh = (a > 0.5).astype(np.uint8)
        _mp = np.pad(_mh, 1)
        _ff = _mp.copy(); _ffm = np.zeros((_mp.shape[0] + 2, _mp.shape[1] + 2), np.uint8)
        cv2.floodFill(_ff, _ffm, (0, 0), 1)
        _holes = ((_mp == 0) & (_ff == 0))[1:-1, 1:-1].astype(np.uint8)
        if _holes.any():
            _nh, _lh, _sh, _ch = cv2.connectedComponentsWithStats(_holes, 8)
            _fill = np.zeros_like(_holes)
            for _hi in range(1, _nh):
                if _sh[_hi, cv2.CC_STAT_AREA] < 1200: _fill[_lh == _hi] = 1
            if _fill.any():
                a = np.where(_fill > 0, np.maximum(a, 0.98), a)
                print(f"p{n:02d} 人物 {h['id']} 填实内部斑点洞 {int(_fill.sum())}px", flush=True)
        # 人物让位已声明物件：SAM 分人常把手持物（手机等）一并划进人物 alpha，
        # 物件抠取又按 person_alpha 排他——物件就只剩碎片。做法分两步：
        # ①排他掩膜（person_alpha）在物件框内挖洞，把框内像素让给物件去认领；
        # ②人物层本体先存全 alpha，物件抠完后按物件"真实 alpha"冲孔重存——
        #   若直接按 bbox 挖矩形洞，框内非物件的像素（手机旁的衬衫）谁都不认领，
        #   会被钉死在背景上成为拖动残留
        _a_excl = a
        if _obj_guard:
            _a_excl = a.copy(); _pc = 0
            for _g in _obj_guard:
                gy0, gy1 = max(0, _g[1]), min(a.shape[0], _g[3])
                gx0, gx1 = max(0, _g[0]), min(a.shape[1], _g[2])
                if gy1 > gy0 and gx1 > gx0:
                    _pc += int((_a_excl[gy0:gy1, gx0:gx1] > 0.5).sum()); _a_excl[gy0:gy1, gx0:gx1] = 0.0
            if _pc: print(f"p{n:02d} 人物 {h['id']} 排他让位物件框 {_pc}px", flush=True)
        rgba, box = alpha_to_rgba(cur, a)
        if rgba is None: print(f"p{n:02d} person {h['id']} empty"); continue
        fn = _ufn(h["name"]); save_rgba(f"{out_dir}/{fn}", rgba)
        person_layers.append({"type": "person", "id": h["id"], "name": h["name"], "file": fn, "box": box})
        _punch_people.append((fn, a, person_layers[-1]))
        person_alpha = np.maximum(person_alpha, _a_excl)
        person_alpha_full = np.maximum(person_alpha_full, a)
        print(f"p{n:02d} 人物 {h['id']} box={box} area={(a>0.5).sum()}", flush=True)
    # 4) 物品（显著性 − 人物）
    object_layers = []; obj_alpha = np.zeros(cur.shape[:2], np.float32)
    for o in spec.get("objects", []):
        method = o.get("method") or ("table" if "桌" in o["name"] or "船" in o["name"] else "grabcut")
        cboxes, cnote = composite_boxes(cur, o["name"], o["bbox"], o.get("hint_en"))
        if cboxes:   # 组合物件：逐成员 SAM 后掩膜求并
            print(f"p{n:02d} 物品 {o['id']} {cnote}", flush=True)
            bx = [min(b[0] for b in cboxes), min(b[1] for b in cboxes),
                  max(b[2] for b in cboxes), max(b[3] for b in cboxes)]
            a_sam = None
            for cb in cboxes:
                ai_ = sam_alpha(cur, cb, expand=0.15)
                if ai_ is not None:
                    a_sam = ai_ if a_sam is None else np.maximum(a_sam, ai_)
        else:
            bx, note = refine_box(cur, o["name"], o["bbox"], o.get("hint_en"))
            if note: print(f"p{n:02d} 物品 {o['id']} {note}：{o['bbox']} → {bx}", flush=True)
            a_sam = sam_alpha(cur, bx, expand=0.10)
        cov = 0.0
        if a_sam is not None:
            a_sam = a_sam * (1 - np.clip(person_alpha * 1.5, 0, 1))       # 不抢已抠走的人物像素
            cov = float((a_sam > 0.5).sum()) / max(1.0, (bx[2] - bx[0]) * (bx[3] - bx[1]))
        if a_sam is not None and cov > 0.03:
            _sc = score_object_alpha(a_sam, bx)
            if _sc["fill"] < 0.30 or _sc["compact"] < 0.20 or _sc["touch_edge"] < 0.30:
                # 双验收：抠出来的不像一件完整物品（大虚框内碎块/背景泄漏）——宁缺毋滥，留在背景里视觉无损
                o["notes"] = (o.get("notes") or "") + "｜层验收未过（覆盖/紧凑/贴边），留在背景；可收紧框重试"
                print(f"p{n:02d} 物品 {o['id']} 验收不过 fill{_sc['fill']:.2f} compact{_sc['compact']:.2f} edge{_sc['touch_edge']:.2f}，留背景", flush=True)
                continue
            a = a_sam
            print(f"p{n:02d} 物品 {o['id']} 用 SAM2（覆盖率 {cov:.2f}）", flush=True)
        else:                                                             # SAM 拿不到 → 老方法兜底
            sal = rembg_alpha(cur, bx, "isnet-general-use", margin_frac=0.08)
            sal = sal * (1 - np.clip(person_alpha * 1.5, 0, 1))
            if method == "table":
                a = table_alpha(cur, bx, exclude_alpha=person_alpha, seed_alpha=sal)
            elif method == "grabcut":
                a = grabcut_alpha(cur, bx, exclude_alpha=person_alpha, seed_alpha=sal)
                if a is None: a = sal
            else:
                a = sal
        a = solidify_alpha(a)
        # 去小碎块
        m = (a > 0.3).astype(np.uint8); nn, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if nn > 1:
            big = stats[1:, cv2.CC_STAT_AREA].max()
            for i in range(1, nn):
                if stats[i, cv2.CC_STAT_AREA] < 0.02 * big: a[lab == i] = 0
        rgba, box = alpha_to_rgba(cur, a)
        if rgba is None: print(f"p{n:02d} object {o['id']} empty"); continue
        fn = _ufn(o["name"]); save_rgba(f"{out_dir}/{fn}", rgba)
        object_layers.append({"type": "object", "id": o["id"], "name": o["name"], "file": fn, "box": box})
        obj_alpha = np.maximum(obj_alpha, a)
        print(f"p{n:02d} 物品 {o['id']} box={box} area={(a>0.5).sum()}", flush=True)
    # 4.5) 自动补检：通用物件词表全图检测，未被认领的高置信物件直接补层并回写标注
    #      （治"起草漏标物品"——画面里有什么不该由语言模型说了算）
    #      放在物品循环之后：排重用"成品框"而非起草粗框——起草框脱靶时 IoU 挡不住，
    #      精修后撞到同一个检测框就重复抠层（茶具×2 之祸）；小件对大并集框还要按被包含率排除
    _xcfg = {}
    try:
        _xcfg = json.load(open(os.path.join(os.path.dirname(deck_dir(deck)), "..", "slidelift.json"))).get("extract") or {}
    except Exception:
        pass
    _hard = [L["box"] for L in person_layers + object_layers + cal_layers]
    _soft = [L["box"] for L in panel_layers] + [p["bbox"] for p in spec.get("panels", [])]
    if _xcfg.get("mode") == "standard":      # 标准：只补高把握的少量散件
        found = find_unclaimed_objects(cur, _soft + _hard, max_n=6, conf=0.25, contain_boxes=_hard)
    else:                                    # 全部检出（默认）：散件物品尽数成层
        found = find_unclaimed_objects(cur, _soft + _hard, max_n=12, conf=0.15, contain_boxes=_hard)
    if found:
        seen_names = {o["name"] for o in spec.get("objects", [])}
        existing_ids = {o.get("id") for o in spec.get("objects", [])}
        adds = []; oxn = 0
        for b, c, nm in found:
            a_sam = sam_alpha(cur, b, expand=0.10)
            if a_sam is None:
                print(f"p{n:02d} 补检 {nm} SAM 无掩膜，放弃", flush=True); continue
            a_sam = a_sam * (1 - np.clip(person_alpha * 1.5, 0, 1))
            if float((a_sam > 0.5).sum()) / max(1.0, (b[2] - b[0]) * (b[3] - b[1])) <= 0.03:
                print(f"p{n:02d} 补检 {nm} 覆盖过低，放弃", flush=True); continue
            _sc2 = score_object_alpha(a_sam, b)
            if _sc2["fill"] < 0.30 or _sc2["compact"] < 0.20 or _sc2["touch_edge"] < 0.30:
                print(f"p{n:02d} 补检 {nm} 验收不过 fill{_sc2['fill']:.2f} compact{_sc2['compact']:.2f}，放弃", flush=True); continue
            a = solidify_alpha(a_sam)
            m = (a > 0.3).astype(np.uint8); nn_, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
            if nn_ > 1:
                big = stats[1:, cv2.CC_STAT_AREA].max()
                for i in range(1, nn_):
                    if stats[i, cv2.CC_STAT_AREA] < 0.02 * big: a[lab == i] = 0
            rgba, box = alpha_to_rgba(cur, a)
            if rgba is None: continue
            name = f"物品_{nm}"; k2 = 2
            while name in seen_names: name = f"物品_{nm}{k2}"; k2 += 1
            seen_names.add(name)
            oxn += 1
            while f"ox{oxn}" in existing_ids: oxn += 1
            existing_ids.add(f"ox{oxn}")
            fn = _ufn(name); save_rgba(f"{out_dir}/{fn}", rgba)
            object_layers.append({"type": "object", "id": f"ox{oxn}", "name": name, "file": fn, "box": box})
            obj_alpha = np.maximum(obj_alpha, a)
            adds.append({"id": f"ox{oxn}", "name": name, "bbox": [int(v) for v in b], "method": "saliency",
                         "notes": f"引擎自动补检（{nm} {c:.2f}），不需要可在标注里删除"})
            print(f"p{n:02d} 补检 {name} box={box} area={(a>0.5).sum()}", flush=True)
        if adds:
            try:   # 只回写抠成功的 → 界面可见、可删；重跑幂等（已回写的按成品框排重）
                spec.setdefault("objects", []).extend(adds)
                json.dump(spec, open(f"{deck_dir(deck)}/spec/p{n:02d}.json", "w"), ensure_ascii=False, indent=1)
            except Exception as e:
                print(f"p{n:02d} 补检回写失败: {e}", flush=True)
    # 5) 背景：把人物+物品的"实体核心"修补掉。只擦 alpha>0.35 的核心+少量膨胀——
    #    软边/辉光留在背景原像素上：不动图层时与原图逐像素一致；擦得越宽，修补脑补越多、贴回越露馅
    if _punch_people and obj_alpha.max() > 0:
        _oa = np.clip(obj_alpha * 1.5, 0, 1)
        for _fn, _pa, _entry in _punch_people:
            _pp = _pa * (1 - _oa)
            if float((_pa - _pp).sum()) < 30: continue      # 没被物件压住的人不动
            _rg, _bx = alpha_to_rgba(cur, _pp)
            if _rg is None: continue
            save_rgba(f"{out_dir}/{_fn}", _rg); _entry["box"] = _bx
            print(f"p{n:02d} 人物层 {_entry['id']} 按物件实形冲孔 {int((_pa - _pp).sum())}px", flush=True)
    _core = np.maximum(person_alpha_full, obj_alpha)
    bg = remove_and_fill(cur, (_core > 0.35).astype(np.float32), dilate_px=4, margin=300) if (person_layers or object_layers) else cur
    save_rgb(f"{out_dir}/背景.png", bg)
    manifest["layers"] = [{"type": "background", "name": "背景", "file": "背景.png", "box": [0, 0, img.shape[1], img.shape[0]]}] + person_layers + object_layers + panel_layers + cal_layers
    # 图层有效性：alpha 覆盖率（相对自身 bbox），过低=抠取疑似失败（框偏/目标没抓到）——供界面与报告告警
    for L in manifest["layers"]:
        if L["type"] == "background": continue
        try:
            im4 = cv2.imread(f"{out_dir}/{L['file']}", cv2.IMREAD_UNCHANGED)
            if im4 is not None and im4.ndim == 3 and im4.shape[2] == 4:
                bw = max(1, (L["box"][2] - L["box"][0]) * (L["box"][3] - L["box"][1]))
                L["alpha_ratio"] = round(float((im4[:, :, 3] > 40).sum()) / bw, 4)
        except Exception: pass
    json.dump(manifest, open(f"{out_dir}/manifest.json", "w"), ensure_ascii=False, indent=1)
    print(f"p{n:02d} layers done {time.time()-t:.1f}s", flush=True)

os._exit(0)
