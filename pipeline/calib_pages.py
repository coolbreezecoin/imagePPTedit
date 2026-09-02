"""阶段：文字几何标定。用法：calib_pages.py <deck> <iters> <pages...>；输出 work/<deck>/calib/params.json 与日志。"""
import sys, json, os, numpy as np, cv2, time
from ppt2layers.common import load_spec, deck_dir, W0, H0, line_text
from ppt2layers.pptx_build import new_prs, blank_slide, add_textbox, export_pdf, render_pdf, PX2PT, PT2PX, cal_font
from ppt2layers.calib import init_params, textbox_geom, measure_lines, update_params
deck = sys.argv[1]; iters = int(sys.argv[2]); pages = [int(p) for p in sys.argv[3:]]
cdir = f"{deck_dir(deck)}/calib"; os.makedirs(cdir, exist_ok=True)
pfile = f"{cdir}/params.json"
params = json.load(open(pfile)) if os.path.exists(pfile) else {}
specs = {n: load_spec(deck, n) for n in pages}
inks = {n: json.load(open(f"{deck_dir(deck)}/ink/p{n:02d}.json")) for n in pages}
for n in pages:
    params.setdefault(str(n), {})
    zi = {z["id"]: z for z in inks[n]["zones"]}
    for z in specs[n]["text_zones"]:
        if z["id"] in params[str(n)] and not params[str(n)][z["id"]].get("_reinit"): continue
        if z["id"] not in zi or not zi[z["id"]]["lines"]: print(f"p{n:02d} {z['id']} no ink lines, skip"); continue
        params[str(n)][z["id"]] = init_params(z, zi[z["id"]])
def build_deck(path):
    prs = new_prs(); order = []
    for n in pages:
        s = blank_slide(prs); order.append(n)
        zi = {z_["id"]: z_ for z_ in inks[n]["zones"]}
        for z in specs[n]["text_zones"]:
            p = params[str(n)].get(z["id"]);
            if not p: continue
            x, y, w = textbox_geom(p, z["lines"])
            # 加粗判定与组装保持一致（描边强→组装会加粗）：标定几何必须按最终字重来校
            ol = (zi.get(z["id"]) or {}).get("outline")
            bold_eff = z.get("bold", False) or bool(ol and ol.get("strength", 0) > 0.25)
            add_textbox(s, [line_text(t) for t in z["lines"]], x, y, w, p["size_pt"], font=cal_font(z["font"]), bold=bold_eff, color="#000000",
                        align=p["align"], spc_pt=p.get("spc_pt", 0), pitch_pt=(p.get("pitch_pts") or p["pitch_pt"]), name=f"calib_{z['id']}",
                        latin_font=("Arial" if z["font"] in ("微软雅黑", "黑体") else None))
    prs.save(path); return order
for it in range(iters):
    t = time.time()
    pptx_path = f"{cdir}/calib_iter{it}.pptx"; pdf_path = f"{cdir}/calib_iter{it}.pdf"
    order = build_deck(pptx_path); export_pdf(pptx_path, pdf_path); renders = render_pdf(pdf_path)
    stats = []
    for si, n in enumerate(order):
        img = renders[si]; zi = {z["id"]: z for z in inks[n]["zones"]}
        all_lines = [L["bbox"] for z_ in specs[n]["text_zones"] if z_["id"] in zi
                     for L in zi[z_["id"]]["lines"]]
        for z in specs[n]["text_zones"]:
            p = params[str(n)].get(z["id"])
            if not p: continue
            info = zi[z["id"]]; targets = [L["bbox"] for L in info["lines"]]
            own = {tuple(t) for t in targets}
            others = [b for b in all_lines if tuple(b) not in own]
            measured = measure_lines(img, targets, p["size_pt"] * PT2PX, p["pitch_pt"] * PT2PX, others=others)
            newp, err = update_params(p, z, info, measured)
            params[str(n)][z["id"]] = newp
            stats.append((n, z["id"], err))
    json.dump(params, open(pfile, "w"), ensure_ascii=False, indent=1)
    wr = [abs(e["w_rel"]) for _, _, e in stats if e["w_rel"] is not None]
    dx = [abs(e["dx"]) for _, _, e in stats if e["dx"] is not None]
    dy = [abs(e["dy"]) for _, _, e in stats if e["dy"] is not None]
    if not stats or not wr:   # 全卷无可标定文字（用户把标注全删/全成层）——np.max 对空数组会崩（u83 真实用户案）
        print(f"iter {it}: zones=0（本卷无可标定文字，跳过统计）", flush=True)
        continue
    print(f"iter {it}: zones={len(stats)} |w_rel| mean={np.mean(wr):.4f} max={np.max(wr):.4f} | |dx| mean={np.mean(dx):.2f}px max={np.max(dx):.2f} | |dy| mean={np.mean(dy):.2f}px max={np.max(dy):.2f}  ({time.time()-t:.1f}s)", flush=True)
    worst = sorted(stats, key=lambda s: -(abs(s[2]["w_rel"] or 0)))[:3]
    for n, zid, e in worst: print(f"   worst w: p{n:02d} {zid} {e}")
    # 保存最后一轮渲染供检查
    if it == iters - 1:
        for si, n in enumerate(order): cv2.imwrite(f"{cdir}/render_p{n:02d}.png", cv2.cvtColor(renders[si], cv2.COLOR_RGB2BGR))
os._exit(0)
