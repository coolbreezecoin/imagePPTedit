"""阶段：组装最终 PPTX。用法：assemble.py <deck> <out.pptx> [pages...]（缺省全部 1..N；没有 spec/图层的页用原图整页占位）"""
import sys, json, os, glob
import numpy as np
from ppt2layers.common import load_spec, deck_dir, W0, H0, ROOT, line_text
from ppt2layers.pptx_build import new_prs, blank_slide, add_textbox, add_picture, PT2PX
from ppt2layers.calib import textbox_geom
deck = sys.argv[1]; out = sys.argv[2]
dd = deck_dir(deck)
n_src = len(glob.glob(f"{dd}/src/p*.png"))
pages = [int(p) for p in sys.argv[3:]] or list(range(1, n_src + 1))
params = json.load(open(f"{dd}/calib/params.json")) if os.path.exists(f"{dd}/calib/params.json") else {}
prs = new_prs(); report = []
for n in pages:
    s = blank_slide(prs)
    man_path = f"{dd}/layers/p{n:02d}/manifest.json"
    if not os.path.exists(man_path) or not os.path.exists(f"{dd}/spec/p{n:02d}.json"):
        add_picture(s, f"{dd}/src/p{n:02d}.png", [0, 0, W0, H0], name="原图_未处理"); report.append((n, "placeholder")); continue
    spec = load_spec(deck, n); man = json.load(open(man_path))
    ink = json.load(open(f"{dd}/ink/p{n:02d}.json")); zinfo = {z["id"]: z for z in ink["zones"]}
    # 用户成品编辑（审校台"编辑成品"模式）：挪位 dx/dy、隐藏 hidden——只改组装，不碰引擎标注
    _ed = spec.get("edits") or {}
    edL, edZ = (_ed.get("layers") or {}), (_ed.get("zones") or {})
    for L in man["layers"]:
        e = edL.get(str(L.get("id") or "")) or edL.get(L["name"]) or {}
        if L.get("type") != "background" and e.get("hidden"): continue
        bx = L["box"]
        dx, dy = int(e.get("dx") or 0), int(e.get("dy") or 0)
        if dx or dy: bx = [bx[0] + dx, bx[1] + dy, bx[2] + dx, bx[3] + dy]
        add_picture(s, f"{dd}/layers/p{n:02d}/{L['file']}", bx, name=L["name"])
    nz = 0
    for z in spec["text_zones"]:
        ez = edZ.get(z["id"]) or {}
        if ez.get("hidden"): continue
        p = params.get(str(n), {}).get(z["id"])
        if not p: report.append((n, f"zone {z['id']} no params")); continue
        info = zinfo.get(z["id"], {})
        # 颜色优先级：用户手动指定（color_user）> 原图实测 > spec 默认——否则界面改色会被实测色无声覆盖
        color = (z.get("color") if z.get("color_user") else None) or info.get("color") or z.get("color", "#000000")
        glow = None
        if info.get("glow", {}).get("has"):
            r_px = info["glow"]["radius"]; glow = {"radius_pt": r_px * 0.6 / PT2PX, "color": color, "alpha": 0.4}
        x, y, w = textbox_geom(p, z["lines"])
        x += int(ez.get("dx") or 0); y += int(ez.get("dy") or 0)
        # 段首缩进：行文本以全角空格开头 → 去空格、按测得墨迹 x0 设精确 indent
        indents = []
        meas_by_idx = {L["idx"]: L for L in info.get("lines", [])}
        flush_x0 = [meas_by_idx[li]["bbox"][0] for li, ln in enumerate(z["lines"])
                    if li in meas_by_idx and not line_text(ln).startswith("　")]
        base_x0 = min(flush_x0) if flush_x0 else None
        for li, ln in enumerate(z["lines"]):
            txt = line_text(ln)
            if txt.startswith("　") and li in meas_by_idx and p["align"] == "left":
                if base_x0 is not None:
                    indents.append(max(0.0, meas_by_idx[li]["bbox"][0] - base_x0))
                else:
                    indents.append(max(0.0, meas_by_idx[li]["bbox"][0] - (x + 0.06 * p["size_pt"] * PT2PX)))
            else:
                indents.append(0.0)
        def strip_lead(ln):
            if isinstance(ln, str): return ln.lstrip("　")
            rr = [dict(r) for r in ln["runs"]]
            if rr: rr[0]["t"] = rr[0]["t"].lstrip("　")
            return {"runs": rr}
        z_lines = [strip_lead(ln) if ind > 0 else ln for ln, ind in zip(z["lines"], indents)]
        # runs 富文本：把测得的各 run 颜色填回（spec 未指定颜色的 run）
        lines_out = []
        for li, ln in enumerate(z_lines):
            if isinstance(ln, str): lines_out.append(ln); continue
            meas = next((L for L in info.get("lines", []) if L["idx"] == li), None)
            rc = meas.get("run_colors") if meas else None
            runs = []
            for ri, r in enumerate(ln["runs"]):
                rr = dict(r)
                if not rr.get("color"):
                    if z.get("color_user"): rr["color"] = color          # 用户改色时整块统一用户色
                    elif rc and ri < len(rc) and rc[ri]: rr["color"] = rc[ri]
                runs.append(rr)
            lines_out.append({"runs": runs})
        outline = None
        ol = info.get("outline")
        bold = z.get("bold", False)
        if ol and ol.get("strength", 0) > 0.25:
            # 实测最接近原图观感的组合：加粗 + 细描边（描边一半吃进字形内，正好抵消加粗的过重）
            wpx = float(np.clip(ol["width_px"] * 0.5, 1.0, 1.6))
            outline = {"mode": "ln", "color": ol["color"], "width_pt": wpx / PT2PX, "alpha": 1.0}
            bold = True
        add_textbox(s, lines_out, x, y, w, p["size_pt"], font=z["font"], bold=bold, color=color,
                    align=p["align"], spc_pt=p.get("spc_pt", 0), pitch_pt=(p.get("pitch_pts") or p["pitch_pt"]), indents_px=indents,
                    name=f"文字_{line_text(z['lines'][0])[:8]}", glow=glow, outline=outline, latin_font=("Arial" if z["font"] in ("微软雅黑", "黑体") else None))
        nz += 1
    report.append((n, f"ok layers={len(man['layers'])} zones={nz}"))
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
prs.save(out)
json.dump(pages, open(out + ".pages.json", "w"))
for r in report: print(r)
print("saved", out)
