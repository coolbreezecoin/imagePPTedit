"""阶段：QA。渲染最终 pptx → 与原图对比：MAE + 并排图（原图|重建|差异）。用法：qa_pages.py <deck> <pptx> [pages...]"""
import sys, json, os, numpy as np, cv2, glob
from ppt2layers.common import deck_dir, load_src, W0, H0
from ppt2layers.pptx_build import export_pdf, render_pdf
deck = sys.argv[1]; pptx = sys.argv[2]
dd = deck_dir(deck); qd = f"{dd}/qa"; os.makedirs(qd, exist_ok=True)
pdf = os.path.splitext(os.path.abspath(pptx))[0] + ".pdf"
export_pdf(pptx, pdf); renders = render_pdf(pdf, width=W0)
order = json.load(open(pptx + ".pages.json")) if os.path.exists(pptx + ".pages.json") else list(range(1, len(renders) + 1))
sel = [int(p) for p in sys.argv[3:]] or order
res = {}
for n in sel:
    if n not in order: continue
    rr = renders[order.index(n)]; src = load_src(deck, n)
    if rr.shape != src.shape: rr = cv2.resize(rr, (src.shape[1], src.shape[0]))
    diff = np.abs(rr.astype(int) - src.astype(int)).mean(axis=2)
    mae = float(diff.mean()); res[n] = mae
    d3 = np.clip(diff * 3, 0, 255).astype(np.uint8); d3 = cv2.cvtColor(d3, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(f"{qd}/p{n:02d}_rebuilt.png", cv2.cvtColor(rr, cv2.COLOR_RGB2BGR))
    top = np.concatenate([src, rr], axis=1); bot = np.concatenate([d3, np.zeros_like(d3)], axis=1)
    sheet = np.concatenate([top, bot], axis=0)
    sheet = cv2.resize(sheet, (sheet.shape[1] // 2, sheet.shape[0] // 2), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f"{qd}/p{n:02d}_cmp.jpg", cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"p{n:02d} MAE={mae:.2f}/255 {'OK' if mae < 8 else 'CHECK'}")
json.dump(res, open(f"{qd}/mae.json", "w"), indent=1)
