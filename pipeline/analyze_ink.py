"""对指定页运行 zone 墨迹分析，输出 work/<deck>/ink/pNN.json 与可视化 pNN_ink.png"""
import sys, json, os, numpy as np, cv2
from ppt2layers.common import load_spec, load_src, deck_dir, save_rgb
from ppt2layers.ink import analyze_zone
deck = sys.argv[1]; pages = [int(p) for p in sys.argv[2:]]
for n in pages:
    spec = load_spec(deck, n); img = load_src(deck, n)
    kinds = {p["id"]: p["kind"] for p in spec.get("panels", [])}
    vis = img.copy(); out = {"page": n, "zones": []}
    for z in spec["text_zones"]:
        ck = "bg"
        if ":" in z.get("container", "bg"): ck = kinds.get(z["container"].split(":")[1], "card")
        r = analyze_zone(img, z, ck)
        zi = {k: v for k, v in r.items() if k not in ("mask", "d")}; zi["id"] = z["id"]; zi["font"] = z["font"]
        out["zones"].append(zi)
        # 可视化：zone 框蓝，行框绿/红
        x0, y0, x1, y1 = z["bbox"]; cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 120, 255), 1)
        for L in r["lines"]:
            a, b, c, d = L["bbox"]; col = (0, 200, 0) if L["ok"] else (255, 0, 0)
            cv2.rectangle(vis, (a, b), (c, d), col, 1)
        flag = "" if not r["polarity_warn"] else " POLWARN"
        print(f"p{n:02d} {z['id']:>4} pol={r['polarity']}(auto {r['polarity_auto']} {r['polarity_scores']}){flag} k={r['k']} frac={r['mask_frac']:.3f} lines {r['n_found']}/{r['n_expected']} color={r['color']} glow={r['glow']}")
        for L in r["lines"]:
            print(f"      L{L['idx']} {L['bbox']} n={L['n']} adv={L['advance']} ratio={L['ratio']} {'OK' if L['ok'] else 'BAD'} {L['text'][:18]}")
    os.makedirs(f"{deck_dir(deck)}/ink", exist_ok=True)
    json.dump(out, open(f"{deck_dir(deck)}/ink/p{n:02d}.json", "w"), ensure_ascii=False, indent=1)
    save_rgb(f"{deck_dir(deck)}/ink/p{n:02d}_ink.png", vis)
