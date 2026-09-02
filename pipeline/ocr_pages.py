# 对 work/<deck>/src/*.png 跑 RapidOCR，输出 ocr/pNN.json + ocr/pNN_vis.png（候选文字，供人工核对）
import sys, json, glob, os
import numpy as np, cv2
from rapidocr_onnxruntime import RapidOCR
deck = sys.argv[1]
src_dir = f"work/{deck}/src"; out_dir = f"work/{deck}/ocr"; os.makedirs(out_dir, exist_ok=True)
ocr = RapidOCR()
for p in sorted(glob.glob(f"{src_dir}/p*.png")):
    name = os.path.splitext(os.path.basename(p))[0]
    img = cv2.imread(p)
    # 放大 2 倍提升小字识别率
    big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    res, _ = ocr(big)
    items = []
    vis = img.copy()
    if res:
        for box, txt, conf in res:
            pts = (np.array(box, dtype=np.float32) / 2.0)
            x0, y0 = pts.min(0); x1, y1 = pts.max(0)
            items.append({"text": txt, "conf": round(float(conf), 3),
                          "box": [round(float(x0),1), round(float(y0),1), round(float(x1),1), round(float(y1),1)]})
            cv2.polylines(vis, [pts.astype(np.int32)], True, (0, 0, 255), 1)
    json.dump(items, open(f"{out_dir}/{name}.json", "w"), ensure_ascii=False, indent=1)
    cv2.imwrite(f"{out_dir}/{name}_vis.png", vis)
    print(name, len(items), "items", flush=True)
