# -*- coding: utf-8 -*-
"""解包"图片集" → work/<deck>/src/pNN.png + deck.json
用法: unpack_images.py <deck> <img1> [img2 ...] [--force]
规则: 每张图一页，按文件名自然序排列；统一归一到 1672×941（流水线按此标定）：
  - 宽高比与 16:9 相差 ≤3% → 直接 LANCZOS 重采样；
  - 方图/竖图/超宽 → 等比缩放居中放入画布，露出的部分用同图"放大模糊+压暗"补满
    （舞台式衬底：不变形、不难看，衬底会自然归入背景层）；
  尺寸不一的多张图可混传。"""
import sys, os, re, json, time, shutil
from PIL import Image, ImageFilter, ImageOps, ImageEnhance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TW, TH = 1672, 941


def natkey(s):
    """文件名自然序：p2 < p10。"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", os.path.basename(s))]


def to_canvas(im):
    """任意尺寸 → 1672×941 页图。返回 (canvas, mode, content_box)；mode ∈ resize|pad。"""
    im = ImageOps.exif_transpose(im).convert("RGB")
    ar, tar = im.width / im.height, TW / TH
    if abs(ar - tar) / tar <= 0.03:
        return im.resize((TW, TH), Image.LANCZOS), "resize", [0, 0, TW, TH]
    if ar > tar:
        w, h = TW, max(1, round(TW / ar))
    else:
        h, w = TH, max(1, round(TH * ar))
    fg = im.resize((w, h), Image.LANCZOS)
    scale = max(TW / im.width, TH / im.height)          # 衬底：cover 放大 + 模糊 + 压暗
    bw, bh = round(im.width * scale), round(im.height * scale)
    bg = im.resize((bw, bh), Image.LANCZOS).filter(ImageFilter.GaussianBlur(24))
    bg = ImageEnhance.Brightness(bg).enhance(0.72)
    canvas = bg.crop(((bw - TW) // 2, (bh - TH) // 2, (bw - TW) // 2 + TW, (bh - TH) // 2 + TH))
    x, y = (TW - w) // 2, (TH - h) // 2
    canvas.paste(fg, (x, y))
    return canvas, "pad", [x, y, x + w, y + h]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    if len(args) < 2:
        print("用法: unpack_images.py <deck> <img1> [img2 ...] [--force]"); sys.exit(1)
    deck, imgs = args[0], sorted(args[1:], key=natkey)
    dd = os.path.join(ROOT, "work", deck)
    fresh = not os.path.exists(dd)
    if os.path.exists(os.path.join(dd, "src")) and not force:
        print(f"ERROR: work/{deck}/src 已存在（加 --force 覆盖）"); sys.exit(2)
    os.makedirs(os.path.join(dd, "src"), exist_ok=True)
    os.makedirs(os.path.join(dd, "orig"), exist_ok=True)   # 原件副本（转正后）：差量导出以它为底，未修改区域逐像素保真
    items = []
    for i, p in enumerate(imgs, 1):
        try:
            im = Image.open(p)
            im0 = ImageOps.exif_transpose(im).convert("RGB")   # 与 to_canvas 同一转正，坐标系一致
            cv, mode, box = to_canvas(im)
        except Exception as e:
            print(f"ERROR: {os.path.basename(p)} 无法解码（{e}）")
            if not items and fresh:
                shutil.rmtree(dd, ignore_errors=True)   # 一页没成的失败上传不留空壳
            sys.exit(3)
        cv.save(os.path.join(dd, "src", f"p{i:02d}.png"))
        im0.save(os.path.join(dd, "orig", f"p{i:02d}.png"))
        items.append({"file": os.path.basename(p), "orig": list(im0.size), "fit": mode, "content_box": box})
        print(f"p{i:02d} <- {os.path.basename(p)} {im.size[0]}x{im.size[1]} {mode}", flush=True)
    json.dump({"name": deck, "source": "images", "pages": len(items), "items": items,
               "orig_size": sorted({tuple(it["orig"]) for it in items}),
               "resized": True, "padded": any(it["fit"] == "pad" for it in items),
               "created": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(os.path.join(dd, "deck.json"), "w"), ensure_ascii=False, indent=1)
    print(f"DONE {len(items)} 页 -> work/{deck}/src")


if __name__ == "__main__":
    main()
