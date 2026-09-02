"""解包"整屏位图型" PPTX → work/<deck>/src/pNN.png + deck.json
用法: unpack_pptx.py <file.pptx> <deck> [--force]
规则: 每页取面积最大的图片作为整屏源图；尺寸≠1672×941 时（与 16:9 容差 3% 内）重采样到
1672×941 并在 deck.json 标记 resized（流水线 common.py 按 1672×941 标定）。"""
import sys, os, re, io, json, zipfile, time, shutil
from PIL import Image
from unpack_images import to_canvas   # 任意尺寸归一 1672×941（方图/竖图=舞台式衬底）

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TW, TH = 1672, 941

def rels_map(z, path):
    if path not in z.namelist(): return {}
    xml = z.read(path).decode("utf-8", "ignore")
    return dict(re.findall(r'Id="([^"]+)"[^>]*?Target="([^"]+)"', xml))

def slide_order(z):
    pres = z.read("ppt/presentation.xml").decode("utf-8", "ignore")
    rid2t = rels_map(z, "ppt/_rels/presentation.xml.rels")
    out = []
    for rid in re.findall(r'<p:sldId[^>]*r:id="([^"]+)"', pres):
        t = rid2t.get(rid, "")
        t = t[1:] if t.startswith("/") else "ppt/" + t
        out.append(os.path.normpath(t).replace("\\", "/"))
    return out

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 2:
        print("用法: unpack_pptx.py <file.pptx> <deck> [--force]"); sys.exit(1)
    pptx, deck = args[0], args[1]
    force = "--force" in sys.argv
    dd = os.path.join(ROOT, "work", deck)
    fresh = not os.path.exists(dd)
    if os.path.exists(os.path.join(dd, "src")) and not force:
        print(f"ERROR: work/{deck}/src 已存在（加 --force 覆盖）"); sys.exit(2)
    z = zipfile.ZipFile(pptx)
    slides = slide_order(z)
    if not slides:
        print("ERROR: 未在 pptx 中找到幻灯片列表"); sys.exit(2)
    os.makedirs(os.path.join(dd, "src"), exist_ok=True)
    names = set(z.namelist())
    resized = False; padded = False; orig_sizes = set(); missing = []; n_ok = 0
    for i, sp in enumerate(slides, 1):
        sx = z.read(sp).decode("utf-8", "ignore")
        srel = rels_map(z, sp.replace("slides/", "slides/_rels/") + ".rels")
        best = None  # (area, media_path)
        for m in re.finditer(r'<p:pic>.*?</p:pic>', sx, re.S):
            pic = m.group(0)
            rid = re.search(r'r:embed="([^"]+)"', pic)
            ext = re.search(r'<a:ext cx="(\d+)" cy="(\d+)"', pic)
            if not rid: continue
            t = srel.get(rid.group(1), "")
            t = os.path.normpath(os.path.join("ppt/slides", t)).replace("\\", "/")
            area = int(ext.group(1)) * int(ext.group(2)) if ext else 0
            if t in names and (best is None or area > best[0]):
                best = (area, t)
        if best is None:   # 兜底：图片可能在背景填充/形状填充里（<p:bg>/<p:sp> 的 blipFill）——取 rels 里最大的图
            cand = []
            for t in srel.values():
                t2 = os.path.normpath(os.path.join("ppt/slides", t)).replace("\\", "/")
                if t2 in names and re.search(r"\.(png|jpe?g|webp|bmp|gif|tiff?)$", t2, re.I):
                    cand.append((z.getinfo(t2).file_size, t2))
            if cand:
                best = max(cand)
        if best is None:
            missing.append(i); continue
        try:
            im = Image.open(io.BytesIO(z.read(best[1])))
            cv, mode, _box = to_canvas(im)   # 任意尺寸归一：16:9 重采样；方图/竖图舞台式衬底
        except Exception as e:
            print(f"ERROR: 第{i}页图片 {best[1]} 无法解码（{e}）")
            if n_ok == 0 and fresh: shutil.rmtree(dd, ignore_errors=True)
            sys.exit(3)
        orig_sizes.add(im.size)
        if im.size != (TW, TH): resized = True
        if mode == "pad": padded = True
        cv.save(os.path.join(dd, "src", f"p{i:02d}.png"))
        n_ok += 1
        print(f"p{i:02d} <- {best[1]} {im.size[0]}x{im.size[1]} {mode}", flush=True)
    if missing:
        print(f"ERROR: 第 {missing} 页没有图片元素（本工具只处理每页一张整屏位图的 PPT）")
        if n_ok == 0 and fresh: shutil.rmtree(dd, ignore_errors=True)   # 失败上传不留空壳
        sys.exit(4)
    json.dump({"name": deck, "source_pptx": os.path.abspath(pptx), "pages": n_ok,
               "orig_size": [list(s) for s in sorted(orig_sizes)], "resized": resized, "padded": padded,
               "created": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(os.path.join(dd, "deck.json"), "w"), ensure_ascii=False, indent=1)
    print(f"DONE {n_ok} 页 -> work/{deck}/src  resized={resized}")

if __name__ == "__main__":
    main()
