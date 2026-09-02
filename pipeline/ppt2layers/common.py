import os, json, re, numpy as np, cv2
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
W0, H0 = 1672, 941               # 本 deck 源图尺寸
REF_W = 3840                     # 规范中的参考宽度
def S(px_at_4k):                 # 规范里 4K 像素参数 → 当前分辨率
    return px_at_4k * W0 / REF_W
def odd(k): k=int(round(k)); return k if k%2==1 else k+1
def deck_dir(deck): return os.path.join(ROOT, "work", deck)
def load_spec(deck, n):
    return json.load(open(os.path.join(deck_dir(deck), "spec", f"p{n:02d}.json")))
def load_src(deck, n):
    p=os.path.join(deck_dir(deck), "src", f"p{n:02d}.png")
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
def save_rgb(path, rgb):
    os.makedirs(os.path.dirname(path), exist_ok=True); cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
def save_rgba(path, rgba):
    os.makedirs(os.path.dirname(path), exist_ok=True); cv2.imwrite(path, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
def gray_of(rgb): return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
def clip_box(b, W=W0, H=H0):
    x0,y0,x1,y1=b; return [max(0,int(x0)),max(0,int(y0)),min(W,int(x1)),min(H,int(y1))]
def hex2rgb(h): h=h.lstrip("#"); return tuple(int(h[i:i+2],16) for i in (0,2,4))
def rgb2hex(c): return "#%02X%02X%02X" % tuple(int(v) for v in c)
CJK_RE = re.compile(r'[一-鿿　-〿＀-￯]')
def is_cjk_char(ch): return bool(CJK_RE.match(ch))
def char_count(s):
    """步进计数：CJK/全角标点算 1，半角字母数字算 0.5（粗略）。"""
    n=0.0
    for ch in s:
        if ch==" ": n+=0.5
        elif is_cjk_char(ch): n+=1
        else: n+=0.5
    return n

def line_text(line):
    """spec 行 → 纯文本（支持 runs 富文本）"""
    if isinstance(line, str): return line
    return "".join(r["t"] for r in line["runs"])
def line_runs(line):
    if isinstance(line, str): return [{"t": line}]
    return line["runs"]
