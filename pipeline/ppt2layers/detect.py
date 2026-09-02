# -*- coding: utf-8 -*-
"""开放词汇检测（YOLO-World）：按对象名在全图定位真框——"Canva 架构"的几何来源。

VLM 起草的框会脱靶（灯笼标进星空）；检测器按语义词在全图找目标，
与 VLM 框做匹配：有交集取检测框（纠偏），完全脱靶取最高置信检测框并告警。
权重 models/yolov8s-worldv2.pt（约 26MB，首次自动下载；文本编码依赖 ultralytics 的 CLIP fork）。
"""
import os
import re

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_model = None
_classes = None

# 课件常见对象：中文名关键词 → 英文检测词（可多个候选词）
CN2EN = [
    ("人物", ["person"]), ("女孩", ["girl", "person"]), ("男孩", ["boy", "person"]),
    ("老人", ["elderly person", "person"]), ("女性", ["woman", "person"]), ("男性", ["man", "person"]),
    ("灯笼", ["lantern", "chinese lantern"]), ("灯", ["lamp", "lantern"]), ("油灯", ["oil lamp", "lamp"]),
    ("桌", ["table", "desk"]), ("椅", ["chair"]), ("凳", ["stool", "chair"]),
    ("茶具", ["teapot", "tea set"]), ("茶壶", ["teapot"]), ("碗", ["bowl"]), ("杯", ["cup"]),
    ("书", ["book"]), ("船", ["boat"]), ("被子", ["blanket", "quilt"]), ("枕", ["pillow"]),
    ("窗", ["window"]), ("门", ["door"]), ("花", ["flower"]), ("树", ["tree"]), ("鸟", ["bird"]),
    ("猫", ["cat"]), ("狗", ["dog"]), ("笔", ["pen", "brush"]), ("纸", ["paper"]), ("扇", ["hand fan"]),
]


def en_terms(cn_name, hint_en=None):
    """对象中文名 → 英文检测词列表。"""
    if hint_en:
        return [t.strip() for t in re.split(r"[,/;]", hint_en) if t.strip()]
    out = []
    for key, terms in CN2EN:
        if key in cn_name:
            out += [t for t in terms if t not in out]
    return out


def _yw(classes):
    global _model, _classes
    from ultralytics import YOLOWorld
    if _model is None:
        _model = YOLOWorld(os.path.join(_ROOT, "models", "yolov8s-worldv2.pt"))
    if _classes != tuple(classes):
        _model.set_classes(list(classes))
        _classes = tuple(classes)
    return _model


def detect(img_rgb, terms, conf=0.05):
    """按英文词表在全图检测。返回 [(box[x0,y0,x1,y1], conf, term), ...] 按置信度降序。"""
    if not terms:
        return []
    try:
        r = _yw(terms)(img_rgb, device="cpu", conf=conf, verbose=False)[0]
        out = []
        for b in r.boxes:
            out.append(([float(v) for v in b.xyxy[0].tolist()], float(b.conf[0]), terms[int(b.cls[0])]))
        out.sort(key=lambda x: -x[1])
        return out
    except Exception as e:
        print(f"YOLO-World 失败: {str(e)[:120]}", flush=True)
        return []


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(1.0, ua)


EN2CN = {"lantern": "灯笼", "lamp": "灯", "candle": "烛灯", "teapot": "茶壶", "bowl": "碗", "cup": "杯",
         "vase": "花瓶", "book": "书", "basket": "篮子", "clock": "钟", "hand fan": "扇子", "umbrella": "伞",
         "kite": "风筝", "boat": "船", "birdcage": "鸟笼", "desk lamp": "台灯", "chair": "椅子",
         "stool": "凳子", "plate": "盘子", "teacup": "茶杯", "brush pen": "毛笔", "scroll": "卷轴"}
# 注意：词表只收"散件物品"（离散可移动）；天空/树木/屋檐/墙体等"场景织物"永不入表——
# 提出来层边缘发虚、背景多一块凭空补丁，两头受损。


def _contain(b, cb):
    """b 被 cb 覆盖的面积占 b 自身的比例。"""
    ix = max(0.0, min(b[2], cb[2]) - max(b[0], cb[0]))
    iy = max(0.0, min(b[3], cb[3]) - max(b[1], cb[1]))
    return ix * iy / max(1.0, (b[2] - b[0]) * (b[3] - b[1]))


def find_unclaimed_objects(img_rgb, claimed_boxes, max_n=6, conf=0.15, contain_boxes=None):
    """自动补检：通用物件词表全图检测，返回未被任何已有标注认领的实例。
    治"VLM 起草漏标物品"——有什么东西不该由语言模型说了算。
    claimed_boxes：IoU>0.15 即排除（对底板等大框安全——板上物件仍可抠）。
    contain_boxes：已抠成层的人物/物品/书法的**成品框**——检测框大半落入（>0.6）也排除；
    小件对大并集框的 IoU 天然小，只靠 IoU 挡不住重复抠层（茶具×2 之祸）。"""
    H, W = img_rgb.shape[:2]
    page = float(W * H)
    dets = detect(img_rgb, list(EN2CN.keys()), conf=conf)
    out = []
    for b, c, t in dets:
        area = (b[2] - b[0]) * (b[3] - b[1])
        if area < 0.002 * page or area > 0.25 * page:
            continue
        # 面积重叠排除（防与已标对象重复）；不能按"中心落入"排除——桌上物件中心必然在桌框内
        if any(_iou(b, cb) > 0.15 for cb in claimed_boxes):
            continue
        if contain_boxes and any(_contain(b, cb) > 0.6 for cb in contain_boxes):
            continue
        if any(_iou(b, o[0]) > 0.5 for o in out):
            continue
        out.append(([int(v) for v in b], c, t))
    # 邻近同族聚类：壶身/壶嘴/壶盖会被拆成多个 cup/teapot 检测 → 外扩相交即并为一件
    TEA = {"cup", "bowl", "teapot", "tea set", "plate"}
    def _ex(b, f=0.18):
        w, h = b[2] - b[0], b[3] - b[1]
        return [b[0] - w * f, b[1] - h * f, b[2] + w * f, b[3] + h * f]
    groups = [{"box": list(b), "conf": c, "terms": [t]} for b, c, t in out]
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, g = _ex(groups[i]["box"]), _ex(groups[j]["box"])
                if min(a[2], g[2]) > max(a[0], g[0]) and min(a[3], g[3]) > max(a[1], g[1]):
                    bi, bj = groups[i]["box"], groups[j]["box"]
                    groups[i]["box"] = [min(bi[0], bj[0]), min(bi[1], bj[1]), max(bi[2], bj[2]), max(bi[3], bj[3])]
                    groups[i]["conf"] = max(groups[i]["conf"], groups[j]["conf"])
                    groups[i]["terms"] += groups[j]["terms"]
                    del groups[j]; changed = True; break
            if changed: break
    res = []
    for g in sorted(groups, key=lambda x: -x["conf"])[:max_n]:
        ts = g["terms"]
        if len(ts) > 1 and set(ts) <= TEA:
            name = "茶具"                      # 命中组合词表 → 抠取时按成员并集
        elif len(ts) > 1:
            name = EN2CN.get(ts[0], ts[0]) + "组"
        else:
            name = EN2CN.get(ts[0], ts[0])
        res.append(([int(v) for v in g["box"]], g["conf"], name))
    return res


COMPOSITE_CN = {
    "茶具": ["teapot", "tea set", "teacup", "bowl", "cup"],
    "餐具": ["bowl", "plate", "cup", "chopsticks", "spoon"],
    "文具": ["pen", "pencil", "writing brush", "ink stone", "book"],
    "水果": ["fruit", "apple", "orange", "banana", "peach"],
    "花草": ["flower", "potted plant", "vase"],
    "书堆": ["book", "stack of books"],
}


def composite_boxes(img_rgb, cn_name, vlm_box, hint_en=None):
    """组合物件（茶具/餐具…散布成员）：检出原框邻域内的全部成员实例框。
    返回 (boxes, note)；非组合名或没检到成员 → ([], None)。"""
    terms = None
    for key, ts in COMPOSITE_CN.items():
        if key in cn_name:
            terms = ts
            break
    if hint_en and re.search(r"[,/;]", hint_en):          # 多词提示也按组合处理
        terms = en_terms(cn_name, hint_en)
    if not terms:
        return [], None
    dets = detect(img_rgb, terms, conf=0.08)
    if not dets:
        return [], None
    x0, y0, x1, y1 = vlm_box
    ex, ey = (x1 - x0) * 0.9, (y1 - y0) * 0.9
    inside = [d for d in dets
              if x0 - ex <= (d[0][0] + d[0][2]) / 2 <= x1 + ex and y0 - ey <= (d[0][1] + d[0][3]) / 2 <= y1 + ey]
    if not inside:
        return [], None
    boxes = [[int(v) for v in d[0]] for d in inside]
    kinds = "、".join(sorted({d[2] for d in inside}))
    return boxes, f"组合物件：检出 {len(boxes)} 个成员（{kinds}）"


def assign_person_boxes(img_rgb, persons):
    """spec 人物列表 → {id: (box, note)}。全图检测 person 后与 spec 框做一对一贪心匹配
    （IoU 最高的配对优先、每个检测框只用一次）——防止相邻/重叠人物被错认到同一个框。"""
    dets = detect(img_rgb, ["person"], conf=0.15)
    boxes = [d[0] for d in dets]
    result = {}
    cx = lambda b: (b[0] + b[2]) / 2
    if boxes and len(boxes) == len(persons):
        # 人数与检测数相等 → 两边按中心横坐标同序配对（VLM 常整体错位，IoU 贪心会"集体左移一位"）
        sp = sorted(persons, key=lambda h: cx(h["bbox"]))
        db = sorted(boxes, key=cx)
        for h, b in zip(sp, db):
            iou = _iou(b, h["bbox"])
            note = None if iou >= 0.6 else f"框已按人物检测同序校正（IoU {iou:.2f}）"
            result[h["id"]] = ([int(v) for v in b], note)
        return result
    pairs = sorted(((_iou(db_, h["bbox"]), i, j) for i, h in enumerate(persons) for j, db_ in enumerate(boxes)),
                   reverse=True)
    used_i, used_j = set(), set()
    for iou, i, j in pairs:
        if iou < 0.05 or i in used_i or j in used_j:
            continue
        used_i.add(i); used_j.add(j)
        note = None if iou >= 0.6 else f"框已按人物检测校正（IoU {iou:.2f}）"
        result[persons[i]["id"]] = ([int(v) for v in boxes[j]], note)
    # 剩余：未匹配的 spec 人物 × 未用的检测框，按中心距离续配（处理"集体错位"下的漏配）
    left_i = [i for i in range(len(persons)) if i not in used_i]
    left_j = [j for j in range(len(boxes)) if j not in used_j]
    for i in left_i:
        if not left_j: break
        h = persons[i]
        j = min(left_j, key=lambda jj: abs(cx(boxes[jj]) - cx(h["bbox"])))
        left_j.remove(j)
        result[h["id"]] = ([int(v) for v in boxes[j]], "框已按检测补配（原框严重脱靶）")
    for h in persons:
        result.setdefault(h["id"], (h["bbox"], None))
    return result


def refine_box(img_rgb, cn_name, vlm_box, hint_en=None):
    """用检测结果校正 VLM 框。返回 (box, note)；note 为 None 表示未纠偏。"""
    terms = en_terms(cn_name, hint_en)
    dets = detect(img_rgb, terms)
    if not dets:
        return vlm_box, None
    best = max(dets, key=lambda d: _iou(d[0], vlm_box) + d[1] * 0.01)
    iou = _iou(best[0], vlm_box)
    if iou >= 0.45:
        return vlm_box, None                       # 框基本对，尊重 VLM/人工
    box = [int(v) for v in best[0]]
    if iou > 0.03:
        return box, f"框已按检测微调（{best[2]} {best[1]:.2f}）"
    return box, f"框脱靶，已按检测重定位（{best[2]} {best[1]:.2f}）"
def score_object_alpha(alpha, box, term=None, emb=None):
    """SAM 多假设候选评分骨架（暂未接线）：对一份候选 alpha 给几何分。
    alpha：整页 HxW float(0~1)；box：[x0,y0,x1,y1] 像素框（检测/精修框）。
    返回 {"fill","compact","touch_edge"}，均 0~1、越高越好：
      fill       框内覆盖率——0.35~0.95 区间满分（过空=没抠到主体，过满=连背景一起抓）；
      compact    最大连通域质量占比——碎裂掩膜说明抓了散落误检；
      touch_edge 框缘接触惩罚——掩膜大面积贴满框缘即背景泄漏（≤20% 接触不罚，≥80% 归零）。
    term/emb 为语义分预留（CLIP 文本词 / 图像嵌入）；接入后在返回 dict 追加 "sem" 键，
    本函数的几何键名与取值域届时不得变动（打分器的消费方按键读取）。"""
    import numpy as np
    import cv2
    zero = {"fill": 0.0, "compact": 0.0, "touch_edge": 0.0}
    if alpha is None:
        return zero
    H, W = alpha.shape[:2]
    x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
    x1, y1 = min(W, int(box[2])), min(H, int(box[3]))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return zero
    m = (np.asarray(alpha[y0:y1, x0:x1], dtype=np.float32) > 0.5).astype(np.uint8)
    fg = int(m.sum())
    if fg == 0:
        return zero
    fill_raw = fg / float(m.size)
    if fill_raw < 0.35:
        fill = fill_raw / 0.35
    elif fill_raw > 0.95:
        fill = max(0.0, (1.0 - fill_raw) / 0.05)
    else:
        fill = 1.0
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(m, connectivity=8)
    compact = float(stats[1:, cv2.CC_STAT_AREA].max()) / fg if n > 1 else 0.0
    border = int(m[0, :].sum() + m[-1, :].sum() + m[:, 0].sum() + m[:, -1].sum())
    frac = border / float(2 * (m.shape[0] + m.shape[1]))
    touch_edge = float(np.clip(1.0 - (frac - 0.2) / 0.6, 0.0, 1.0))
    return {"fill": round(float(fill), 4), "compact": round(compact, 4), "touch_edge": round(touch_edge, 4)}
