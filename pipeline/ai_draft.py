# -*- coding: utf-8 -*-
"""AI 标注起草：视觉大模型按标注规范起草整页标注（spec 草稿）。

- draft_page(deck, n, cfg) -> (spec, warnings)   供 app/server.py 单页调用
- CLI: ai_draft.py <deck> <pages...>             批量起草（跳过已有 spec 的页），供任务队列调用
配置读 <ROOT>/slidelift.json 的 ai 段：{"enabled", "model", "key", "base_url"?}。
提供 Mock 通道（model="Mock"）：不联网，用 OCR 行几何分组出草稿，供开发自测走通全链路。
"""
import os, sys, json, re, base64, time, urllib.request, urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W0, H0 = 1672, 941
FONTS = ["楷体", "微软雅黑", "黑体", "华文行楷", "宋体", "Arial"]
PANEL_KINDS = ["card", "capsule", "panel", "cell", "box", "badge"]

PRESETS = {
    "Qwen-VL-Max": dict(kind="openai", base="https://dashscope.aliyuncs.com/compatible-mode/v1", model="qwen-vl-max"),
    "豆包-Vision":  dict(kind="openai", base="https://ark.cn-beijing.volces.com/api/v3", model="doubao-seed-1-6-250615"),
    "GLM-4.5V":    dict(kind="openai", base="https://open.bigmodel.cn/api/paas/v4", model="glm-4.5v"),
    "Claude":      dict(kind="anthropic", model="claude-opus-5"),
    "GPT-4o":      dict(kind="openai", base="https://api.openai.com/v1", model="gpt-4o"),
    "Kimi-VL":     dict(kind="openai", base="https://api.moonshot.cn/v1", model="kimi-latest"),
    "自定义(OpenAI兼容)": dict(kind="openai", base=None, model=None),
    "Mock":        dict(kind="mock"),
}

SYSTEM = """你是课件页面标注员，负责把"整屏位图"的 PPT 页面拆解成结构化标注（供分层还原引擎使用）。
画布坐标：左上角为原点，宽 1672 高 941；bbox=[x0,y0,x1,y1] 整数。

【图层分类（按"用户会不会想单独动它"来分）】
- text_zones：常规印刷字体排版的一切文字（含页码、表格单元格文字、胶囊标签文字、选项字母）→ 将重建为可编辑文本框。**每个 zone 必须填 style 分级**：
  · style="plain"：普通印刷体，转成可编辑文字后能以假乱真；
  · style="art"：特效美术字——立体挤出/描边勾边/渐变彩色/金属光泽/发光爆炸效果/艺术变形/超粗装饰大标题（海报主标题几乎都是）→ 引擎会整块抠成图层保真，不转文字。一行里多种颜色的标语（如彩色胶囊词组）也算 art。拿不准时选 art（保真优先）
- calligraphy：毛笔书法/手写字（笔画边缘毛糙有飞白、同字内粗细变化大、像真毛笔写的；章节大标题常见）→ 抠成透明图层不转文字。拿不准时选 calligraphy（保真优先）。**若 OCR 候选里识别出了这段书法文字，给它填 ocr_ids**（引擎会按行框收紧位置）
- persons：人物（整个人，含头发、手、衣摆），name 用"人物_xx"
- objects：大件可移动物品（桌子及桌上书本/油灯合为一层"物品_书桌"、独立的灯/船/椅），name 用"物品_xx"；method 从 saliency(默认)/table(平直大件如桌子)/grabcut(被子船等) 选；散布的组合物件（茶具=壶+碗、文具一套）标**一个**物品即可，引擎会自动检出全部成员
- panels：字底板——文字下方的卡片/圆角框/胶囊标签/半透明白板/表格单元格色块/答案框/笔刷徽章，bbox 要包住整个底板含边框；kind: card|capsule|panel(大面积半透明白板)|cell|box(答案框)|badge(笔刷徽章)；fill: opaque_light|opaque_dark|translucent_light|translucent_dark（透过底板能看到照片细节=translucent）
- decor_keep_in_bg：装饰件（星星、分隔线、虚线、下划线、水彩晕染、批注圈）→ 字符串数组，留在背景不拆
- 鲜艳彩色/渐变的装饰横条、色带（如底部黄绿渐变条）**不要标 panel**——写进 decor_keep_in_bg 留在背景（panel 只给灰白/单色的文字衬板；彩条标板会产出空图层）
- **边界清晰的独立小图标/贴纸**（对话气泡、徽章、小 logo、emoji 风格贴片——压在标题或人物上、有完整轮廓的）→ 标 objects（method:grabcut），框要紧贴图标本身；漏标会被邻近大元素的图层吸走一半，拖动时撕裂
- 文字下面的衬板**不论多大**都要标 panel：大面积柔光/渐变白雾板 kind="panel"、fill="translucent_light"（暗色雾板用 translucent_dark），引擎会解混成真正的半透明层；只有整页纯色平涂底才归背景

【text_zones 划分硬规则】
- 一个 zone = 同字体、同颜色、同粗细、同对齐、同容器的连续文字，可多行；各行必须左边缘对齐或中心对齐；缩进不同的行拆成不同 zone，或段首缩进用两个全角空格"　　"开头表示
- 行距突变处拆开；表格每个单元格一个 zone；胶囊标签文字单独 zone；页码单独 zone（container="bg", font="Arial"），并同时填 page_number
- lines 按画面视觉分行逐行写，保留全角标点；行内多个空格按原样保留
- 行内前缀不同色加粗（如「点拨：」「示例1：」）时该行写成 {"runs":[{"t":"点拨：","bold":true},{"t":"其余文字"}]}（run 颜色留空由引擎实测）
- **OCR 候选只是候选**：文字内容、标点、分行必须对着图片逐字核对纠错；OCR 漏掉的（页码、小标签、表头）要补上
- ocr_ids：该 zone 各行对应的 OCR 候选行编号，**必须与 lines 逐行一一对应且文本一致**（引擎会按 OCR 行框收紧 bbox）；对不上就留空数组并自己给准 bbox，禁止乱填
- **已标为 calligraphy 的文字绝不能再建 text_zone**（同一标题标两次是最常见错误）
- persons/objects 的 bbox 宁大勿小，必须完整包住对象（含被桌子遮住前的身体、灯穗、壶把）
- bbox 每边比文字墨迹松 4~8px，但绝不能越出所在底板/卡片边框
- font 只能从 楷体|微软雅黑|黑体|华文行楷|宋体|Arial 中选：楷体=有起收笔像手写的印刷体；微软雅黑/黑体=无衬线等粗（粗体写 bold=true）；宋体=横细竖粗有衬线；华文行楷=连笔行书印刷体；Arial=纯数字英文
- polarity：light=深底浅字，dark=浅底深字；color 目测粗略即可
- container："bg" 或 "panel:<该底板id>"；effects：字缘有柔光晕=glow:true，有偏移暗影=shadow:true
- line_mode：normal|single(单个大字或字母)|tight(行距很小)|pinyin(带声调拼音)

【输出】只输出一个 JSON 对象，不要 markdown 代码围栏，不要任何解释文字。
text_zones/calligraphy/persons/objects/panels/decor_keep_in_bg 必须是 JSON 数组 [...]，
数组元素直接写对象 {...}，严禁写成 "t1": {...} 这种键值形式。结构：
{"layout_notes":"一句话版式描述",
 "text_zones":[{"id":"t1","lines":["…"],"bbox":[x0,y0,x1,y1],"ocr_ids":[0,1],"style":"plain","font":"微软雅黑","bold":false,"color":"#FFFFFF","polarity":"light","align":"left","container":"bg","effects":{"glow":false,"shadow":false},"line_mode":"normal","notes":""}],
 "calligraphy":[{"id":"k1","text":"课前导入","bbox":[…],"ocr_ids":[0],"color":"#FFFFFF","polarity":"light","notes":""}],
 "persons":[{"id":"h1","name":"人物_女孩","bbox":[…],"notes":""}],
 "objects":[{"id":"o1","name":"物品_书桌","bbox":[…],"method":"table","notes":""}],
 "panels":[{"id":"c1","name":"底板_卡片1","kind":"card","bbox":[…],"shape":"rounded_rect","fill":"translucent_dark","border":true,"notes":""}],
 "decor_keep_in_bg":["标题下方紫色装饰线与星星"],
 "page_number":{"text":"4","bbox":[…],"color":"#FFFFFF","font":"Arial"}}
自检：画面上每一个字都被某个 text_zone 或 calligraphy 覆盖；zone 不压底板边框；书法判定拿不准写进 notes。"""


def _user_text(ocr):
    lines = [f"[{i}] conf={o.get('conf', 0):.2f} box={[round(v) for v in o['box']]} 文本：{o['text']}"
             for i, o in enumerate(ocr or [])]
    return ("下面是本页 OCR 候选行（编号从 0 开始，box=[x0,y0,x1,y1]）：\n"
            + ("\n".join(lines) if lines else "（本页 OCR 没有识别到文字）")
            + "\n\n请对照图片完成整页标注，按规范输出 JSON。")


# ---------------------------------------------------------------- 各家适配
def _call_openai(cfg, img_b64, user_text):
    base = cfg.get("base_url") or cfg["_preset"]["base"]
    if not base:
        raise RuntimeError("该预设需要在设置里填写「Base URL 覆盖」（OpenAI 兼容服务地址）")
    if not (cfg.get("model_id") or cfg["_preset"]["model"]):
        raise RuntimeError("该预设需要在设置里填写「模型 ID 覆盖」")
    base = base.rstrip("/")
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps({
            "model": cfg.get("model_id") or cfg["_preset"]["model"],
            "temperature": 0,
            "max_tokens": 6000,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img_b64}},
                    {"type": "text", "text": user_text},
                ]},
            ],
        }).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + cfg["key"]},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:400]
        raise RuntimeError(f"模型服务返回 {e.code}：{body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"无法连接模型服务：{e.reason}")
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError("模型响应格式异常：" + json.dumps(data)[:300])


def _call_anthropic(cfg, img_b64, user_text):
    import anthropic
    client = anthropic.Anthropic(api_key=cfg["key"])
    try:
        resp = client.messages.create(
            model=cfg.get("model_id") or cfg["_preset"]["model"],
            max_tokens=8000,
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                {"type": "text", "text": user_text},
            ]}],
            # Claude Opus 5：按官方建议默认启用服务端安全回退（拒答时同一请求内换模型续跑）
            extra_headers={"anthropic-beta": "server-side-fallback-2026-07-01"},
            extra_body={"fallbacks": "default"},
        )
    except anthropic.AuthenticationError:
        raise RuntimeError("API Key 无效（AuthenticationError）")
    except anthropic.RateLimitError:
        raise RuntimeError("触发限流，请稍后重试（RateLimitError）")
    except anthropic.APIStatusError as e:
        raise RuntimeError(f"Anthropic API 错误 {e.status_code}：{getattr(e, 'message', '')[:300]}")
    except anthropic.APIConnectionError:
        raise RuntimeError("无法连接 Anthropic API（网络错误）")
    if resp.stop_reason == "refusal":
        raise RuntimeError("模型拒绝了该请求（safety refusal）")
    return "".join(b.text for b in resp.content if b.type == "text")


def _call_mock(ocr):
    """离线自测：按 OCR 行左缘/间距分组成 zone，产出与真模型同构的草稿。"""
    zones, cur = [], None
    for i, o in enumerate(ocr or []):
        b = o["box"]
        if cur and abs(b[0] - cur["_x0"]) < 14 and 0 <= b[1] - cur["_y1"] < 46:
            cur["lines"].append(o["text"]); cur["ocr_ids"].append(i); cur["_y1"] = b[3]
        else:
            cur = {"lines": [o["text"]], "ocr_ids": [i], "_x0": b[0], "_y1": b[3]}
            zones.append(cur)
    for z in zones:
        z.pop("_x0", None); z.pop("_y1", None)
        z.update({"font": "微软雅黑", "bold": False, "color": "#FFFFFF", "polarity": "light",
                  "align": "left", "container": "bg", "notes": "Mock 草稿（离线几何分组）"})
    return json.dumps({"layout_notes": "（Mock 起草，仅供联调）", "text_zones": zones,
                       "calligraphy": [], "persons": [], "objects": [], "panels": [],
                       "decor_keep_in_bg": []}, ensure_ascii=False)


# ---------------------------------------------------------------- 解析与归一
def _close_brackets(s):
    """截断修补：数未闭合的 {} / []（跳过字符串字面量），把缺的收尾补上。"""
    stack, ins, esc = [], False, False
    for ch in s:
        if ins:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': ins = False
        else:
            if ch == '"': ins = True
            elif ch in "{[": stack.append("}" if ch == "{" else "]")
            elif ch in "}]" and stack: stack.pop()
    s2 = (s + '"') if ins else s
    return re.sub(r",\s*$", "", s2.rstrip()) + "".join(reversed(stack))


def _fix_keyed_arrays(s):
    """结构漂移修补：模型把数组写成字典条目（["t1": {...}, "t2": {...}]）。
    合法 JSON 的数组里字符串后不可能跟冒号——遇到即把 "键": 剥掉，只留值。"""
    out, stack = [], []
    ins = esc = False; sstart = -1
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ins:
            out.append(ch)
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"':
                ins = False
                if stack and stack[-1] == "[":
                    j = i + 1
                    while j < n and s[j] in " \t\r\n": j += 1
                    if j < n and s[j] == ":":
                        del out[sstart:]   # 丢掉整个 "键"
                        i = j              # 连冒号一起跳过
            i += 1; continue
        if ch == '"':
            ins = True; sstart = len(out); out.append(ch)
        elif ch in "{[":
            stack.append(ch); out.append(ch)
        elif ch in "}]":
            if stack: stack.pop()
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def _escape_inner_quotes(s):
    """字符串值里未转义的英文引号：闭引号后面必须是 , : } ] 或结尾，否则视为内容引号转义之。"""
    out = []; ins = esc = False
    n = len(s)
    for i, ch in enumerate(s):
        if not ins:
            if ch == '"': ins = True
            out.append(ch); continue
        if esc: out.append(ch); esc = False; continue
        if ch == "\\": out.append(ch); esc = True; continue
        if ch == '"':
            j = i + 1
            while j < n and s[j] in " \t\r\n": j += 1
            if j >= n or s[j] in ",:}]":
                ins = False; out.append(ch)
            else:
                out.append('\\"')
        else:
            out.append(ch)
    return "".join(out)


def _extract_json(text):
    """容错梯子：模型 JSON 常见毛病逐级修补——裸换行/控制符（strict=False）、尾逗号、
    数组被写成字典条目、字符串内裸引号、输出截断补括号。"""
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        if t.find("{") >= 0:
            t = t[t.find("{"):]; a, b = 0, len(t) - 1   # 截断到连 } 都没有：全量进修补
        else:
            raise RuntimeError("模型输出中没有 JSON 对象：" + text[:200])
    s = t[a:b + 1]
    s1 = re.sub(r",\s*([}\]])", r"\1", s)
    fk = _fix_keyed_arrays(s1)
    err = None
    for cand in (s, s1, fk, _close_brackets(fk), _escape_inner_quotes(fk),
                 _close_brackets(_escape_inner_quotes(fk))):
        for strict in (True, False):        # strict=False 容忍字符串内裸换行/制表符
            try:
                return json.loads(cand, strict=strict)
            except Exception as e:
                err = e
    raise RuntimeError(f"JSON 解析失败：{err}")


def _clampbox(bb):
    x0, y0, x1, y1 = [float(v) for v in bb]
    x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1))
    return [max(0, min(W0 - 2, round(x0))), max(0, min(H0 - 2, round(y0))),
            min(W0, max(2, round(x1))), min(H0, max(2, round(y1)))]


def _normalize(spec, ocr, warnings):
    for k in ("text_zones", "calligraphy", "persons", "objects", "panels", "decor_keep_in_bg"):
        if isinstance(spec.get(k), dict):      # 模型把集合写成 {"t1": {...}} 字典 → 取值成列表
            spec[k] = list(spec[k].values())
    out = {"layout_notes": str(spec.get("layout_notes") or ""), "text_zones": [], "calligraphy": [],
           "persons": [], "objects": [], "panels": [], "decor_keep_in_bg": []}
    # panels 先归一（zone 的 container 要引用它们的新 id）
    idmap = {}
    for i, p in enumerate([p for p in spec.get("panels") or [] if isinstance(p, dict) and p.get("bbox")], 1):
        kind = {"answer": "box", "translucent": "panel"}.get(p.get("kind"), p.get("kind"))
        if kind not in PANEL_KINDS: kind = "card"; warnings.append(f"底板类型未知，按卡片处理：{p.get('kind')}")
        fill = p.get("fill") or "opaque_light"
        if fill == "translucent": fill = "translucent_light"
        if fill not in ("opaque_light", "opaque_dark", "translucent_light", "translucent_dark"): fill = "opaque_light"
        nid = f"c{i}"
        if p.get("id"): idmap[str(p["id"])] = nid
        out["panels"].append({"id": nid, "name": p.get("name") or f"底板_{i}", "kind": kind,
                              "bbox": _clampbox(p["bbox"]), "shape": p.get("shape") or "rounded_rect",
                              "fill": fill, "border": bool(p.get("border")), "notes": str(p.get("notes") or "")})
    def _sim(a, b):
        """粗相似：去空白后一方为另一方前缀（前6字），容错标点差异。"""
        a = re.sub(r"\s", "", str(a))[:6]; b = re.sub(r"\s", "", str(b))[:6]
        return bool(a) and bool(b) and (a.startswith(b[:3]) or b.startswith(a[:3]))
    calli_texts = {re.sub(r"\s", "", k.get("text") or "") for k in spec.get("calligraphy") or [] if isinstance(k, dict)}
    used_ocr = set(); art_pending = []   # style=art 的文字块改道进 calligraphy 通道（method=art）
    for i, z in enumerate([z for z in spec.get("text_zones") or [] if isinstance(z, dict)], 1):
        lines = [ln for ln in (z.get("lines") or []) if (isinstance(ln, str) and ln.strip()) or (isinstance(ln, dict) and ln.get("runs"))]
        if not lines: continue
        ztxt = re.sub(r"\s", "", "".join(l if isinstance(l, str) else "".join(r.get("t", "") for r in l.get("runs") or []) for l in lines))
        if ztxt and ztxt in calli_texts:   # 与书法重复标注 → 丢弃文字块（书法优先，保真）
            warnings.append(f"文字块「{ztxt[:8]}」与书法层重复，已丢弃（书法只抠图不转字）"); continue
        ids = []
        for x in (z.get("ocr_ids") or []):
            try: xi = int(x)
            except (TypeError, ValueError): continue
            if 0 <= xi < len(ocr or []): ids.append(xi)
        # 锚定前校验：ids 与 lines 一一对应时，逐行核对文本相似，防错行锚定把框拉歪
        if ids and len(ids) == len(lines):
            bad = [j for j, (li, oi) in enumerate(zip(lines, ids))
                   if not _sim(li if isinstance(li, str) else "".join(r.get("t", "") for r in li.get("runs") or []), ocr[oi]["text"])]
            if bad:
                warnings.append(f"文字块 t{i} 有 {len(bad)} 行与所引 OCR 文本不符，已忽略其锚定")
                ids = []
        elif ids and len(ids) != len(lines):
            if not any(_sim("".join(l if isinstance(l, str) else "" for l in lines), ocr[j]["text"]) for j in ids):
                ids = []
        # 锚定失败兜底：按文本检索重锚——AI 抄错字/漏填 ocr_ids 会让上面的校验弃锚，弃锚=用 AI 的
        # 偏框（框偏→标定在空框里测不到墨→退默认小字号，"试图写写"渲染到页面左侧案）。
        # 每行在全 OCR 里找相似文本行（同句多处时取 y 最接近 AI 框的），全部行都命中才采纳，防错锚拉框
        if not ids and lines:
            _zb = z.get("bbox") or [0, 0, 1672, 941]
            _cand = []
            for _li in lines:
                _lt = _li if isinstance(_li, str) else "".join(r.get("t", "") for r in (_li.get("runs") or []))
                _ms = [_oi for _oi, _o in enumerate(ocr or []) if _oi not in used_ocr and _oi not in _cand and _sim(_lt, _o["text"])]
                if not _ms:
                    _cand = []; break
                _cand.append(min(_ms, key=lambda _oi: abs(ocr[_oi]["box"][1] - _zb[1])))
            if _cand:
                ids = _cand
                warnings.append(f"文字块 t{i} 锚定失效，已按文本检索重锚 {len(ids)} 行")
        # 转写纠错：行与 OCR 行去空格后等长、只差 1-2 个字 → 信 OCR（AI 抄写爱把"试着"手滑成"试图"
        # 这类高频搭配；OCR 是视觉识别，形近字反而不易错）。带格式 runs 的行不动。
        if ids and len(ids) == len(lines):
            for _j, (_li, _oi) in enumerate(zip(lines, ids)):
                if isinstance(_li, str):
                    _lt = re.sub(r"\s", "", _li); _ot = re.sub(r"\s", "", ocr[_oi]["text"] or "")
                    if _ot and _lt and len(_ot) == len(_lt) and _ot != _lt:
                        _df = sum(1 for _a, _b in zip(_lt, _ot) if _a != _b)
                        if 0 < _df <= 2:
                            lines[_j] = ocr[_oi]["text"].strip()
                            warnings.append(f"文字块 t{i} 第{_j+1}行按 OCR 纠字 {_df} 处")
        used_ocr.update(ids)
        if ids:  # OCR 锚定：bbox 收紧为对应行框并集 + 6px
            xs0 = [ocr[j]["box"][0] for j in ids]; ys0 = [ocr[j]["box"][1] for j in ids]
            xs1 = [ocr[j]["box"][2] for j in ids]; ys1 = [ocr[j]["box"][3] for j in ids]
            bb = _clampbox([min(xs0) - 6, min(ys0) - 6, max(xs1) + 6, max(ys1) + 6])
        elif z.get("bbox"):
            bb = _clampbox(z["bbox"])
        else:
            warnings.append(f"文字块「{str(lines[0])[:8]}…」没有 bbox 也没有 ocr_ids，已丢弃"); continue
        if str(z.get("style") or "").lower() == "art":   # 特效美术字：改道图层通道（转不像的不转，整块抠图保真）
            _pol = z.get("polarity") if z.get("polarity") in ("light", "dark") else "light"
            art_pending.append({"text": ztxt[:20] or "艺术字", "bbox": bb, "ocr_ids": [],
                                "color": z.get("color") or "#FFFFFF", "polarity": _pol, "method": "art",
                                "notes": "AI 分级 art：" + str(z.get("notes") or "特效字整块成层")})
            continue
        font = z.get("font") if z.get("font") in FONTS else None
        if not font: warnings.append(f"字体「{z.get('font')}」不在可用清单，改用微软雅黑"); font = "微软雅黑"
        cont = z.get("container") or "bg"
        if ":" in str(cont):
            ref = str(cont).split(":", 1)[1]
            nid = idmap.get(ref) or (ref if any(p["id"] == ref for p in out["panels"]) else None)
            cont = f"panel:{nid}" if nid else "bg"
            if cont == "bg": warnings.append(f"文字块 t{i} 的底板引用无效，改为背景")
        pol = z.get("polarity") if z.get("polarity") in ("light", "dark") else "light"
        eff = z.get("effects") or {}
        out["text_zones"].append({
            "id": f"t{i}", "lines": lines, "bbox": bb, "ocr_ids": ids,
            "font": font, "bold": bool(z.get("bold")),
            "color": z.get("color") if re.match(r"^#[0-9A-Fa-f]{6}$", str(z.get("color") or "")) else ("#FFFFFF" if pol == "light" else "#333333"),
            "polarity": pol,
            "align": z.get("align") if z.get("align") in ("left", "center", "right") else "left",
            "container": cont,
            "effects": {"glow": bool(eff.get("glow")), "shadow": bool(eff.get("shadow"))},
            "line_mode": z.get("line_mode") if z.get("line_mode") in ("normal", "single", "tight", "pinyin") else "normal",
            "notes": ("AI 起草 · " + str(z.get("notes") or "")).rstrip(" ·")})
    _cal_src = [k for k in spec.get("calligraphy") or [] if isinstance(k, dict) and k.get("bbox")] + art_pending
    for i, k in enumerate(_cal_src, 1):
        bb = _clampbox(k["bbox"])
        ktext = str(k.get("text") or "书法字")[:20]
        ids = [int(x) for x in (k.get("ocr_ids") or []) if isinstance(x, (int, float)) and 0 <= int(x) < len(ocr or [])]
        if not ids and ocr:   # 自动锚定：中心落在书法框(外扩40%)内、与书法文本有交集字符的未引用高置信 OCR 行
            ex, ey = (bb[2] - bb[0]) * 0.4, (bb[3] - bb[1]) * 0.4
            for j, o_ in enumerate(ocr):
                if j in used_ocr or o_.get("conf", 0) < 0.85: continue
                ox0, oy0, ox1, oy1 = o_["box"]
                ix = max(0.0, min(bb[2] + ex, ox1) - max(bb[0] - ex, ox0))
                iy = max(0.0, min(bb[3] + ey, oy1) - max(bb[1] - ey, oy0))
                if ix * iy >= 0.3 * max(1.0, (ox1 - ox0) * (oy1 - oy0)) \
                        and (not ktext or set(ktext) & set(o_.get("text", ""))):
                    ids.append(j)
        if ids:   # 书法笔画出格多，比印刷体多留边
            xs0 = [ocr[j]["box"][0] for j in ids]; ys0 = [ocr[j]["box"][1] for j in ids]
            xs1 = [ocr[j]["box"][2] for j in ids]; ys1 = [ocr[j]["box"][3] for j in ids]
            pady = max(12, 0.10 * (max(ys1) - min(ys0)))
            bb = _clampbox([min(xs0) - 14, min(ys0) - pady, max(xs1) + 14, max(ys1) + pady])
            used_ocr.update(ids)
        entry = {"id": f"k{i}", "text": ktext, "bbox": bb, "ocr_ids": ids,
                 "color": k.get("color") or "#FFFFFF",
                 "polarity": k.get("polarity") if k.get("polarity") in ("light", "dark") else "light",
                 "notes": ("AI 起草 · " + str(k.get("notes") or "")).rstrip(" ·")}
        if k.get("method") == "art": entry["method"] = "art"
        out["calligraphy"].append(entry)
    _pnames = set()
    for i, h in enumerate([h for h in spec.get("persons") or [] if isinstance(h, dict) and h.get("bbox")], 1):
        name = str(h.get("name") or f"人物_{i}")
        name = name if name.startswith("人物_") else "人物_" + name
        k2 = 2   # 重名去重：两个"人物_女孩"图层文件会互相覆盖
        base = name
        while name in _pnames: name = f"{base}{k2}"; k2 += 1
        _pnames.add(name)
        out["persons"].append({"id": f"h{i}", "name": name,
                               "bbox": _clampbox(h["bbox"]), "notes": str(h.get("notes") or "")})
    for i, o in enumerate([o for o in spec.get("objects") or [] if isinstance(o, dict) and o.get("bbox")], 1):
        name = str(o.get("name") or f"物品_{i}")
        out["objects"].append({"id": f"o{i}", "name": name if name.startswith("物品_") else "物品_" + name,
                               "bbox": _clampbox(o["bbox"]),
                               "method": o.get("method") if o.get("method") in ("saliency", "table", "grabcut") else "saliency",
                               "notes": str(o.get("notes") or "")})
    out["decor_keep_in_bg"] = [str(x if isinstance(x, str) else (x or {}).get("notes") or "") for x in spec.get("decor_keep_in_bg") or []]
    pn = spec.get("page_number")
    if isinstance(pn, dict) and pn.get("bbox"):
        out["page_number"] = {"text": str(pn.get("text") or ""), "bbox": _clampbox(pn["bbox"]),
                              "color": pn.get("color") or "#FFFFFF", "font": "Arial"}
    missed = [i for i, o in enumerate(ocr or []) if i not in used_ocr and o.get("conf", 0) >= 0.7]
    if missed:
        warnings.append(f"有 {len(missed)} 条 OCR 高置信行未被任何文字块引用（编号 {missed[:8]}…），请核对是否漏标")
    return out


def test_connection(cfg=None):
    """轻量连通性测试（不传图片）。返回描述字符串；失败抛 RuntimeError。"""
    cfg = dict(cfg or load_ai_cfg())
    preset = PRESETS.get(cfg.get("model") or "")
    if not preset:
        raise RuntimeError(f"未知模型预设：{cfg.get('model')}")
    cfg["_preset"] = preset
    if preset["kind"] == "mock":
        return "Mock 通道正常（离线联调模式）"
    if not (cfg.get("key") or "").strip():
        raise RuntimeError("尚未填写 API Key")
    t = time.time()
    if preset["kind"] == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=cfg["key"])
        try:
            r = client.messages.create(model=cfg.get("model_id") or preset["model"], max_tokens=16,
                                       messages=[{"role": "user", "content": "回复：OK"}])
        except anthropic.AuthenticationError:
            raise RuntimeError("API Key 无效")
        except anthropic.APIConnectionError:
            raise RuntimeError("无法连接 Anthropic API")
        except anthropic.APIStatusError as e:
            raise RuntimeError(f"API 错误 {e.status_code}")
        model_used = r.model
    else:
        base = cfg.get("base_url") or preset["base"]
        if not base:
            raise RuntimeError("该预设需要在设置里填写「Base URL 覆盖」")
        base = base.rstrip("/")
        req = urllib.request.Request(base + "/chat/completions",
            data=json.dumps({"model": cfg.get("model_id") or preset["model"], "max_tokens": 16,
                             "messages": [{"role": "user", "content": "回复：OK"}]}).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + cfg["key"]})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                model_used = json.load(r).get("model", preset["model"])
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"服务返回 {e.code}：{e.read().decode(errors='replace')[:160]}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"无法连接：{e.reason}")
    return f"连接正常 · {model_used} · {time.time()-t:.1f}s"


# ---------------------------------------------------------------- 入口
def load_ai_cfg():
    # 云端：server 派发任务时注入生效配置（账号中心产品属性，key 在后台管理系统里维护）
    env_ai = os.environ.get("SLIDELIFT_AI_JSON")
    if env_ai:
        try:
            cfg = json.loads(env_ai)
            if isinstance(cfg, dict) and cfg:
                cfg.setdefault("enabled", True)
                return cfg
        except Exception:
            pass
    try:
        cfg = json.load(open(os.path.join(ROOT, "slidelift.json"))).get("ai") or {}
    except Exception:
        cfg = {}
    # 云端兜底（直跑脚本、无环境注入时）：slidelift_cloud.json 的 ai 段
    if os.environ.get("SLIDELIFT_CLOUD") == "1":
        try:
            cloud_ai = json.load(open(os.path.join(ROOT, "slidelift_cloud.json"))).get("ai") or {}
        except Exception:
            cloud_ai = {}
        if cloud_ai:
            cfg = {**cfg, **cloud_ai}
            cfg["enabled"] = bool(cloud_ai.get("enabled", True))
    return cfg


def draft_page(deck, n, cfg=None):
    cfg = dict(cfg or load_ai_cfg())
    preset = PRESETS.get(cfg.get("model") or "Qwen-VL-Max")
    if not preset:
        raise RuntimeError(f"未知模型预设：{cfg.get('model')}")
    cfg["_preset"] = preset
    if preset["kind"] != "mock" and not (cfg.get("key") or "").strip():
        raise RuntimeError("尚未配置 API Key（设置 → AI 自动标注起草）")
    dd = os.path.join(ROOT, "work", deck)
    src = os.path.join(dd, "src", f"p{n:02d}.png")
    if not os.path.exists(src):
        raise RuntimeError(f"源图不存在：p{n:02d}")
    ocr = []
    try:
        ocr = json.load(open(os.path.join(dd, "ocr", f"p{n:02d}.json")))
    except Exception:
        pass
    if preset["kind"] == "mock":
        raw = _call_mock(ocr)
        parsed = _extract_json(raw)
    else:
        img_b64 = base64.standard_b64encode(open(src, "rb").read()).decode()
        call = _call_anthropic if preset["kind"] == "anthropic" else _call_openai
        parsed, raw, last_err = None, "", None
        for attempt in range(3):   # 文字密集页模型偶发吐坏 JSON——带错误提示重试
            ut = _user_text(ocr) if attempt == 0 else _user_text(ocr) + (
                f"\n\n【重要】你上一次的输出不是合法 JSON（{last_err}）。这次只输出一个严格合法的 JSON 对象："
                '字符串值里的英文双引号必须转义为 \\"，字符串内不得出现换行符，JSON 之外不要有任何文字。')
            raw = call(cfg, img_b64, ut)
            try:
                parsed = _extract_json(raw); break
            except Exception as e:
                last_err = str(e)[:160]
                print(f"p{n:02d} 第 {attempt + 1} 次起草输出解析失败（{last_err}），重试…", flush=True)
        if parsed is None:
            rawp = os.path.join(dd, "spec", f"p{n:02d}.draft_raw.txt")
            os.makedirs(os.path.dirname(rawp), exist_ok=True)
            open(rawp, "w").write(raw or "")
            raise RuntimeError(f"模型 3 次输出均无法解析（{last_err}）；原始返回已存 spec/p{n:02d}.draft_raw.txt")
    warnings = []
    spec = _normalize(parsed, ocr, warnings)
    spec = {"page": n, "size": [W0, H0], **spec}
    return spec, warnings


def _draft_one(deck, n, cfg):
    sp = os.path.join(ROOT, "work", deck, "spec", f"p{n:02d}.json")
    if os.path.exists(sp):
        print(f"p{n:02d} skip（已有标注，不覆盖）", flush=True); return "skip"
    try:
        t = time.time()
        spec, warns = draft_page(deck, n, cfg)
        os.makedirs(os.path.dirname(sp), exist_ok=True)
        json.dump(spec, open(sp, "w"), ensure_ascii=False, indent=1)
        print(f"p{n:02d} ok 文字块={len(spec['text_zones'])} 书法={len(spec['calligraphy'])} "
              f"人物={len(spec['persons'])} 底板={len(spec['panels'])} {time.time()-t:.0f}s"
              + (f" 提醒:{'；'.join(warns[:2])}" if warns else ""), flush=True)
        return "ok"
    except Exception as e:
        print(f"p{n:02d} FAIL: {e}", flush=True); return "fail"


def main():
    deck = sys.argv[1]
    pages = [int(p) for p in sys.argv[2:]]
    cfg = load_ai_cfg()
    # 多页并发起草：各页互相独立（独立 OCR 输入、独立 spec 输出），API 等待占大头，并发直接摊薄
    par = max(1, int(os.environ.get("SLIDELIFT_DRAFT_PAR", "4")))
    if par > 1 and len(pages) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(par, len(pages))) as ex:
            rs = list(ex.map(lambda n: _draft_one(deck, n, cfg), pages))
    else:
        rs = [_draft_one(deck, n, cfg) for n in pages]
    ok, skip, fail = rs.count("ok"), rs.count("skip"), rs.count("fail")
    print(f"DONE 起草 {ok} 页，跳过 {skip} 页，失败 {fail} 页", flush=True)
    sys.exit(1 if (fail and not ok) else 0)


if __name__ == "__main__":
    main()
