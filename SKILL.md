---
name: slidelift
description: 把"整屏位图"的 PPT 页面或海报图片还原为分层可编辑的 PPTX（背景/人物/物品/文字底板分层，文字转可编辑文本框）。当用户提供扁平化的 PPTX 或图片并想要可编辑分层版本时使用。
---

# SlideLift — 位图 PPT 分层还原

把只有整屏大图的 PPT（或海报图片）拆解为：干净背景 + 人物/物品/文字底板独立图层 + 可编辑文字，重新组装成还原度逐像素级的 PPTX。

## 首次使用（一次性安装）

```bash
bash setup.sh
```

安装 Python 环境、下载模型权重（约 630MB）、检测渲染器。Mac 需要 Microsoft PowerPoint 或 LibreOffice；Linux 需要 LibreOffice（质检渲染用）。Windows 未经测试。

## 标准工作流

用户给你一个 PPTX 或若干图片后，按顺序执行（`<deck>` 用简短英文/拼音名）：

```bash
# 1) 解包（二选一）
.venv/bin/python pipeline/unpack_pptx.py 输入.pptx work/<deck>
.venv/bin/python pipeline/unpack_images.py work/<deck> 图1.png 图2.jpg

# 2) OCR 候选
.venv/bin/python pipeline/ocr_pages.py <deck>
```

**3) 起草标注（你亲自做，这是效果的关键）**：逐页 Read `work/<deck>/src/pNN.png`（页面图）和 `work/<deck>/ocr/pNN.json`（OCR 候选行，含行号与框），按 [references/spec-rules.md](references/spec-rules.md) 的规则 Write `work/<deck>/spec/pNN.json`。每页都写完再进下一步。

```bash
# 4) 全自动处理（起草步会自动跳过已有 spec 的页）
zsh pipeline/run_all.sh <deck>
```

**5) 质检（你亲自看）**：Read `work/<deck>/qa/pNN_cmp.jpg`（左原图右重建）逐页对比；每页还原度分数在 `qa/mae.json`（<8.5 为佳）。发现问题按 [references/troubleshooting.md](references/troubleshooting.md) 修改该页 spec，然后只重跑该页：

```bash
zsh pipeline/rerun_pages.sh <deck> 3 5      # 只重跑第 3、5 页
```

**6) 交付**：`output/<deck>_分层可编辑.pptx`。同时把 `qa/` 里的对比图给用户看一眼还原效果。

## 用户要求修改时

- "把 XX 移一下 / 改个字"：这类编辑直接改 spec（文字在 text_zones 的 lines；位置在 bbox），然后 `zsh pipeline/rerun_pages.sh <deck> <页码>`。
- 只是组装层面的改动（挪图层、删图层）：改 spec 顶层 `edits` 后跑 `zsh pipeline/reassemble.sh <deck>`（秒级，不重新抠图）。

## 边界与已知限制

- 躺姿/严重遮挡的人物可能抠不全（分割模型能力边界）。
- 超宽物料（横幅）装进 16:9 画布后精度受损。
- 密排小字（行高 <20px）默认整块保留为图片层——要编辑哪块，把那块的 style 改为 "plain" 单独重跑。
- 毛笔书法永远保留为图片层（转字体必失真）。
