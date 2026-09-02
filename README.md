# imagePPTedit（SlideLift）— 位图 PPT 分层还原

把"整屏都是一张大图"的 PPT 或海报，还原成**分层可编辑**的 PPTX：干净背景、人物/物品/文字底板独立图层、可编辑文字，还原度逐像素级。

专为 AI 编码助手设计：在 **Claude Code / Codex** 里加载本仓库，agent 会亲自完成看图标注、跑引擎、看质检对比图、按速查表修正的全流程——你只需要把文件给它，用自然语言提要求（"把标题往左挪""这个字错了"）。

## 快速开始

```bash
git clone git@github.com:coolbreezecoin/imagePPTedit.git
cd imagePPTedit
bash setup.sh        # 一次性：Python 环境 + 模型权重(约630MB) + 渲染器检测
```

然后在 Claude Code（或 Codex）中打开本目录，说：

> 把 ~/Desktop/xxx.pptx 还原成分层可编辑版本

agent 会按 [SKILL.md](SKILL.md) 的工作流执行。手动使用见同文件的命令序列。

## 它能做什么

- 整屏位图 PPTX / 单张或多张图片（海报、截图）→ 分层 PPTX
- 文字重建为可编辑文本框（字体、字号、行距、字距自动校准到与原图对齐）
- 人物、物品、文字底板（含半透明雾板的真实 alpha 解混）各自成层，可独立移动
- 毛笔书法/特效艺术字整块保真成层（转字体必失真的内容不硬转）
- 质检：每页生成"原图 vs 重建"对比图与还原度分数

## 注意

- 仓库请放在**用户目录**下（如 ~/imagePPTedit）——macOS 的 PowerPoint 无法访问 /tmp 等系统路径，渲染会报 -9074。

## 环境要求

- macOS / Linux（Windows 未测试）
- Microsoft PowerPoint 或 LibreOffice（质检渲染用；setup.sh 会检测）
- 磁盘 ~3GB（依赖 + 权重），无 GPU 要求（CPU 推理，Apple Silicon 上单页约 10-30 秒）

## 不想装环境？

云端版开箱即用（网页上传、在线审校工作台）：**https://montage.ink**

## 已知限制

躺姿/严重遮挡人物可能抠不全；超宽物料（横幅比例）精度受损；密排小字默认整块保真成层（可按块改判）。详见 [SKILL.md](SKILL.md)。
