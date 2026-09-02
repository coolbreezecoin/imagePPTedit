# PPT 位图 → 可编辑分层稿：工作流说明

> 目标：把"每页一张整屏位图"的 PPTX 还原成同版式的可编辑 PPTX——文字是真文本框、画面拆成中文命名的独立图层（背景/人物/物品/字底板/书法艺术字），并与原图逐像素接近。
> 规范来源：`../PPT图片分层还原_作业规范.md`（本流水线按其 §4~§10 实现，像素参数按 `宽度/3840` 缩放）。

## 0. 环境（一次性）
```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy opencv-contrib-python-headless pillow scipy scikit-image \
   python-pptx lxml pymupdf fonttools matplotlib tqdm "rembg[cpu]" simple-lama-inpainting rapidocr-onnxruntime
# 模型权重：rembg 首次运行自动下载到 ~/.rembg/models/（u2net_human_seg, isnet-general-use）
# LaMa：curl -L -C - -o models/big-lama.pt https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt
```
依赖本机：**Microsoft PowerPoint（AppleScript 导出 PDF 作为渲染器）**、ImageMagick（裁剪看图）。

## 1. 目录约定
```
work/<deck>/src/pNN.png        源图（从 pptx 的 ppt/media 解包，按幻灯片顺序重命名）
work/<deck>/ocr/pNN.json       OCR 候选（RapidOCR，2x 放大识别）+ pNN_vis.png（带编号红框）
work/<deck>/spec/pNN.json      页面规格（文字区/书法/人物/物品/底板/装饰），见 SPEC_GUIDE.md
work/<deck>/ink/pNN.json       每个 zone 的墨迹测量（行框、极性、颜色、描边/发光）
work/<deck>/clean/pNN_text.png 去文字后的整图 + pNN_textmask.png
work/<deck>/layers/pNN/        图层 PNG（背景.png / 人物_xx.png / 物品_xx.png / 底板_xx.png / 书法_xx.png）+ manifest.json + _sheet.jpg（检查图）
work/<deck>/calib/params.json  文字几何标定结果（字号/行距/位置），calib_iterN.pptx/pdf 为标定稿
work/<deck>/qa/pNN_cmp.jpg     原图|重建|差异 并排图；mae.json
output/<deck>_分层可编辑.pptx  成品
```

## 2. 流程（对一个新 deck）
1. **解包**：`unzip` pptx → 把 `ppt/media/imageN.png` 按 `ppt/slides/slideN.xml` 的引用顺序拷到 `work/<deck>/src/pNN.png`；记录 `p:sldSz`。
2. **OCR**：`.venv/bin/python pipeline/ocr_pages.py <deck>`
3. **写 spec**（人工/模型逐页核对，最关键的一步）：按 `SPEC_GUIDE.md`。要点：
   - 每页亲眼看全图 + 放大核对字体/粗细/标点/分行；OCR 只当候选。
   - zone 内各行必须同左缘或同中心；缩进行拆 zone 或用全角空格“　　”表示段首缩进。
   - 同一行不同颜色的前缀（“点拨：”“示例1：”）用 runs 富文本：`{"runs":[{"t":"点拨：","bold":true},{"t":"其余..."}]}`，颜色留空由程序实测。
   - 楷体正文常是"楷体+加粗"（原稿笔画明显粗于常规楷体时 `bold:true`）。
   - 毛笔书法/手写艺术字 → `calligraphy`（抠图不转字）；印刷楷体/行楷 → `text_zones`。
   - 大面积纸色/深色面板（占半屏以上、柔和过渡）归背景；卡片/胶囊/单元格/答案框/笔刷徽章 → `panels`。
   - 桌子类平直大件 `"method":"table"`（桌沿线+GrabCut），被子/船等 `"grabcut"`，小件 `"saliency"`。
4. **一键跑**：`pipeline/run_all.sh <deck> [pages...]`（erase → layers → sheets → calib(5轮) → assemble → qa）
5. **验收**：看 `work/<deck>/qa/pNN_cmp.jpg`（**逐页目视是唯一合格判据**，MAE 只用来筛可疑页）、`layers/pNN/_sheet.jpg`（抠图质量）。有问题 → 改 spec/参数 → 只重跑该页：`pipeline/run_all.sh <deck> 12 13`。
6. **交付**：`output/<deck>_分层可编辑.pptx`（原文件不动）。

## 3. 各阶段做了什么（对应规范条款）
| 脚本 | 内容 |
|---|---|
| `erase_pages.py` + `ppt2layers/ink.py` `erase.py` | §4.2 腐蚀细度判极性（spec 为准、程序告警）；§5.1 形态学包络掩膜（亮字 OPEN/暗字 CLOSE，核≈0.28 行高 5~15）；§5.2 色块/胶囊/单元格用色片百分位阈值+中位色重涂；§4.3 行投影切分（自适应阈值、按期望行数纠偏、OCR 行框先验、列间隙聚类剔除亮星）；§4.4 墨迹盒 ratio 检查；§4.6 发光/描边检测；§5.3 逐行掩膜+高阈值硬裁剪+自适应膨胀；**全页文字掩膜合并后一次 LaMa**（逐 zone 修补会被邻近文字诱导出笔画残影）。 |
| `layers_pages.py` + `ppt2layers/matte.py` | §6 书法 alpha（大核估最粗笔画→包络→相对局部峰值归一化→抬地板→去细线→封闭孔填补→颜色外推）；底板最近颜色分割（阈值=中心色与外围色差一半）；人物 u2net_human_seg 软 alpha + smoothstep 压实内部；物品：显著性 isnet + 桌沿 Hough 线 + GrabCut；背景 = LaMa 修补所有被抠走的区域。书法压在不透明色块上时用色块中位色重涂。 |
| `calib_pages.py` + `ppt2layers/calib.py` `pptx_build.py` | §4.5 渲染闭环：白底黑字标定稿 → PowerPoint 导出 PDF → PyMuPDF 渲染 1672px → 在目标行框附近测墨迹盒 → CJK 行按行宽定字号（阻尼 0.65）、数字行按高度；dx/dy/行距分别修正；5 轮收敛到 |w|≈0.5%、|dx|<0.2px。 |
| `assemble.py` | §9 像素→EMU；文本框关闭换行/自适应、零内边距、固定行距；字体名写入 latin/ea/cs（楷体/微软雅黑/黑体 + Arial）；描边（a:ln）/发光（effectLst）效果；图层中文命名；叠放顺序 背景→人物→物品→底板→书法→文字。 |
| `qa_pages.py` | 成品 → PDF → 渲染 → 与原图 MAE + 并排差异图。 |

## 4. 关键经验（本 deck 踩过的坑）
- **逐 zone 调 LaMa 会留字形残影**（模型顺着邻近文字"续写"），必须全页合并掩膜一次修补。
- **书法抠图的核必须大于最粗笔画**，否则粗笔内部出空洞；细笔/飞白靠"相对局部峰值"的地板保住。
- 分割模型的软 alpha 内部常只有 0.8~0.95，直接用会让修补背景透出来 → smoothstep 压实。
- 白字带深色细描边：最贴近原图的重建是 **加粗 + 1~1.6px 描边**（描边一半吃进字形，正好抵消加粗）；单用外阴影太糊、单用描边字会变瘦。
- Mac Office 自带 KaiTi/SimHei/MicrosoftYaHei/STXingkai 等 Windows 字体 → 直接写 Windows 字体名即可跨平台。
- 桌子这类平直大件显著性模型抓不到：桌沿线以下 + 显著物件 作为 GrabCut 确定前景，亮色区域作可能背景。
- 文字 zone 的 bbox 不能越出容器/照片边缘（亮星会把行宽撑爆）；给每行 OCR 框做先验最稳。

## 5. 已知限制 / 待改进
- 半透明白色柔光板（§7 解混）未单独实现：本 deck 的半透明卡片按"不透明裁切"处理（移动时会带走一点底下的照片）。
- 被人物/桌子遮住的背景由 LaMa 想象补全，只保证"移开图层时不难看"。
- 书法层不带发光效果（原稿若有轻微光晕会丢失）。
- 字体：原稿若用了非系统字体（如思源黑体/苹方），重建用微软雅黑/楷体替代，字形略有差异（位置与字号精确匹配）。

## 6. 运行注意（血泪补充）
- **PowerPoint 是全局单例**：绝不能让两条流水线同时导出（已在 export_pdf 加锁 + 失败重试）。让 PowerPoint 打开不存在的文件会弹模态框，卡死后续一切 AppleEvent——出现 -9074/-1712/“用户已取消”连环报错时，`pkill -9` PowerPoint 再 `open -a` 重启即可恢复。
- **阶段日志不要过滤 stderr**：calib 崩了但 grep 只留 "iter" 行，导致组装静默产出"零文本框"的成品；而这样的成品 MAE 依然只有 ~7/255。**MAE 连"文字全丢"都测不出来，唯一可靠的验收是逐页并排目视。**
- run_tail.sh 只跑 标定→组装→QA；改了 spec/抠图代码要先跑对应页的 erase/layers。

## 7. 本地图形界面（SlideLift 正式版，M2）
- 启动：`.venv/bin/python app/server.py`，或双击 `app/启动SlideLift.command`（自动开浏览器）。地址 http://127.0.0.1:8765，仅本机可访问。
- 前端 = 交互原型（prototype/）的实装版：S0 环境自检 / S1 项目库 / S2 导入 / S3 总览（筛页+快速分诊+批量重跑）/ S4 审校台（审阅四式对比 + 画布标注 + 属性面板 + 自动保存 + 单页重跑）/ S6 导出（PPTX/PNG 包 + 还原报告.html）/ S7 设置 / 任务中心。旧版简易 UI 备份在 `app/static_v1/`。
- 后端新增 v2 API：项目载荷（对齐原型 data.js 形状）、审校标记 `work/<deck>/review.json`、热区图按需生成缓存 `qa/pNN_heat.jpg`、还原报告生成、图层 ZIP 导出、回收站恢复、环境自检、设置 `slidelift.json`、任务进度解析。
- **标注即保存**：审校台里的改动 0.9s 后自动写回 spec（带 .bak 备份与服务端校验）；「重跑本页」= 保存 + rerun_pages.sh + 轮询刷新。
- 功能：上传 PPTX 自动解包（`pipeline/unpack_pptx.py`；非 1672×941 的 16:9 源图会重采样并在 deck.json 标记 resized，宽高比偏离 16:9 超 3% 直接拒绝）；页面网格显示 spec/图层/MAE 状态；页面详情 = QA对比/原图/图层拼版/OCR标注/去字图/重建图 六视图 + spec 编辑器（保存前校验、自动 .bak 备份、可从 OCR 起草）；任务串行队列（保证 PowerPoint 单例安全），提供 OCR / 一键全跑 / 尾段 / 重跑选中页 / 重拼检查图 / 取消 / PowerPoint 急救（pkill）。
- **重跑选中页**走 `pipeline/rerun_pages.sh`：仅对选中页做 擦除→图层→拼版，随后**全量**标定→组装→QA——避免 assemble 传子集页码把完整成品覆盖成只含子集的 pptx（run_all.sh 传子集会这样，仅适合出中间稿）。

## 8. AI 标注起草（M3）
- `pipeline/ai_draft.py`：视觉大模型按标注规范起草整页标注。四家适配：Qwen-VL-Max（DashScope 兼容模式）/ GPT-4o / Kimi-VL（均为 OpenAI 兼容协议，urllib 直连）+ Claude（官方 anthropic SDK，模型 claude-opus-5，默认启用服务端安全回退 fallbacks）。另有 Mock 通道（model="Mock"）离线联调。
- **OCR 锚定**是准确率关键：提示词让模型给每个文字块填 ocr_ids，服务端把 bbox 收紧为对应 OCR 行框并集 +6px——模型只负责"分组与判断"，几何精度由 OCR 保证。归一化还做：字体白名单、底板 id 重映射、bbox 收编、未覆盖高置信 OCR 行告警。
- 入口：单页 `POST /api/v2/deck/<deck>/page/<n>/ai_draft`（返回草稿不落盘，前端确认后自动保存）；批量=任务阶段 `ai_draft`（逐页写 spec，**跳过已有标注的页**，不覆盖）；连通测试 `POST /api/v2/ai_test`。配置在 `slidelift.json` 的 ai 段（设置页维护，key 明文存本机）。
- 界面入口：审校台"✨ AI 起草（推荐）"（空页起草）、总览"AI 起草未标注页 N"（批量）、导入向导勾选"OCR 完成后自动起草全部页"（与 OCR 任务串行排队）。

## 9. 检测校正 + SAM2 分割（M3.5，"Canva 架构"落地）
- 病根：VLM 报坐标不可靠（框脱靶/整体错位），框提示分割救不了脱靶框。解法=几何与语义分家：**YOLO-World 开放词汇检测负责"在哪"，SAM2 负责"抠多准"，VLM 只管"是什么/叫什么"**。
- `ppt2layers/detect.py`：YOLO-World（models/yolov8s-worldv2.pt）+ 中文名→英文检测词表 CN2EN（spec 可带 hint_en 覆盖）。`refine_box`（物品：IoU≥0.45 尊重原框；有交集微调；脱靶取最高置信检测框）；`assign_person_boxes`（人物：人数=检测数时**按中心横坐标同序配对**——VLM 常整体错位，IoU 贪心会集体连坐；否则贪心+中心距补配）。
- `ppt2layers/sam_matte.py`：SAM2（models/sam2.1_t.pt）框提示分割。**必须把目标区域裁出并补成正方形再喂**——ultralytics 对非方形整图的 bboxes 提示存在坐标缩放 bug（掩膜系统性偏移）；默认 CPU（MPS 数值不稳）。
- `layers_pages.py` 接线：人物=检测框→SAM 实例门×u2net 软 alpha（门内保发丝，实例间切开重叠；不一致时回退 u2net）；物品=检测校正→SAM（覆盖率>3%），失败回退 saliency/table/grabcut 老链。
- 坑：ultralytics 的 CLIP fork 读**相对路径 `weights/clip/ViT-B-32.pt`**（跟随 CWD），自带下载器还会反复下 338MB 且校验失败——权重已放 `weights/clip/` 并软链到 `pipeline/weights/clip/`，别删。
- 实测（演示文稿5 p01，Qwen 起草框全偏）：灯笼从星空拉回真位、书桌薄框长成整桌、三人同序归位重叠切开——**人工修正 5–6 处 → 0 处**，逐层 alpha 覆盖率 0.35–0.79 全健康。
- 小尾巴三连（同日）：**组合物件**（茶具/餐具/文具… COMPOSITE_CN 词表）逐成员检测→逐框 SAM→掩膜求并，"茶具"从单壶变两碗+壶一组；**书法 OCR 锚定**（起草归一化：模型给 ocr_ids 或按"OCR 行与外扩框重叠≥30%+字符交集"自动锚定，框收紧到笔迹±14px、纵向 +10% 留飞白），偏一个身位的书法框能自动归位，且不再误报"OCR 行未引用"；**底板 SAM 兜底**（颜色分割覆盖率<25% 时改用 SAM，救笔刷徽章/渐变底板）。
- 演示文稿7 两问题的根治（同日）：**自动补检**——VLM 起草漏标物品是常态（同页两次起草结果都不同），layers 阶段用通用词表（EN2CN）全图检测，未被认领（与已有标注 IoU>0.15 才算认领；不能按"中心落入"排除——桌上物件中心必然在桌框内）的高置信(≥0.15)物件自动补层并回写 spec（notes 标"引擎自动补检"，界面可删、重跑幂等）；邻近同族检测先聚类合并（壶身/嘴/盖会拆成多个 cup 框），茶系混合命名"茶具"走组合成员抠取。**半透明衬板解混**（unmix_translucent）——LaMa 修补板区得背景 B，按 I=aC+(1-a)B 反解 alpha 与板色，板成真半透明单色层、背景干净；提示词规则同步改为"文字衬板不论多大都标 panel（fill=translucent_*）"，不再"大面积归背景"。

## 10. 傻瓜模式（上传即全自动，2026-08-28）
- `pipeline/auto_all.sh <deck>`：一条龙 OCR → AI 起草（跳过已有标注）→ 整卷生产。任务阶段 "auto"；上传时若 AI 已启用即自动排队（app.js S2）。**坑：ocr_pages.py 用相对路径必须从项目根跑**；每步产物有硬校验快速失败（OCR 无产出/零标注即报错，防止静默空跑连环炸）。
- **每页信心聚合** `_page_confidence`（server）：还原度≥8 / 图层抠空⚠ / 引用 OCR 行置信<0.88（错字嫌疑）/ >30% 行未锚定 OCR / 有 OCR 文字但零文字块 → "建议看一眼"，否则"有把握"。deck 载荷带 `check_pages`，页载荷带 `confidence/reasons`（审校台顶栏黄签展示）。
- 界面：完成弹结果卡（N 页有把握 · K 页建议看一眼 → [直接导出] [看一眼]）；项目卡完成态主按钮=导出，副按钮=看一眼 K 页；导出报告"结论"含机器自检行与"未经人工复核以自检为准"声明。导出永不拦死。
