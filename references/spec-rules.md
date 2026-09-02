# 页面标注规则（spec/pNN.json）

画布坐标：左上角为原点，宽 1672 高 941；bbox=[x0,y0,x1,y1] 整数。写标注前必须仔细看页面图 + OCR 候选行（`ocr/pNN.json`，行号从 0 起，含 box 与文本）。

## 图层分类（按"用户会不会想单独动它"来分）

- **text_zones**：常规印刷字体的一切文字（含页码、表格单元格、胶囊标签文字）→ 重建为可编辑文本框。每个 zone 必须填 style：
  - `"plain"`：普通印刷体，转成可编辑文字能以假乱真；
  - `"art"`：特效美术字——立体挤出/描边勾边/渐变彩色/金属光泽/发光/艺术变形/超粗装饰大标题（海报主标题几乎都是）→ 引擎整块抠成图层保真。一行里多种颜色的标语也算 art。拿不准选 art（保真优先）。
- **calligraphy**：毛笔书法/手写字（笔画毛糙有飞白、同字内粗细变化大）→ 抠成透明图层。若 OCR 识别出了这段文字，填 ocr_ids（引擎会按行框收紧位置）。
- **persons**：人物（整个人，含头发、手、衣摆），name 用"人物_xx"。bbox 宁大勿小，必须完整包住（含被桌子遮住前的身体）。
- **objects**：大件可移动物品（桌子及桌上物合为一层、独立的灯/船/椅），name 用"物品_xx"；method 从 saliency(默认)/table(平直大件)/grabcut(被子船等) 选。**notes 里写英文物体名**（如 smartphone、lantern——检测模型用）。散布的组合物件（茶具一套）标一个即可，引擎自动检出成员。**边界清晰的独立小图标/贴纸**（对话气泡、徽章、小 logo——压在标题或人物上的）也标 objects（method:grabcut），**框要紧贴图标本身**；漏标会被邻近大层吸走，拖动时撕裂。框大而虚（内容大半是背景）的物件会被引擎验收拒绝。
- **panels**：字底板——文字下方的卡片/圆角框/胶囊/半透明白板/表格色块/答案框，bbox 包住整个底板含边框；kind: card|capsule|panel(大面积半透明板)|cell|box|badge；fill: opaque_light|opaque_dark|translucent_light|translucent_dark（透过底板能看到背景细节=translucent）。文字衬板**不论多大**都要标（大面积柔光白雾板 kind="panel" fill="translucent_light"）。
- **decor_keep_in_bg**：装饰件（星星、分隔线、虚线、水彩晕染）→ 字符串数组，留在背景。**鲜艳彩色/渐变装饰横条不要标 panel**（会产空层）——写这里。

## text_zones 硬规则

- 一个 zone = 同字体、同颜色、同粗细、同对齐、同容器的连续文字，可多行；缩进不同的行拆 zone，或段首缩进用两个全角空格表示。
- 行距突变处拆开；表格每格一个 zone；页码单独 zone（container="bg", font="Arial"），并同时填 page_number。
- lines 按画面视觉分行逐行写，保留全角标点。**OCR 候选只是候选**：内容、标点、分行必须对图逐字核对；OCR 漏掉的要补。
- 行内前缀不同色加粗（如「点拨：」）写成 `{"runs":[{"t":"点拨：","bold":true},{"t":"其余文字"}]}`。
- ocr_ids：与 lines 逐行一一对应且文本一致（引擎按行框收紧 bbox）；对不上就留空数组并自己给准 bbox，禁止乱填。
- **已标 calligraphy 的文字绝不能再建 text_zone**（同一标题标两次是最常见错误）。
- bbox 每边比墨迹松 4~8px，但不越出所在底板边框。
- font 只能选：楷体|微软雅黑|黑体|华文行楷|宋体|Arial。polarity：light=深底浅字，dark=浅底深字。
- container："bg" 或 "panel:<底板id>"；effects：柔光晕 glow:true，偏移暗影 shadow:true。
- line_mode：normal|single(单个大字)|tight(行距很小)|pinyin(带声调拼音)。

## 输出结构（JSON 文件，数组必须是 `[...]`，严禁键值形式）

```json
{"layout_notes":"一句话版式描述",
 "text_zones":[{"id":"t1","lines":["…"],"bbox":[0,0,0,0],"ocr_ids":[0,1],"style":"plain","font":"微软雅黑","bold":false,"color":"#FFFFFF","polarity":"light","align":"left","container":"bg","effects":{"glow":false,"shadow":false},"line_mode":"normal","notes":""}],
 "calligraphy":[{"id":"k1","text":"课前导入","bbox":[0,0,0,0],"ocr_ids":[0],"color":"#FFFFFF","polarity":"light","notes":""}],
 "persons":[{"id":"h1","name":"人物_女孩","bbox":[0,0,0,0],"notes":""}],
 "objects":[{"id":"o1","name":"物品_书桌","bbox":[0,0,0,0],"method":"table","notes":"desk with books"}],
 "panels":[{"id":"c1","name":"底板_卡片1","kind":"card","bbox":[0,0,0,0],"shape":"rounded_rect","fill":"translucent_dark","border":true,"notes":""}],
 "decor_keep_in_bg":["标题下方紫色装饰线与星星"],
 "page_number":{"text":"4","bbox":[0,0,0,0],"color":"#FFFFFF","font":"Arial"}}
```

自检：画面上每一个字都被某个 text_zone 或 calligraphy 覆盖；zone 不压底板边框；书法拿不准写 notes。
