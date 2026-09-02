# 页面规格（spec）标注指南

每页产出一个 JSON：`work/<deck>/spec/pNN.json`。像素坐标以源图 **1672×941** 为准（左上角为原点，bbox=[x0,y0,x1,y1]）。
可参考 `work/<deck>/ocr/pNN.json`（OCR 候选文字，带 box 与 conf）和 `work/<deck>/ocr/pNN_vis.png`（红框+编号）。
OCR 只是候选：文字内容、标点、分行都必须对着原图逐字核对；OCR 漏掉的文字（页码、小标签、表头）要补上。

## 1. 图层分类（按"用户会不会想单独动它"来分）

| 字段 | 含义 | 归类规则 |
|---|---|---|
| `text_zones` | 常规字体排版的一切文字（含页码、表格单元格里的字、胶囊标签里的字、选项字母） | 重建为可编辑文本框 |
| `calligraphy` | 毛笔书法/手写艺术字（有飞白、笔锋、粗细变化明显，笔触像真毛笔） | 抠成透明 PNG，不转文字 |
| `persons` | 人物 | 人像分割 |
| `objects` | 大件可移动物品：桌子+桌上书本/油灯/笔筒（合为一层"物品_书桌"）、独立的油灯/灯笼、船/椅子等 | 显著性分割 |
| `panels` | 字底板：文字下方的卡片/圆角框/胶囊标签/半透明白板/表格单元格色块/答案框 | 单独抠出（含边框和上面的小图标） |
| `decor_keep_in_bg` | 装饰件：星星、流星、装饰横线、分隔线、竖线、括号下划线、水彩晕染、批注圈等 | **留在背景层，不拆、不擦** |

叠放顺序（下→上）：背景 → 人物 → 物品 → 字底板 → 书法 → 文字。

**书法 vs 可编辑文字的判定**：
- 毛笔书法：笔画边缘毛糙/有飞白、同一字内粗细变化大、像用真毛笔写的（如封面大标题、"课前导入""教材习题答案""课堂小结""结构梳理""课后作业"这类章节标题）→ `calligraphy`。
- 楷体/行楷等**印刷字体**（笔画干净、同字号的字粗细一致、标点规整）→ `text_zones`，font 写"楷体"或"华文行楷"。
- 拿不准时写进 notes，并选 `calligraphy`（保真优先）。

## 2. text_zones 划分规则（硬规则）

一个 zone = 一块"同字体、同颜色、同粗细、同对齐、同容器"的连续文字，可以多行。
- **同一 zone 的各行必须左边缘对齐（左对齐）或中心对齐（居中）**。缩进不同的行（如题干第二行缩进、带"A."前缀的选项与正文）要拆成不同 zone。
- 行与行之间的行距明显不同（如小标题与正文之间有大空隙）→ 拆开。
- 表格每个单元格的文字各自一个 zone；胶囊标签里的字单独一个 zone；页码单独一个 zone。
- bbox 要略松（每边比文字墨迹多 4~8px），**但绝不能越出容器**（不能跨过卡片边框、色块边缘、表格线、面板边缘）。
- `lines`：按画面上的视觉分行逐行写，一行一个字符串，保留全角标点（，。！？“”（）、：；）。行内的多个空格（如"繁  星"）按原样保留。
- `ocr_ids`：对应的 OCR 条目编号（可多个，可为空）。

字段说明：
```
{
  "id": "t1",
  "lines": ["好像看见无数萤火虫", "在我的周围飞舞。"],
  "bbox": [1170, 185, 1490, 280],
  "ocr_ids": [2, 3],
  "font": "微软雅黑",          // 楷体 | 微软雅黑 | 黑体 | 华文行楷 | 宋体 | Arial(纯数字/英文)
  "bold": false,
  "color": "#FFFFFF",          // 目测文字颜色（粗略即可，后续程序会重测）
  "polarity": "light",         // light=亮字压暗底；dark=暗字压亮底
  "align": "left",             // left | center | right
  "container": "card:c1",      // bg（直接压在照片/大背景上） | card:<panel id> | panel:<panel id> | cell:<panel id>
  "effects": {"glow": false, "shadow": false},   // 字有没有外发光/投影（放大看边缘：有柔光晕=glow；有偏移的暗影=shadow）
  "line_mode": "normal",       // normal | single（单个大字/单个字母，如答案"B"） | pinyin（带声调的拼音）
  "notes": ""
}
```
字体判断要点：楷体=撇捺有起收笔、像手写的印刷体；微软雅黑/黑体=无衬线、笔画等粗（粗体写 bold=true，font 仍写"微软雅黑"）；宋体=横细竖粗有衬线；华文行楷=连笔的行书印刷体。

## 3. 其他字段

```
"calligraphy": [ {"id":"k1", "text":"课前导入", "bbox":[110,95,460,190], "color":"#FFFFFF", "polarity":"light", "notes":"白色毛笔字；正下方装饰线与星星不属于它"} ]
"persons":     [ {"id":"h1", "name":"人物_女孩", "bbox":[...], "notes":"站姿/坐姿、与桌子/书的遮挡关系"} ]
"objects":     [ {"id":"o1", "name":"物品_书桌", "bbox":[...], "notes":"桌子+书+笔筒合为一层"} ]
"panels":      [ {"id":"c1", "name":"底板_卡片1", "kind":"card", "bbox":[1105,160,1592,300], "shape":"rounded_rect",
                  "fill":"translucent_dark",   // opaque_light | opaque_dark | translucent_light | translucent_dark
                  "border": true, "notes":"左上角有⭐图标，属于底板"} ]
   kind 取值：card（圆角卡片）| capsule（胶囊标签）| panel（大面积半透明白板/柔光板，照片在下面透出来）| cell（表格单元格色块）| box（答案框/空白填写框）| badge（笔刷色块徽章，如"课堂练习"下的紫色笔刷块）
"decor_keep_in_bg": ["标题下方的紫色装饰线与星星", "左下角水彩晕染", "竖线分隔符"]
"page_number": {"text":"4", "bbox":[1626,888,1646,910], "color":"#FFFFFF", "font":"Arial"}
"layout_notes": "一句话描述版式"
```
- `panels` 的 bbox 要包住整个底板（含边框、圆角）；`fill` 判定：透过底板能看到照片细节=translucent，看不到=opaque。
- 人物 bbox 包住整个人（含头发、手、衣摆）；物品 bbox 包住整件。可以略松。
- 页码也要作为一个 text_zone 列出（container=bg），同时填 page_number。
- 大面积的纸张色背景（整面米色纸/深蓝色面板，占半屏以上、边缘与照片柔和过渡）**不算 panel**，归背景；写进 layout_notes 即可。

## 4. 完整示例（p04）
```json
{
  "page": 4,
  "size": [1672, 941],
  "layout_notes": "顶部米色横幅（楷体正文）；左侧照片（女孩凭栏望星空）；右侧深蓝面板上四张圆角卡片，白色黑体字，每张左上角一个彩色图标；右下角页码",
  "text_zones": [
    {"id":"t1","lines":["船在动，星也在动，它们是这样低，真是摇摇欲坠呢！ 渐渐地我的眼睛模糊了，我"],"bbox":[48,36,1630,82],"ocr_ids":[0,1],"font":"楷体","bold":false,"color":"#1A1A1A","polarity":"dark","align":"left","container":"bg","effects":{"glow":false,"shadow":false},"line_mode":"normal","notes":"米色横幅上的一整行"},
    {"id":"t2","lines":["好像看见无数萤火虫","在我的周围飞舞。"],"bbox":[1172,186,1486,278],"ocr_ids":[2,3],"font":"微软雅黑","bold":false,"color":"#FFFFFF","polarity":"light","align":"left","container":"card:c1","effects":{"glow":false,"shadow":false},"line_mode":"normal","notes":""},
    {"id":"t3","lines":["将繁星比作萤火虫，“无数”","写出了星星的多，对应题目","“繁星”。"],"bbox":[1170,348,1560,490],"ocr_ids":[4,5,6],"font":"微软雅黑","bold":false,"color":"#FFFFFF","polarity":"light","align":"left","container":"card:c2","effects":{"glow":false,"shadow":false},"line_mode":"normal","notes":""},
    {"id":"t4","lines":["比喻"],"bbox":[1176,534,1266,588],"ocr_ids":[7],"font":"微软雅黑","bold":false,"color":"#FFFFFF","polarity":"light","align":"left","container":"card:c3","effects":{"glow":false,"shadow":false},"line_mode":"normal","notes":""},
    {"id":"t5","lines":["“半明半昧”“摇摇欲坠”","极具动感，生动展现出繁星","明暗交替的光亮和星星低垂","的视觉效果。"],"bbox":[1170,646,1556,834],"ocr_ids":[8,9,10,11],"font":"微软雅黑","bold":false,"color":"#FFFFFF","polarity":"light","align":"left","container":"card:c4","effects":{"glow":false,"shadow":false},"line_mode":"normal","notes":""},
    {"id":"t6","lines":["4"],"bbox":[1624,886,1648,912],"ocr_ids":[],"font":"Arial","bold":false,"color":"#FFFFFF","polarity":"light","align":"right","container":"bg","effects":{"glow":false,"shadow":false},"line_mode":"single","notes":"页码"}
  ],
  "calligraphy": [],
  "persons": [ {"id":"h1","name":"人物_女孩","bbox":[95,330,420,941],"notes":"侧身凭栏，手搭在栏杆上，栏杆与手交叠"} ],
  "objects": [],
  "panels": [
    {"id":"c1","name":"底板_卡片1","kind":"card","bbox":[1104,160,1594,302],"shape":"rounded_rect","fill":"translucent_dark","border":true,"notes":"左上角⭐图标属于底板"},
    {"id":"c2","name":"底板_卡片2","kind":"card","bbox":[1104,322,1594,506],"shape":"rounded_rect","fill":"translucent_dark","border":true,"notes":"左上角羽毛图标"},
    {"id":"c3","name":"底板_卡片3","kind":"card","bbox":[1104,516,1594,606],"shape":"rounded_rect","fill":"translucent_dark","border":true,"notes":"左侧花朵图标"},
    {"id":"c4","name":"底板_卡片4","kind":"card","bbox":[1104,624,1594,852],"shape":"rounded_rect","fill":"translucent_dark","border":true,"notes":"左上角✨图标"}
  ],
  "decor_keep_in_bg": [],
  "page_number": {"text":"4","bbox":[1624,886,1648,912],"color":"#FFFFFF","font":"Arial"}
}
```

## 5. 自检清单（交付前逐条过）
1. 画面上每一个字都被某个 text_zone 或 calligraphy 覆盖了吗？（页码、表头、标签、选项字母、括号里的空白不算字）
2. 每个 zone 的行是否左对齐或居中对齐一致？缩进不同的行拆开了吗？
3. zone bbox 有没有压到卡片边框/表格线/色块边缘？
4. 字体、粗细、颜色、极性是否逐个看过放大图？
5. 书法判定是否有把握？不确定写 notes。
6. 人物/物品/底板 bbox 是否完整包住？
7. 装饰件是否都列进 decor_keep_in_bg（提醒程序不要擦）？
