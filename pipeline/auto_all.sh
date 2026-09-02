#!/bin/zsh
# 全自动一条龙：OCR → AI 起草（跳过已有标注的页）→ 整卷生产（擦除/图层/标定/组装/QA）。
# 用法: pipeline/auto_all.sh <deck>   —— 傻瓜模式的后端：上传后一个任务跑完全部。
set -e; setopt shwordsplit
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT/pipeline"
DECK="$1"; PY="$ROOT/.venv/bin/python"
echo "[ocr]"
(cd "$ROOT" && $PY pipeline/ocr_pages.py "$DECK")   # ocr_pages 用相对路径，必须从项目根跑
[ -s "$ROOT/work/$DECK/ocr/p01.json" ] || { echo "OCR FAILED（没有产出 ocr/p01.json）"; exit 1; }
N=$(ls "$ROOT/work/$DECK/src"/p*.png | wc -l | tr -d ' ')
echo "[draft]"
$PY ai_draft.py "$DECK" $(seq 1 $N) || { echo "DRAFT FAILED（全部页起草失败，请检查设置里的 AI 配置）"; exit 1; }
SPECN=$(ls "$ROOT/work/$DECK/spec"/p*.json 2>/dev/null | wc -l | tr -d ' ')
[ "$SPECN" -ge 1 ] || { echo "DRAFT FAILED（没有产出任何标注）"; exit 1; }
exec zsh "$ROOT/pipeline/run_all.sh" "$DECK"
