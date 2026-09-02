#!/bin/zsh
# 只跑 标定→组装→QA（图层已就绪时用）。用法: pipeline/run_tail.sh <deck> [pages...]
# 标定只对"有 spec 的页"跑（支持页码不连续的部分标注 deck）；组装与 QA 永远全卷——
# 未标注页由 assemble.py 以原图整页占位，保证成品不缺页。
set -e; setopt shwordsplit
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT/pipeline"; DECK="$1"; shift; PY="$ROOT/.venv/bin/python"
if [ $# -eq 0 ]; then
  PAGES=$(ls "$ROOT/work/$DECK/spec"/p*.json 2>/dev/null | sed -E 's|.*/p0*([0-9]+)\.json|\1|' | sort -n)
else PAGES="$@"; fi
[ -n "$PAGES" ] || { echo "ERROR: 没有任何页面标注（spec），无法标定"; exit 1; }
mkdir -p "$ROOT/logs" "$ROOT/output"
echo "[calib]";   rm -f "$ROOT/work/$DECK/calib/params.json"; $PY calib_pages.py "$DECK" 5 $PAGES > "$ROOT/logs/_calib_full.log" 2>&1 || { echo "CALIB FAILED"; tail -5 "$ROOT/logs/_calib_full.log"; exit 1; }; grep -E "^iter|skip" "$ROOT/logs/_calib_full.log"
OUT="$ROOT/output/${DECK}_分层可编辑.pptx"
echo "[assemble]"; $PY assemble.py "$DECK" "$OUT" 2>&1 | tail -3
echo "[qa]";      $PY qa_pages.py "$DECK" "$OUT" > "$ROOT/logs/_qa_full.log" 2>&1 || { echo "QA FAILED"; tail -5 "$ROOT/logs/_qa_full.log"; }; grep -E "MAE" "$ROOT/logs/_qa_full.log" || true
echo "DONE -> $OUT"
