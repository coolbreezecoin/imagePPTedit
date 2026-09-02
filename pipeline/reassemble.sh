#!/bin/zsh
# 秒级重组装：只跑 组装→QA（图层与标定都不动）——供审校台"编辑成品"（挪位/隐藏）后快速出新成品。
# 用法: pipeline/reassemble.sh <deck>
set -e; setopt shwordsplit
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT/pipeline"; DECK="$1"
PY="$ROOT/.venv/bin/python"
mkdir -p "$ROOT/logs" "$ROOT/output"
OUT="$ROOT/output/${DECK}_分层可编辑.pptx"
echo "[assemble]"; $PY assemble.py "$DECK" "$OUT" 2>&1 | tail -3
echo "[qa]";      $PY qa_pages.py "$DECK" "$OUT" > "$ROOT/logs/_qa_full.log" 2>&1 || { echo "QA FAILED"; tail -5 "$ROOT/logs/_qa_full.log"; }; grep -E "MAE" "$ROOT/logs/_qa_full.log" || true
echo "DONE -> $OUT"
