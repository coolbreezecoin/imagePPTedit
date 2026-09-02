#!/bin/zsh
# 用法: pipeline/rerun_pages.sh <deck> <pages...>
# 安全的"重跑选中页"：只对选中页重做 擦除→图层→拼版，然后 标定(全部有 spec 的页)→组装(全卷)→QA(全卷)。
set -e; setopt shwordsplit
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT/pipeline"
DECK="$1"; shift
PY="$ROOT/.venv/bin/python"
if [ $# -lt 1 ]; then echo "ERROR: 需要页码，如 rerun_pages.sh <deck> 21 24"; exit 1; fi
echo "[gate]";   $PY art_gate.py     "$DECK" "$@" 2>&1 | grep -E "分级|done|rror" || true
echo "[erase]";  $PY erase_pages.py  "$DECK" "$@"
echo "[layers]"; $PY layers_pages.py "$DECK" "$@"
echo "[sheets]"; $PY layer_sheet.py  "$DECK" "$@" > /dev/null 2>&1 || true
exec zsh "$ROOT/pipeline/run_tail.sh" "$DECK"
