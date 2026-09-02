#!/bin/zsh
# 用法: pipeline/run_all.sh <deck> [pages...]   （缺省=全部"有 spec 的页"，支持页码不连续）
# 阶段：erase → layers → sheets → calib → assemble(全卷) → qa(全卷)
# 组装永远全卷：未标注页由 assemble.py 以原图整页占位，成品不缺页。
# 多页时 erase/layers 按页轮转分组并行（SLIDELIFT_PAR 路，默认 2）——页间完全独立，
# 每组一个进程各自加载模型；每组限 OMP 线程数防止互相抢核。
set -e
setopt shwordsplit
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT/pipeline"
DECK="$1"; shift
PY="$ROOT/.venv/bin/python"
if [ $# -eq 0 ]; then
  PAGES=$(ls "$ROOT/work/$DECK/spec"/p*.json 2>/dev/null | sed -E 's|.*/p0*([0-9]+)\.json|\1|' | sort -n)
else PAGES="$@"; fi
[ -n "$PAGES" ] || { echo "ERROR: 没有任何页面标注（spec）"; exit 1; }
mkdir -p "$ROOT/logs" "$ROOT/output"

pages_arr=(${=PAGES})
NP=${#pages_arr[@]}
PAR=${SLIDELIFT_PAR:-2}
W=$(( NP < PAR ? NP : PAR ))
CORES=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)

# 按页轮转分组并行跑一个阶段：$1=脚本名 $2=输出过滤正则
run_stage() {
  local script="$1" pat="$2"
  if [ "$W" -le 1 ]; then
    $PY "$script" "$DECK" $PAGES 2>&1 | grep -E "$pat" || true
    return 0
  fi
  local i j grp
  for ((i = 1; i <= W; i++)); do
    grp=""
    for ((j = i; j <= NP; j += W)); do grp="$grp ${pages_arr[j]}"; done
    OMP_NUM_THREADS=$(( CORES / W > 1 ? CORES / W : 1 )) \
      $PY "$script" "$DECK" ${=grp} 2>&1 | { grep -E "$pat" || true; } &
  done
  wait
  return 0
}

echo "[gate]";    $PY art_gate.py     "$DECK" $PAGES 2>&1 | grep -E "分级|done|rror|Trace" || true
echo "[erase]";  [ "$W" -gt 1 ] && echo "（$W 路并行）"
run_stage erase_pages.py  "done|rror|bad=[1-9]|POLWARN|lines [0-9]+/[0-9]+"
echo "[layers]"; [ "$W" -gt 1 ] && echo "（$W 路并行）"
run_stage layers_pages.py "done|rror|empty|failed"
echo "[sheets]";  $PY layer_sheet.py  "$DECK" $PAGES > /dev/null 2>&1 || true
exec zsh "$ROOT/pipeline/run_tail.sh" "$DECK"
