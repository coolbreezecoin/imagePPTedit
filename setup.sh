#!/bin/bash
# SlideLift skill 一次性安装：Python 环境 + 模型权重 + 渲染器检测
set -e
cd "$(dirname "$0")"

echo "== 1/4 Python 环境（锁定 3.12，依赖版本经过引擎调校）=="
if ! command -v uv >/dev/null 2>&1; then
  echo "   安装 uv（会自动管理 Python 版本）..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
[ -d .venv ] || uv venv .venv --python 3.12
uv pip install -q -r requirements.txt --python .venv/bin/python
echo "   依赖安装完成"

echo "== 2/4 模型权重（约 630MB，断点续传）=="
mkdir -p models weights/clip
DL="${SLIDELIFT_DL:-https://montage.ink/dl}"
fetch() {  # fetch <url> <目标路径> <最小字节数>
  if [ -f "$2" ] && [ "$(wc -c < "$2")" -ge "$3" ]; then echo "   已存在: $2"; return; fi
  echo "   下载: $2"
  curl -fL --retry 3 -C - -o "$2" "$1"
}
fetch "$DL/big-lama.pt"          models/big-lama.pt          200000000
fetch "$DL/sam2.1_t.pt"          models/sam2.1_t.pt          70000000
fetch "$DL/yolov8s-worldv2.pt"   models/yolov8s-worldv2.pt   20000000
fetch "$DL/ViT-B-32.pt"          weights/clip/ViT-B-32.pt    330000000
# ultralytics 的 CLIP 下载器有校验缺陷会反复重下——预置到相对路径并做软链
mkdir -p pipeline/weights && ln -sfn ../../weights/clip pipeline/weights/clip 2>/dev/null || true

echo "== 3/4 渲染器检测（质检对比图用）=="
if [ "$(uname)" = "Darwin" ] && osascript -e 'id of app "Microsoft PowerPoint"' >/dev/null 2>&1; then
  echo "   ✓ Microsoft PowerPoint（默认渲染器）"
elif command -v soffice >/dev/null 2>&1 || [ -x "/Applications/LibreOffice.app/Contents/MacOS/soffice" ]; then
  echo "   ✓ LibreOffice — 运行时请加环境变量 SLIDELIFT_RENDERER=libreoffice"
else
  echo "   ⚠ 未检测到 PowerPoint / LibreOffice：质检渲染将不可用。"
  echo "     Mac: brew install --cask libreoffice   Linux: apt install libreoffice"
fi

echo "== 4/4 自检 =="
.venv/bin/python - <<'PY'
import torch, cv2, numpy
from rapidocr_onnxruntime import RapidOCR
import pptx, fitz, rembg, ultralytics
print("   ✓ 全部依赖可用  torch", torch.__version__)
PY
echo "安装完成。用法见 SKILL.md。"
