# -*- coding: utf-8 -*-
"""SAM2 框提示分割——"Canva 架构"第一步：几何交给分割模型，VLM 只管语义。

- sam_alpha(img_rgb, bbox) -> 全图 float alpha 或 None
实现要点：ultralytics 的 bboxes 提示在非方形整图上存在坐标缩放 bug（y 按 x 比例压缩，
掩膜系统性偏移）——因此这里先把目标区域外扩裁出，再补边成**正方形** crop 喂给 SAM
（方形下 x/y 缩放一致，绕开该 bug），掩膜再贴回原图。附带收益：小图推理更快。
权重 models/sam2.1_t.pt（约 40MB，首次自动下载）；默认 CPU（MPS 数值不稳），可用
SLIDELIFT_SAM_DEVICE=mps 覆盖。
"""
import os
import numpy as np
import cv2

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_model = None


def _sam():
    global _model
    if _model is None:
        from ultralytics import SAM
        _model = SAM(os.path.join(_ROOT, "models", "sam2.1_t.pt"))
    return _model


def sam_alpha(img_rgb, bbox, expand=0.25):
    """框提示 SAM2 → 全图 float32 alpha（轻羽化）。拿不到掩膜返回 None。"""
    H, W = img_rgb.shape[:2]
    x0, y0, x1, y1 = [float(v) for v in bbox]
    bw, bh = max(8.0, x1 - x0), max(8.0, y1 - y0)
    pad = expand * max(bw, bh)
    cx0, cy0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
    cx1, cy1 = int(min(W, x1 + pad)), int(min(H, y1 + pad))
    crop = img_rgb[cy0:cy1, cx0:cx1]
    ch, cw = crop.shape[:2]
    if ch < 12 or cw < 12:
        return None
    side = max(ch, cw)                                  # 补边成正方形，绕开非方形坐标 bug
    sq = np.zeros((side, side, 3), np.uint8)
    sq[:ch, :cw] = crop
    # 目标框在方形 crop 内的坐标
    b = [x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0]
    b = [max(0.0, b[0]), max(0.0, b[1]), min(float(cw), b[2]), min(float(ch), b[3])]
    dev = os.environ.get("SLIDELIFT_SAM_DEVICE") or "cpu"
    try:
        r = _sam()(sq, bboxes=[b], device=dev, verbose=False)[0]
        if r.masks is None or len(r.masks.data) == 0:
            return None
        m = r.masks.data.cpu().numpy().astype(np.float32).max(axis=0)
        if m.shape != (side, side):
            m = cv2.resize(m, (side, side), interpolation=cv2.INTER_LINEAR)
    except Exception as e:
        print(f"SAM({dev}) 失败: {str(e)[:120]}", flush=True)
        return None
    full = np.zeros((H, W), np.float32)
    full[cy0:cy1, cx0:cx1] = m[:ch, :cw]
    return np.clip(cv2.GaussianBlur(full, (5, 5), 0), 0, 1)
