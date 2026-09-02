"""LaMa 修补封装（big-lama TorchScript）。修复 simple-lama 在 Mac 上 CUDA map_location 的问题；支持 MPS/CPU。"""
import os, numpy as np, torch, cv2

_MODEL = None
_DEVICE = None

def _model_path():
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.environ.get("LAMA_MODEL", os.path.join(here, "models", "big-lama.pt"))

def get_lama(device=None):
    global _MODEL, _DEVICE
    if _MODEL is None:
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        m = torch.jit.load(_model_path(), map_location="cpu")
        m.eval()
        try:
            m = m.to(device)
        except Exception:
            device = "cpu"; m = m.to(device)
        _MODEL, _DEVICE = m, device
    return _MODEL, _DEVICE

def _pad8(a, mode="reflect"):
    h, w = a.shape[:2]
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    if a.ndim == 3:
        return np.pad(a, ((0, ph), (0, pw), (0, 0)), mode=mode), (h, w)
    return np.pad(a, ((0, ph), (0, pw)), mode=mode), (h, w)

@torch.no_grad()
def lama_inpaint(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """img_rgb: HxWx3 uint8 (RGB). mask: HxW, >0 表示要修补。返回 HxWx3 uint8。"""
    model, device = get_lama()
    img = img_rgb.astype(np.float32) / 255.0
    m = (mask > 0).astype(np.float32)
    img_p, (h, w) = _pad8(img); m_p, _ = _pad8(m, mode="constant")
    it = torch.from_numpy(img_p).permute(2, 0, 1)[None].to(device)
    mt = torch.from_numpy(m_p)[None, None].to(device)
    out = model(it, mt)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    out = (out[:h, :w] * 255 + 0.5).astype(np.uint8)
    # 只替换掩膜区域，其他像素保持原样
    res = img_rgb.copy()
    mm = mask > 0
    res[mm] = out[mm]
    return res

def inpaint_region(img_rgb, mask, margin=256, max_side=1600):
    """在掩膜包围盒外扩 margin 的裁剪区域上跑 LaMa（更快更准），结果贴回整图。"""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return img_rgb.copy()
    H, W = mask.shape
    y0, y1 = max(0, ys.min() - margin), min(H, ys.max() + margin + 1)
    x0, x1 = max(0, xs.min() - margin), min(W, xs.max() + margin + 1)
    crop = img_rgb[y0:y1, x0:x1]; cm = mask[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    scale = 1.0
    if max(ch, cw) > max_side:
        scale = max_side / max(ch, cw)
        crop_s = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_AREA)
        cm_s = cv2.resize((cm > 0).astype(np.uint8), (crop_s.shape[1], crop_s.shape[0]), interpolation=cv2.INTER_NEAREST)
        out_s = lama_inpaint(crop_s, cm_s)
        out = cv2.resize(out_s, (cw, ch), interpolation=cv2.INTER_CUBIC)
        res = img_rgb.copy(); sub = res[y0:y1, x0:x1]; sub[cm > 0] = out[cm > 0]
        return res
    out = lama_inpaint(crop, cm)
    res = img_rgb.copy(); res[y0:y1, x0:x1] = out
    return res
