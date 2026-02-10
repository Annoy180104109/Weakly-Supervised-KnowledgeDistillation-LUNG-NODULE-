# Weakly/metrics_profile.py
from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn.functional as F

from scipy.ndimage import distance_transform_edt, binary_erosion


@torch.no_grad()
def binary_segmentation_stats_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    """
    logits : [B,1,H,W] (raw)
    targets: [B,1,H,W] with {0,1} (or bool)
    returns tp, fp, tn, fn as float tensors (on CPU)
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).to(torch.bool)
    t = targets.to(torch.bool)

    tp = (preds & t).sum().float().cpu()
    fp = (preds & (~t)).sum().float().cpu()
    tn = ((~preds) & (~t)).sum().float().cpu()
    fn = ((~preds) & t).sum().float().cpu()
    return tp, fp, tn, fn


def binary_metrics_from_stats(tp, fp, tn, fn, eps: float = 1e-7):
    acc = (tp + tn) / (tp + fp + tn + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    specificity = tn / (tn + fp + eps)
    return {
        "acc": acc.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "iou": iou.item(),
        "dice": dice.item(),
        "specificity": specificity.item(),
    }


# ------------------------- HD95 utilities -------------------------

def _surface(mask: np.ndarray) -> np.ndarray:
    """
    Extract 2D surface pixels of a binary mask.
    mask: bool array [H,W]
    """
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    if mask.sum() == 0:
        return mask
    er = binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), iterations=1, border_value=0)
    return mask ^ er  # surface = mask XOR eroded(mask)


def hd95_binary(pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0)):
    """
    95th percentile Hausdorff distance between binary masks (2D).
    Returns:
      float (pixels or physical units depending on spacing), or None if undefined (one empty, other not).
    Conventions:
      - both empty => 0.0
      - one empty  => None  (so you can exclude from averaging, like many papers do)
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    pred_sum = pred.sum()
    gt_sum = gt.sum()

    if pred_sum == 0 and gt_sum == 0:
        return 0.0
    if pred_sum == 0 or gt_sum == 0:
        return None

    spred = _surface(pred)
    sgt = _surface(gt)

    # Distance transform to the OTHER surface:
    # dist_to_sgt at each pixel gives distance to nearest pixel in sgt
    dist_to_sgt = distance_transform_edt(~sgt, sampling=spacing)
    dist_to_spred = distance_transform_edt(~spred, sampling=spacing)

    d1 = dist_to_sgt[spred]   # surface(pred) -> surface(gt)
    d2 = dist_to_spred[sgt]   # surface(gt)   -> surface(pred)

    if d1.size == 0 and d2.size == 0:
        return 0.0
    all_d = np.concatenate([d1, d2]) if d1.size and d2.size else (d1 if d1.size else d2)

    return float(np.percentile(all_d, 95))


@torch.no_grad()
def hd95_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, spacing=(1.0, 1.0)):
    """
    logits/targets: [B,1,H,W]
    Returns list of hd95 values (floats) for valid items; invalid (one empty) are skipped.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).to(torch.bool).detach().cpu().numpy()
    targs = targets.to(torch.bool).detach().cpu().numpy()

    vals = []
    B = preds.shape[0]
    for i in range(B):
        p = preds[i, 0]
        g = targs[i, 0]
        v = hd95_binary(p, g, spacing=spacing)
        if v is not None:
            vals.append(v)
    return vals


# ------------------------- evaluation -------------------------

@torch.no_grad()
def evaluate_binary_segmentation(model, dataloader, device, threshold: float = 0.5, spacing=(1.0, 1.0)):
    """
    Expects dataloader yielding (image, mask) where:
      image: [B,1,H,W], mask: [B,1,H,W]
    Returns:
      dict of scalar pixel metrics + hd95_mean/std/n
    """
    model.eval()
    tp = fp = tn = fn = 0.0

    hd95_vals = []

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)

        _tp, _fp, _tn, _fn = binary_segmentation_stats_from_logits(logits, y, threshold)
        tp += _tp.item(); fp += _fp.item(); tn += _tn.item(); fn += _fn.item()

        hd95_vals.extend(hd95_from_logits(logits, y, threshold=threshold, spacing=spacing))

    out = binary_metrics_from_stats(torch.tensor(tp), torch.tensor(fp), torch.tensor(tn), torch.tensor(fn))

    # HD95 summary (skip undefined cases)
    if len(hd95_vals) > 0:
        hd95_vals = np.asarray(hd95_vals, dtype=np.float32)
        out["hd95_mean"] = float(hd95_vals.mean())
        out["hd95_std"]  = float(hd95_vals.std(ddof=0))
        out["hd95_n"]    = int(hd95_vals.size)
    else:
        out["hd95_mean"] = None
        out["hd95_std"]  = None
        out["hd95_n"]    = 0

    return out


def estimate_gflops(model, input_shape=(1, 1, 256, 256), device="cuda"):
    """
    Tries common FLOPs profilers. Returns (gflops, params_million) or (None, params_million).
    """
    model = model.to(device).eval()
    x = torch.randn(*input_shape, device=device)

    params = sum(p.numel() for p in model.parameters()) / 1e6

    # Try thop
    try:
        from thop import profile  # pip install thop
        macs, _params = profile(model, inputs=(x,), verbose=False)
        # thop returns MACs; FLOPs ~ 2*MACs for conv/linear (rule of thumb).
        gflops = (2.0 * macs) / 1e9
        return gflops, params
    except Exception:
        pass

    # Try fvcore
    try:
        from fvcore.nn import FlopCountAnalysis  # pip install fvcore
        flops = FlopCountAnalysis(model, x).total()
        gflops = flops / 1e9
        return gflops, params
    except Exception:
        pass

    return None, params


@torch.no_grad()
# def benchmark_inference(model, input_shape=(1, 1, 256, 256), device="cuda", warmup=20, iters=100):
#     """
#     Returns: (ms_per_image, fps)
#     """
#     model = model.to(device).eval()
#     x = torch.randn(*input_shape, device=device)

#     # Warmup
#     for _ in range(warmup):
#         _ = model(x)
#     if device.startswith("cuda"):
#         torch.cuda.synchronize()

#     if device.startswith("cuda"):
#         starter = torch.cuda.Event(enable_timing=True)
#         ender = torch.cuda.Event(enable_timing=True)
#         starter.record()
#         for _ in range(iters):
#             _ = model(x)
#         ender.record()
#         torch.cuda.synchronize()
#         total_ms = starter.elapsed_time(ender)
#         ms_per_iter = total_ms / iters
#     else:
#         t0 = time.perf_counter()
#         for _ in range(iters):
#             _ = model(x)
#         t1 = time.perf_counter()
#         total_s = (t1 - t0)
#         ms_per_iter = (total_s * 1000.0) / iters

#     batch = input_shape[0]
#     ms_per_image = ms_per_iter / batch
#     fps = 1000.0 / ms_per_image
#     return ms_per_image, fps

@torch.no_grad()
def benchmark_inference(model, input_shape=(1, 1, 256, 256), device="cuda", warmup=20, iters=100):
    """
    Returns: (ms_per_image, fps)
    device can be str ("cuda"/"cpu") or torch.device
    """
    # normalize device
    dev = device if isinstance(device, torch.device) else torch.device(device)
    is_cuda = (dev.type == "cuda")

    model = model.to(dev).eval()
    x = torch.randn(*input_shape, device=dev)

    # Warmup
    for _ in range(warmup):
        _ = model(x)
    if is_cuda:
        torch.cuda.synchronize()

    if is_cuda:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        for _ in range(iters):
            _ = model(x)
        ender.record()
        torch.cuda.synchronize()
        total_ms = starter.elapsed_time(ender)
        ms_per_iter = total_ms / iters
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(x)
        t1 = time.perf_counter()
        ms_per_iter = ((t1 - t0) * 1000.0) / iters

    batch = input_shape[0]
    ms_per_image = ms_per_iter / batch
    fps = 1000.0 / ms_per_image
    return ms_per_image, fps






def measure_training_time(train_one_epoch_fn, epochs: int):
    """
    Wrap your existing train loop:
      - train_one_epoch_fn(epoch_idx) should run one epoch and return anything.
    Returns dict with per-epoch seconds and total.
    """
    per_epoch = []
    t_total0 = time.perf_counter()
    for e in range(epochs):
        t0 = time.perf_counter()
        train_one_epoch_fn(e)
        t1 = time.perf_counter()
        per_epoch.append(t1 - t0)
    t_total1 = time.perf_counter()
    return {"epoch_seconds": per_epoch, "total_seconds": (t_total1 - t_total0)}
