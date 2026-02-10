# Weakly/train_utils.py
# Train/eval utilities + logging + curves + checkpoints
# + Student profiling: train time, inference speed, and conv-only GFLOPs/params.

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------- Loss --------------------
def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    num = 2 * (prob * target).sum(dim=(1, 2, 3)) + smooth
    den = (prob + target).sum(dim=(1, 2, 3)) + smooth
    dice = 1 - (num / den).mean()
    return bce + dice


# -------------------- Metrics --------------------
@torch.no_grad()
def compute_metrics_from_logits(logits: torch.Tensor, y: torch.Tensor, thr: float = 0.5) -> Dict[str, float]:
    """
    Batch-mean metrics:
      - acc: pixel accuracy (TP+TN)/all
      - iou: foreground IoU
      - dice: foreground Dice
    """
    prob = torch.sigmoid(logits)
    pred = (prob > thr).float()

    acc = (pred == y).float().mean().item()

    inter = (pred * y).sum(dim=(1, 2, 3))
    union = (pred + y - pred * y).sum(dim=(1, 2, 3))
    iou = (inter / (union + 1e-6)).mean().item()

    dice = (2 * inter + 1e-6) / (pred.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3)) + 1e-6)
    dice = dice.mean().item()

    return {"acc": acc, "iou": iou, "dice": dice}

# for confidence filter 
@torch.no_grad()
def make_pseudo_and_mask_from_teacher(
    t_logits: torch.Tensor,
    tau_hi: float = 0.9,
    tau_lo: float = 0.1,
):
    """
    t_logits: (B,1,H,W)
    Returns:
      pseudo_y: (B,1,H,W) float {0,1}
      keep:     (B,1,H,W) float {0,1}  (1 = use in loss, 0 = ignore)
    """
    p = torch.sigmoid(t_logits)

    pseudo_y = (p >= 0.5).float()
    keep = (p >= tau_hi) | (p <= tau_lo)   # confident foreground OR confident background
    keep = keep.float()

    return pseudo_y, keep




def masked_bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    smooth: float = 1.0
) -> torch.Tensor:
    """
    logits/target/mask: (B,1,H,W)
    mask is 1 for valid pixels, 0 for ignore.
    """
    # ---- BCE (masked) ----
    bce_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce_map * mask).sum() / (mask.sum() + 1e-6)

    # ---- Dice (masked) ----
    prob = torch.sigmoid(logits)
    prob = prob * mask
    target = target * mask

    num = 2 * (prob * target).sum(dim=(1, 2, 3)) + smooth
    den = (prob + target).sum(dim=(1, 2, 3)) + smooth
    dice = 1 - (num / den).mean()

    return bce + dice




@torch.no_grad()
def eval_epoch(model: nn.Module, loader, device, amp: bool = True) -> Dict[str, float]:
    model.eval()
    tot = {"loss": 0.0, "acc": 0.0, "iou": 0.0, "dice": 0.0}
    n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp and (device.type == "cuda")):
            logits = model(x)
            loss = bce_dice_loss(logits, y)

        m = compute_metrics_from_logits(logits, y)
        bs = x.size(0)

        tot["loss"] += loss.item() * bs
        tot["acc"]  += m["acc"] * bs
        tot["iou"]  += m["iou"] * bs
        tot["dice"] += m["dice"] * bs
        n += bs

    for k in tot:
        tot[k] /= max(1, n)
    return tot


# -------------------- Train loop (history + checkpoints + time) --------------------
def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    out_dir: Path,
    tag: str,
    lr: float,
    max_epochs: int,
    patience: int,
    amp: bool = True,
) -> Tuple[nn.Module, List[Dict[str, float]]]:

    out_dir.mkdir(parents=True, exist_ok=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and (device.type == "cuda"))

    best_val = float("inf")
    best_state = None
    best_epoch = 0

    history: List[Dict[str, float]] = []

    t_train_start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        t_epoch_start = time.perf_counter()

        model.train()
        tot_loss = 0.0
        tot_acc = 0.0
        tot_iou = 0.0
        tot_dice = 0.0
        n = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(x)
                loss = bce_dice_loss(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            m = compute_metrics_from_logits(logits.detach(), y.detach())
            bs = x.size(0)
            tot_loss += loss.item() * bs
            tot_acc  += m["acc"] * bs
            tot_iou  += m["iou"] * bs
            tot_dice += m["dice"] * bs
            n += bs

        train_metrics = {
            "loss": tot_loss / max(1, n),
            "acc":  tot_acc / max(1, n),
            "iou":  tot_iou / max(1, n),
            "dice": tot_dice / max(1, n),
        }

        val_metrics = eval_epoch(model, val_loader, device, amp=amp)

        epoch_time = time.perf_counter() - t_epoch_start

        row = {
            "epoch": epoch,
            "epoch_time_sec": epoch_time,
            "train_loss": train_metrics["loss"],
            "train_acc":  train_metrics["acc"],
            "train_iou":  train_metrics["iou"],
            "train_dice": train_metrics["dice"],
            "val_loss":   val_metrics["loss"],
            "val_acc":    val_metrics["acc"],
            "val_iou":    val_metrics["iou"],
            "val_dice":   val_metrics["dice"],
        }
        history.append(row)

        print(
            f"[{tag}] ep {epoch:03d} | "
            f"tr loss {row['train_loss']:.4f} dice {row['train_dice']:.3f} acc {row['train_acc']:.3f} | "
            f"va loss {row['val_loss']:.4f} dice {row['val_dice']:.3f} acc {row['val_acc']:.3f} | "
            f"time {epoch_time:.1f}s"
        )

        # early stopping on val loss
        if row["val_loss"] < best_val - 1e-5:
            best_val = row["val_loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            torch.save(
                {"epoch": epoch, "model_state": best_state},
                out_dir / f"{tag}_best.pt"
            )

        if epoch - best_epoch >= patience:
            print(f"[{tag}] Early stop at ep {epoch}. Best val loss {best_val:.4f} @ ep {best_epoch}")
            break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # save last
    torch.save({"epoch": history[-1]["epoch"], "model_state": model.state_dict()}, out_dir / f"{tag}_last.pt")

    # total time
    total_train_time = time.perf_counter() - t_train_start
    with open(out_dir / f"{tag}_train_time.txt", "w") as f:
        f.write(f"total_train_time_sec,{total_train_time}\n")

    # save history CSV + curves
    save_history_csv(history, out_dir / f"{tag}_history.csv")
    plot_history(history, out_dir / f"{tag}_curves.png", title=tag)

    return model, history


def save_history_csv(history: List[Dict[str, float]], path: Path):
    if not history:
        return
    keys = list(history[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(history)


def plot_history(history: List[Dict[str, float]], out_png: Path, title: str = ""):
    if not history:
        return

    epochs = [h["epoch"] for h in history]

    plt.figure(figsize=(12, 8))

    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, [h["train_loss"] for h in history], label="train")
    plt.plot(epochs, [h["val_loss"] for h in history], label="val")
    plt.title("Loss"); plt.xlabel("epoch"); plt.legend()

    # Dice
    plt.subplot(2, 2, 2)
    plt.plot(epochs, [h["train_dice"] for h in history], label="train")
    plt.plot(epochs, [h["val_dice"] for h in history], label="val")
    plt.title("Dice"); plt.xlabel("epoch"); plt.legend()

    # IoU
    plt.subplot(2, 2, 3)
    plt.plot(epochs, [h["train_iou"] for h in history], label="train")
    plt.plot(epochs, [h["val_iou"] for h in history], label="val")
    plt.title("IoU"); plt.xlabel("epoch"); plt.legend()

    # Acc
    plt.subplot(2, 2, 4)
    plt.plot(epochs, [h["train_acc"] for h in history], label="train")
    plt.plot(epochs, [h["val_acc"] for h in history], label="val")
    plt.title("Pixel Acc"); plt.xlabel("epoch"); plt.legend()

    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# -------------------- Checkpoint loading --------------------
def load_checkpoint(model: nn.Module, ckpt_path: Path, map_location="cpu") -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    return model


# -------------------- Student profiling: GFLOPs & speed --------------------
@torch.no_grad()
def profile_conv_gflops(model: nn.Module, input_shape=(1, 1, 256, 256)) -> Dict[str, float]:
    """
    Rough GFLOPs for Conv2d/ConvTranspose2d only (ignores BN/ReLU/LN/attention).
    Best for CNN students like DWUNet.
    """
    model = model.to("cpu").eval()
    x = torch.zeros(*input_shape)

    flops = 0

    def conv2d_flops(h, w, in_ch, out_ch, k, groups=1):
        per_out = (k * k * in_ch // groups)
        return h * w * out_ch * per_out

    def hook(m, inp, out):
        nonlocal flops
        x0 = inp[0]
        _, Cin, _, _ = x0.shape
        _, Cout, Hout, Wout = out.shape
        kH, _ = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
        flops += conv2d_flops(Hout, Wout, Cin, Cout, kH, groups=m.groups)

    handles = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            handles.append(m.register_forward_hook(hook))

    model(x)
    for h in handles:
        h.remove()

    params = sum(p.numel() for p in model.parameters())
    gflops = flops / 1e9
    return {"params": float(params), "conv_flops": float(flops), "conv_gflops": float(gflops)}


@torch.no_grad()
def benchmark_inference(model: nn.Module, loader, device, amp=True, warmup=10, iters=50) -> Dict[str, float]:
    """
    Measures average forward-pass time using batches from loader.
    Returns ms/image and FPS. Uses CUDA sync when on GPU.
    """
    model.eval()
    it = iter(loader)

    # warmup
    for _ in range(warmup):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        x = batch[0].to(device, non_blocking=True)  # works for (x,y) or (x,path)
        with torch.cuda.amp.autocast(enabled=amp and (device.type == "cuda")):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    total_time = 0.0
    total_images = 0

    for _ in range(iters):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        x = batch[0].to(device, non_blocking=True)
        t0 = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=amp and (device.type == "cuda")):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        total_time += (t1 - t0)
        total_images += x.size(0)

    ms_per_img = (total_time / max(1, total_images)) * 1000.0
    fps = 1000.0 / ms_per_img if ms_per_img > 0 else 0.0
    return {
        "ms_per_img": float(ms_per_img),
        "fps": float(fps),
        "images": float(total_images),
        "seconds": float(total_time)
    }
