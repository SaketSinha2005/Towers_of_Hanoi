"""
models/train.py — Optimized Training Loop for Mac M3

Per-epoch console output:
  ┌─────────────────────────────────────────────────────────┐
  │ Epoch  5/100                                            │
  │  Loss      : train=0.4231   val=0.3987                 │
  │  Accuracy  : train=82.34%   val=84.12%                 │
  │  Dice (val): WT=0.8612  TC=0.7843  ET=0.6921           │
  │              Mean=0.7792  ★ New best!                   │
  │  LR        : 2.40e-04                                   │
  │  Time      : 12.4m elapsed  |  No-improve: 0/10        │
  └─────────────────────────────────────────────────────────┘

TensorBoard scalars logged:
  Loss/train, Loss/val
  Accuracy/train, Accuracy/val
  Dice/WT, Dice/TC, Dice/ET, Dice/Mean
  LR
"""

import os
import sys
import time
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models.unet import build_model
from utils.dataloader import get_dataloaders
from utils.metrics import CombinedLoss, MetricMeter, evaluate_brats


BUDGET_SECONDS = 12 * 3600 - 300   # 12h hard budget


# ─── Utilities ────────────────────────────────────────────────────────────────
def mps_clear(device):
    if device.type == "mps":
        torch.mps.empty_cache()


def pixel_accuracy(preds: np.ndarray, gts: np.ndarray) -> float:
    """
    Simple pixel-level accuracy: fraction of correctly classified pixels.
    preds, gts : numpy arrays of any shape with integer class labels.
    """
    return float((preds == gts).sum()) / float(preds.size + 1e-8)


# ─── Checkpoint ───────────────────────────────────────────────────────────────
def save_checkpoint(model, optimizer, scheduler, epoch, best_dice, path):
    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(
        {
            "epoch":      epoch,
            "model":      cpu_state,
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict() if scheduler else None,
            "best_dice":  best_dice,
        },
        path,
    )
    print(f"  [Checkpoint] Saved → {path}")


def load_checkpoint(model, optimizer, path, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception:
            print("  [Checkpoint] Optimizer mismatch — skipping optimizer state.")
    if scheduler and ckpt.get("scheduler"):
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception:
            pass
    return ckpt.get("epoch", 0), ckpt.get("best_dice", 0.0)


# ─── Scheduler ────────────────────────────────────────────────────────────────
def build_scheduler(optimizer, train_loader, num_epochs):
    steps_per_epoch = len(train_loader) // config.GRAD_ACCUM_STEPS
    total_steps     = steps_per_epoch * num_epochs
    return optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr           = config.ONECYCLE_MAX_LR,
        total_steps      = total_steps,
        pct_start        = config.ONECYCLE_PCT_START,
        div_factor       = config.ONECYCLE_DIV,
        final_div_factor = config.ONECYCLE_FINAL_DIV,
        anneal_strategy  = "cos",
    )


# ─── Train Epoch ──────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, loss_fn, device, epoch):
    model.train()
    meter  = MetricMeter()
    accum  = config.GRAD_ACCUM_STEPS
    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(tqdm(loader, desc=f"  Train ep{epoch+1}", leave=False)):
        images = batch["image"].to(device)
        masks  = batch["mask"].to(device)

        # ensure masks are [B, H, W] long
        if masks.dim() == 4:
            masks = masks.squeeze(1)
        if masks.dim() == 4:
            masks = masks.argmax(dim=1)

        logits = model(images)
        loss   = loss_fn(logits, masks) / accum
        loss.backward()

        is_update = ((i + 1) % accum == 0) or (i + 1 == len(loader))
        if is_update:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            preds = logits.argmax(dim=1).cpu().numpy()
            gts   = masks.cpu().numpy()

            # pixel accuracy for this batch
            acc = pixel_accuracy(preds, gts)

            # BraTS dice for first sample in batch
            bm  = evaluate_brats(preds[0], gts[0])

            flat = {
                "loss": loss.item() * accum,
                "accuracy": acc,
            }
            for region, mvals in bm.items():
                flat[f"{region}_dice"] = mvals["dice"]

            meter.update(flat)

    return meter.mean()


# ─── Validation Epoch ─────────────────────────────────────────────────────────
@torch.no_grad()
def val_epoch(model, loader, loss_fn, device):
    model.eval()
    meter = MetricMeter()

    for batch in tqdm(loader, desc="  Val  ", leave=False):
        images = batch["image"].to(device)
        masks  = batch["mask"].to(device)

        # ensure masks are [B, H, W] long
        if masks.dim() == 4:
            masks = masks.squeeze(1)
        if masks.dim() == 4:
            masks = masks.argmax(dim=1)

        logits = model(images)
        loss   = loss_fn(logits, masks)

        preds = logits.argmax(dim=1).cpu().numpy()
        gts   = masks.cpu().numpy()

        # pixel accuracy for whole batch
        acc = pixel_accuracy(preds, gts)

        flat = {
            "loss":     loss.item(),
            "accuracy": acc,
        }

        # BraTS metrics averaged over batch
        for i in range(len(preds)):
            bm = evaluate_brats(preds[i], gts[i])
            for region, mvals in bm.items():
                for metric, val in mvals.items():
                    key = f"{region}_{metric}"
                    flat[key] = flat.get(key, 0) + val / len(preds)

        meter.update(flat)

    mps_clear(device)
    return meter.mean()


# ─── Main Train Function ───────────────────────────────────────────────────────
def train(resume=None):
    device = torch.device(config.DEVICE)

    train_loader, val_loader, _, _ = get_dataloaders()

    model = build_model(config.MODEL_NAME).to(device)

    loss_fn = CombinedLoss(
        num_classes   = config.NUM_CLASSES,
        class_weights = config.CLASS_WEIGHTS,
        dice_w        = config.DICE_WEIGHT,
        ce_w          = config.CE_WEIGHT,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr           = config.LEARNING_RATE,
        weight_decay = config.WEIGHT_DECAY,
    )

    scheduler = build_scheduler(optimizer, train_loader, config.NUM_EPOCHS)

    writer = SummaryWriter(log_dir=os.path.join(config.OUTPUT_DIR, "logs"))

    start_epoch  = 0
    best_dice    = 0.0
    no_improve   = 0
    train_start  = time.time()

    if resume and os.path.exists(resume):
        start_epoch, best_dice = load_checkpoint(model, optimizer, resume, scheduler)
        model = model.to(device)
        print(f"  [Resume] Loaded checkpoint — starting from epoch {start_epoch+1}")

    best_ckpt = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

    print(f"\n{'─'*62}")
    print(f"  Training: {config.MODEL_NAME.upper()}  |  "
          f"Max epochs: {config.NUM_EPOCHS}  |  "
          f"Early stop patience: {config.EARLY_STOP_PAT}")
    print(f"{'─'*62}\n")

    for epoch in range(start_epoch, config.NUM_EPOCHS):

        # ── Time-budget guard ────────────────────────────────────────────────
        if time.time() - train_start > BUDGET_SECONDS:
            print(f"\n[Train] 12-hour budget reached at epoch {epoch+1}. Stopping.")
            break

        # ── Train + Validate ─────────────────────────────────────────────────
        train_m = train_epoch(model, train_loader, optimizer, scheduler,
                              loss_fn, device, epoch)
        val_m   = val_epoch(model, val_loader, loss_fn, device)

        # ── Extract metrics ──────────────────────────────────────────────────
        train_loss = train_m.get("loss",     0.0)
        val_loss   = val_m.get("loss",       0.0)
        train_acc  = train_m.get("accuracy", 0.0) * 100
        val_acc    = val_m.get("accuracy",   0.0) * 100

        wt   = val_m.get("whole_tumor_dice", 0.0)
        tc   = val_m.get("tumor_core_dice",  0.0)
        et   = val_m.get("enhancing_dice",   0.0)
        mean = (wt + tc + et) / 3.0

        current_lr  = optimizer.param_groups[0]["lr"]
        elapsed_min = (time.time() - train_start) / 60
        is_best     = mean > best_dice

        # ── Console output ───────────────────────────────────────────────────
        print(f"┌{'─'*58}┐")
        print(f"│  Epoch {epoch+1:>3}/{config.NUM_EPOCHS:<3}  "
              f"                                         │")
        print(f"│  Loss      : train={train_loss:.4f}   val={val_loss:.4f}"
              f"                   │")
        print(f"│  Accuracy  : train={train_acc:6.2f}%  val={val_acc:6.2f}%"
              f"                 │")
        print(f"│  Dice (val): WT={wt:.4f}  TC={tc:.4f}  ET={et:.4f}  "
              f"          │")
        print(f"│              Mean={mean:.4f}"
              f"  {'★ New best!' if is_best else '          '}"
              f"                      │")
        print(f"│  LR        : {current_lr:.2e}"
              f"                                    │")
        print(f"│  Time      : {elapsed_min:.1f}m elapsed  "
              f"│  No-improve: {no_improve}/{config.EARLY_STOP_PAT}"
              f"             │")
        print(f"└{'─'*58}┘")

        # ── TensorBoard ──────────────────────────────────────────────────────
        writer.add_scalars("Loss",     {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc,  "val": val_acc},  epoch)
        writer.add_scalars("Dice",     {"WT": wt, "TC": tc, "ET": et,
                                        "Mean": mean},                          epoch)
        writer.add_scalar ("LR",       current_lr,                              epoch)

        # ── Early stopping + checkpoint ──────────────────────────────────────
        if is_best:
            best_dice  = mean
            no_improve = 0
            save_checkpoint(model, optimizer, scheduler, epoch, best_dice, best_ckpt)
        else:
            no_improve += 1
            if no_improve >= config.EARLY_STOP_PAT:
                print(f"\n  [EarlyStopping] No improvement for "
                      f"{config.EARLY_STOP_PAT} epochs. "
                      f"Best Mean Dice={best_dice:.4f}. Stopping.\n")
                break

    writer.close()

    total_min = (time.time() - train_start) / 60
    print(f"\n{'─'*62}")
    print(f"  Training complete in {total_min:.1f} minutes")
    print(f"  Best Mean Dice : {best_dice:.4f}")
    print(f"  Best checkpoint: {best_ckpt}")
    print(f"{'─'*62}\n")

    return best_ckpt


if __name__ == "__main__":
    train()