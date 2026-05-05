import time
from pathlib import Path

import torch
from torch import nn, optim

from .config import BackboneSpec, WEIGHT_DECAY
from .models import freeze_backbone, unfreeze_all


def _run_epoch(model, loader, criterion, optimizer, device, scaler, train: bool):
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(x)
                    loss = criterion(logits, y)
            else:
                logits = model(x)
                loss = criterion(logits, y)

        if train:
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def _train_stage(model, loaders, criterion, scaler, device, lr, epochs, stage_name,
                 ckpt_path, best_val_acc, history):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = _run_epoch(model, loaders["train"], criterion, optimizer, device, scaler, train=True)
        va_loss, va_acc = _run_epoch(model, loaders["val"], criterion, None, device, None, train=False)
        scheduler.step()
        dt = time.time() - t0

        history.append({
            "stage": stage_name, "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss, "val_acc": va_acc,
            "lr": optimizer.param_groups[0]["lr"], "time_s": dt,
        })

        improved = va_acc > best_val_acc
        if improved:
            best_val_acc = va_acc
            torch.save(model.state_dict(), ckpt_path)

        print(
            f"  [{stage_name} {epoch:>2}/{epochs}] "
            f"train_loss={tr_loss:.3f} acc={tr_acc:.3f} | "
            f"val_loss={va_loss:.3f} acc={va_acc:.3f} | "
            f"{dt:.1f}s{' *' if improved else ''}",
            flush=True,
        )

    return best_val_acc


def train_model(model, loaders, backbone: BackboneSpec, device, head_epochs, finetune_epochs, ckpt_path: Path):
    criterion = nn.CrossEntropyLoss(weight=loaders["class_weights"].to(device))
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    history = []
    best_val_acc = 0.0

    print("Stage 1: head only")
    freeze_backbone(model)
    best_val_acc = _train_stage(
        model, loaders, criterion, scaler, device,
        backbone.head_lr, head_epochs, "head",
        ckpt_path, best_val_acc, history,
    )

    print("Stage 2: fine-tune")
    unfreeze_all(model)
    best_val_acc = _train_stage(
        model, loaders, criterion, scaler, device,
        backbone.finetune_lr, finetune_epochs, "finetune",
        ckpt_path, best_val_acc, history,
    )

    return history, best_val_acc
