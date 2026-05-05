import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from .config import BACKBONES, FINETUNE_EPOCHS, HEAD_EPOCHS, SEED, TASKS
from .data import make_loaders
from .evaluate import evaluate_split, save_confusion_matrix
from .models import build_model
from .train import train_model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(task_key: str, model_key: str):
    set_seed(SEED)
    device = get_device()
    task = TASKS[task_key]
    backbone = BACKBONES[model_key]

    classes = sorted(d.name for d in task.train_dir.iterdir() if d.is_dir())
    print(f"=== {task_key} | {model_key} | device={device} | classes={classes}")

    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    ckpt = Path(f"models/{task_key}_{model_key}.pt")

    model = build_model(backbone, num_classes=len(classes)).to(device)
    loaders = make_loaders(task, model, backbone.batch_size)
    if loaders["classes"] != classes:
        raise ValueError("Class ordering mismatch")

    print(f"Train counts: {loaders['train_counts']}")

    history, best_val = train_model(
        model, loaders, backbone, device,
        HEAD_EPOCHS, FINETUNE_EPOCHS, ckpt,
    )

    model.load_state_dict(torch.load(ckpt, map_location=device))

    splits = {"test": evaluate_split(model, loaders["test"], classes, device)}
    for name, loader in loaders["extra_tests"].items():
        splits[name] = evaluate_split(model, loader, classes, device)

    results = {
        "task": task_key,
        "model": model_key,
        "timm_name": backbone.timm_name,
        "classes": classes,
        "best_val_acc": best_val,
        "history": history,
        "splits": splits,
    }

    out_json = Path(f"results/{task_key}_{model_key}.json")
    out_json.write_text(json.dumps(results, indent=2))

    for split_name, split in splits.items():
        save_confusion_matrix(
            split["confusion_matrix"], classes,
            f"{model_key} – {task_key} ({split_name}): acc={split['accuracy']:.2%}",
            Path(f"results/{task_key}_{model_key}_{split_name}_cm.png"),
        )

    print(f"=== done. test_acc={splits['test']['accuracy']:.4f} "
          f"macro_f1={splits['test']['macro_f1']:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(prog="src.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_p = sub.add_parser("train", help="Train one (task, backbone) pair")
    train_p.add_argument("--task", choices=list(TASKS), required=True)
    train_p.add_argument("--model", choices=list(BACKBONES), required=True)

    args = parser.parse_args()
    if args.cmd == "train":
        run(args.task, args.model)


if __name__ == "__main__":
    main()
