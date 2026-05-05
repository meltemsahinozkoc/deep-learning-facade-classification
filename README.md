# Building Façade Classification

Transfer-learning benchmark on the [Wang et al. 2024 façade dataset](https://doi.org/10.1016/j.dib.2024.110885), comparing four ImageNet-pretrained backbones (CNN + ViT) on two classification tasks.

## Tasks

| Task | Classes | Train / Val / Test |
|---|---|---|
| **Cladding material** (London → Scotland cross-domain) | Brick, Concrete, Curtain-Wall, Mixed, Others, Stone | 928 / 308 / 314 (+ 208 Scotland) |
| **Number of stories** | 1F, 2F, 3F, 4F, 5F+, Others | 418 / 138 / 144 |

## Backbones

ResNet50, EfficientNetV2-S, ConvNeXt-Tiny, ViT-B/16 — all from `timm`, fine-tuned with the same protocol:

1. Head-only training (5 epochs, AdamW, cosine schedule)
2. Full unfreeze fine-tune (15 epochs, lower LR)
3. Class-weighted cross-entropy, on-the-fly augmentation, mixed precision (CUDA), best-checkpoint by val accuracy

## Results

Run `notebooks/02_results.ipynb` to regenerate the table from `results/*.json`.

<!-- TABLE: filled in after training -->

## Setup

```bash
pip install -r requirements.txt
```

Place the dataset under `data/Building_characteristics/` (default — override with `DATA_ROOT=...`). Folder layout follows the Wang et al. release.

## Usage

```bash
# train one (task, backbone) pair
python -m src.cli train --task cladding --model vit_b16

# train everything
python scripts/run_all.py
```

Per-run artifacts:
- `models/<task>_<model>.pt` — best checkpoint (gitignored)
- `results/<task>_<model>.json` — metrics + history
- `results/<task>_<model>_<split>_cm.png` — confusion matrix

## Layout

```
src/        config, data, models, train, evaluate, cli
notebooks/  01_data_exploration, 02_results
scripts/    run_all.py
results/    per-run JSON + plots
```

## Hardware

Tested on RTX 2060 (6 GB) with mixed precision; all 8 runs complete in ~1–1.5 hours. Auto-detects `cuda` / `mps` / `cpu`.

## Reference

Wang, Y., Zhao, X. et al. *Building façade dataset for visual analysis.* Data in Brief, 2024.
