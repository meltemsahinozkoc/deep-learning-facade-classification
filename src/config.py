from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict
import os

DATA_ROOT = Path(os.getenv("DATA_ROOT", "data/Building_characteristics"))


@dataclass(frozen=True)
class TaskSpec:
    name: str
    train_dir: Path
    val_dir: Path
    test_dir: Path
    extra_test_dirs: Dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    timm_name: str
    batch_size: int
    head_lr: float
    finetune_lr: float


_CLADDING_LON = DATA_ROOT / "Exterior Cladding Material/London/Before Augmentation/Data"
_CLADDING_SCO = DATA_ROOT / "Exterior Cladding Material/Scotland/Before Augmentation/Data"
_STORIES = DATA_ROOT / "Image Augmentation/Number of Stories/Data"

TASKS: Dict[str, TaskSpec] = {
    "cladding": TaskSpec(
        name="cladding",
        train_dir=_CLADDING_LON / "train",
        val_dir=_CLADDING_LON / "val",
        test_dir=_CLADDING_LON / "test",
        extra_test_dirs={"scotland": _CLADDING_SCO / "test"},
    ),
    "stories": TaskSpec(
        name="stories",
        train_dir=_STORIES / "train" / "Raw Data",
        val_dir=_STORIES / "val",
        test_dir=_STORIES / "test",
    ),
}

BACKBONES: Dict[str, BackboneSpec] = {
    "resnet50": BackboneSpec(
        "resnet50", "resnet50.a1_in1k", batch_size=32,
        head_lr=1e-3, finetune_lr=1e-4,
    ),
    "effnetv2s": BackboneSpec(
        "effnetv2s", "tf_efficientnetv2_s.in21k_ft_in1k", batch_size=32,
        head_lr=1e-3, finetune_lr=1e-4,
    ),
    "convnext_tiny": BackboneSpec(
        "convnext_tiny", "convnext_tiny.fb_in22k_ft_in1k", batch_size=32,
        head_lr=1e-3, finetune_lr=1e-4,
    ),
    "vit_b16": BackboneSpec(
        "vit_b16", "vit_base_patch16_224.augreg2_in21k_ft_in1k", batch_size=16,
        head_lr=1e-3, finetune_lr=3e-5,
    ),
}

IMG_SIZE = 224
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
HEAD_EPOCHS = 5
FINETUNE_EPOCHS = 15
WEIGHT_DECAY = 1e-4
SEED = 42
