import timm

from .config import BackboneSpec


def build_model(backbone: BackboneSpec, num_classes: int):
    return timm.create_model(
        backbone.timm_name,
        pretrained=True,
        num_classes=num_classes,
    )


def freeze_backbone(model):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True
