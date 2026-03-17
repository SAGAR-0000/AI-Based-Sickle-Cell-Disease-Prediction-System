"""
MobileNetV2-based model builder for Sickle Cell Disease classification.
Uses pretrained ImageNet weights and replaces the classifier head.
"""
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


def build_mobilenet_v2(num_classes: int, dropout: float, freeze_backbone: bool) -> nn.Module:
    """
    Build a MobileNetV2 model with a custom classification head.

    Strategy:
      - Phase 1 (freeze_backbone=True):  only the new head trains (fast warmup).
      - Phase 2 (freeze_backbone=False): entire network fine-tunes end-to-end.

    Args:
        num_classes:      Number of output classes (2 for binary: Positive/Negative).
        dropout:          Dropout probability for the classification head.
        freeze_backbone:  If True, all feature-extraction layers are frozen.

    Returns:
        nn.Module ready for training.
    """
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze or unfreeze the backbone (all layers except the classifier head)
    for param in model.features.parameters():
        param.requires_grad = not freeze_backbone

    # Replace the default classifier head
    in_features: int = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features=in_features, out_features=num_classes),
    )

    return model


def unfreeze_backbone(model: nn.Module) -> None:
    """
    Unfreeze all backbone parameters so the full network can be fine-tuned.

    Args:
        model: A MobileNetV2 model returned by build_mobilenet_v2.
    """
    for param in model.features.parameters():
        param.requires_grad = True
