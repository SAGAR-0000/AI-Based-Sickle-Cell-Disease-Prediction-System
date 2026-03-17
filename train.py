"""
Main training script for Sickle Cell Disease classification.
Uses MobileNetV2 pretrained on ImageNet with two-phase transfer learning:
  Phase 1 — warm-up: only the custom head is trained (backbone frozen)
  Phase 2 — fine-tune: entire network trained end-to-end with a low LR

Usage:
  python train.py

Output:
  best_model.pth — model checkpoint with highest val F1
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from dataset_utils import (
    build_train_transform,
    build_val_transform,
    compute_class_weights,
)
from evaluate import evaluate_model
from model import build_mobilenet_v2, unfreeze_backbone


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_ROOT = Path("dataset/dataset")
TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "val"
CHECKPOINT_PATH = Path("best_model.pth")

IMAGE_SIZE = 224
NUM_CLASSES = 2
WARMUP_EPOCHS = 5
FINETUNE_EPOCHS = 25
BATCH_SIZE = 16
WARMUP_LR = 1e-3
FINETUNE_LR = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.4
PATIENCE = 7          # early-stopping patience (epochs without val F1 improvement)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_weighted_sampler(dataset: ImageFolder) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler so every training batch is class-balanced.

    Args:
        dataset: ImageFolder training dataset.

    Returns:
        WeightedRandomSampler instance.
    """
    class_counts: list[int] = [0] * NUM_CLASSES
    for _, label in dataset.samples:
        class_counts[label] += 1

    sample_weights: list[float] = [
        1.0 / class_counts[label] for _, label in dataset.samples
    ]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def _train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,  # type: ignore[type-arg]
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Run one training epoch and return average loss.

    Args:
        model:      PyTorch model in train mode.
        dataloader: Training DataLoader.
        criterion:  Loss function.
        optimizer:  Optimizer.
        device:     Torch device.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss: float = 0.0

    for images, labels in tqdm(dataloader, desc="  Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs: torch.Tensor = model(images)
        loss: torch.Tensor = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def _run_phase(
    phase_name: str,
    model: nn.Module,
    train_loader: DataLoader,  # type: ignore[type-arg]
    val_loader: DataLoader,  # type: ignore[type-arg]
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    class_names: list[str],
    best_f1: float,
) -> tuple[nn.Module, float]:
    """
    Run a training phase (warm-up or fine-tune) with early stopping.

    Args:
        phase_name:   Human-readable label for logging.
        model:        PyTorch model.
        train_loader: Training DataLoader.
        val_loader:   Validation DataLoader.
        criterion:    Loss function.
        optimizer:    Optimizer.
        epochs:       Maximum epochs for this phase.
        device:       Torch device.
        class_names:  List of class names.
        best_f1:      Best F1 seen so far (for cross-phase checkpoint tracking).

    Returns:
        (model, best_f1) — model with best weights loaded, updated best F1.
    """
    patience_counter: int = 0

    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate_model(model, val_loader, device, class_names)
        val_f1: float = metrics["f1"]

        print(
            f"[{phase_name}] Epoch {epoch:02d}/{epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"Val Acc: {metrics['accuracy']:.4f} | "
            f"AUC: {metrics['auc_roc']:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  ✓ Saved best model (F1={best_f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping triggered after {epoch} epochs.")
                break

    # Restore best weights before returning
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    return model, best_f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Datasets ---
    train_dataset = ImageFolder(
        root=str(TRAIN_DIR),
        transform=build_train_transform(IMAGE_SIZE),
    )
    val_dataset = ImageFolder(
        root=str(VAL_DIR),
        transform=build_val_transform(IMAGE_SIZE),
    )
    class_names: list[str] = train_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # --- Class weights for loss ---
    raw_weights = compute_class_weights(str(TRAIN_DIR))
    weight_tensor = torch.tensor(
        [raw_weights[cls] for cls in class_names], dtype=torch.float
    ).to(device)
    print(f"Class weights: { {cls: f'{raw_weights[cls]:.3f}' for cls in class_names} }")

    # --- Dataloaders ---
    sampler = _build_weighted_sampler(train_dataset)
    train_loader: DataLoader = DataLoader(  # type: ignore[type-arg]
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )
    val_loader: DataLoader = DataLoader(  # type: ignore[type-arg]
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # --- Loss ---
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # ================================================================
    # PHASE 1: Warm-up — train only the new head, backbone frozen
    # ================================================================
    print("\n=== Phase 1: Warm-up (backbone frozen) ===")
    model = build_mobilenet_v2(
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
        freeze_backbone=True,
    ).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=WARMUP_LR,
        weight_decay=WEIGHT_DECAY,
    )

    model, best_f1 = _run_phase(
        phase_name="Warm-up",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=WARMUP_EPOCHS,
        device=device,
        class_names=class_names,
        best_f1=0.0,
    )

    # ================================================================
    # PHASE 2: Fine-tune — unfreeze backbone, low LR end-to-end
    # ================================================================
    print("\n=== Phase 2: Fine-tuning (full network) ===")
    unfreeze_backbone(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=FINETUNE_LR,
        weight_decay=WEIGHT_DECAY,
    )

    model, best_f1 = _run_phase(
        phase_name="Fine-tune",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=FINETUNE_EPOCHS,
        device=device,
        class_names=class_names,
        best_f1=best_f1,
    )

    # ================================================================
    # Final evaluation
    # ================================================================
    print(f"\n=== Final Evaluation (best model, F1={best_f1:.4f}) ===")
    final_metrics = evaluate_model(model, val_loader, device, class_names)
    print(final_metrics["report"])
    print(f"Checkpoint saved to: {CHECKPOINT_PATH.resolve()}")


if __name__ == "__main__":
    main()
