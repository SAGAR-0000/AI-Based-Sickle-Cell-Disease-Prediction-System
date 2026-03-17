"""
Evaluation utilities for Sickle Cell Disease classification model.
Reports accuracy, F1-score, AUC-ROC, precision, and recall.
"""
from typing import Any

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,  # type: ignore[type-arg]
    device: torch.device,
    class_names: list[str],
) -> dict[str, Any]:
    """
    Run inference on the dataloader and return evaluation metrics.

    Args:
        model:       Trained PyTorch model in eval mode.
        dataloader:  Validation/test DataLoader.
        device:      Torch device (cpu or cuda).
        class_names: List of class name strings in label order.

    Returns:
        Dictionary with keys: accuracy, f1, auc_roc, precision, recall, report.
    """
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[float] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs: torch.Tensor = model(images)
            probs: torch.Tensor = torch.softmax(outputs, dim=1)
            preds: torch.Tensor = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            # Probability for the positive class (index 1)
            all_probs.extend(probs[:, 1].cpu().tolist())

    accuracy: float = accuracy_score(all_labels, all_preds)
    f1: float = f1_score(all_labels, all_preds, average="weighted")
    precision: float = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall: float = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    auc: float = roc_auc_score(all_labels, all_probs)
    report: str = classification_report(all_labels, all_preds, target_names=class_names)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc_roc": auc,
        "report": report,
    }
