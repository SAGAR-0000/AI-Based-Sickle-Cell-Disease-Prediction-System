"""
Dataset utilities for Sickle Cell Disease prediction.
Provides augmentation transforms and class weight computation.
"""
from pathlib import Path

from torchvision import transforms


def compute_class_weights(dataset_root: str) -> dict[str, float]:
    """
    Compute inverse-frequency class weights from a folder of class subdirectories.

    Args:
        dataset_root: Path to a directory containing one subdirectory per class.

    Returns:
        Dictionary mapping class folder name to its inverse-frequency weight.

    Raises:
        ValueError: If dataset_root has no class subdirectories.
    """
    root = Path(dataset_root)
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])

    if not class_dirs:
        raise ValueError(f"No class subdirectories found in: {dataset_root}")

    counts: dict[str, int] = {d.name: len(list(d.glob("*.jpg"))) for d in class_dirs}
    total: int = sum(counts.values())
    weights: dict[str, float] = {
        cls: total / count for cls, count in counts.items()
    }
    return weights


def build_train_transform(image_size: int) -> transforms.Compose:
    """
    Build augmentation transform for the training split.

    Args:
        image_size: Target square image size (e.g. 224).

    Returns:
        Composed torchvision transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_val_transform(image_size: int) -> transforms.Compose:
    """
    Build deterministic transform for the validation split (no augmentation).

    Args:
        image_size: Target square image size (e.g. 224).

    Returns:
        Composed torchvision transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
