"""
fix_dataset_split.py

Detects and fixes data leakage between train/ and val/ splits.

Steps:
  1. Hash every image in train/ and val/ — report overlapping files.
  2. Collect all unique images from the root source folders
     (Positive/ and Labelled/) using their MD5 hashes.
  3. Re-create clean train/ and val/ directories with a proper
     stratified 80/20 split (no overlap guaranteed).

Source folders  (originals):
  dataset/dataset/Positive/   → class "Positive"
  dataset/dataset/Labelled/   → class "Negative"

Output folders  (rebuilt):
  dataset/dataset/train/Positive/
  dataset/dataset/train/Negative/
  dataset/dataset/val/Positive/
  dataset/dataset/val/Negative/

Usage:
  python fix_dataset_split.py
  python fix_dataset_split.py --dry-run    # report only, no file changes
"""

import argparse
import hashlib
import random
import shutil
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_ROOT = Path("dataset/dataset")

SOURCE_CLASSES: dict[str, Path] = {
    "Positive": DATASET_ROOT / "Positive",
    "Negative": DATASET_ROOT / "Labelled",
}

TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR   = DATASET_ROOT / "val"

VAL_RATIO  = 0.20
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def md5_hash(path: Path) -> str:
    """Return hex MD5 digest of file content."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_hashes(directory: Path) -> dict[str, Path]:
    """
    Walk a directory and return {md5_hash: file_path} for every .jpg.

    Raises:
        FileNotFoundError: If directory does not exist.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    return {md5_hash(p): p for p in sorted(directory.rglob("*.jpg"))}


def detect_overlap(
    train_root: Path,
    val_root: Path,
) -> tuple[set[str], dict[str, Path], dict[str, Path]]:
    """
    Compare all images in train_root and val_root by MD5 hash.

    Returns:
        (overlap_hashes, train_map, val_map)
        overlap_hashes: set of hashes present in BOTH splits.
        train_map / val_map: {hash: path} for each split.
    """
    print("Hashing train images …")
    train_map: dict[str, Path] = {}
    for class_dir in sorted(train_root.iterdir()):
        if class_dir.is_dir():
            train_map.update(collect_hashes(class_dir))

    print("Hashing val images …")
    val_map: dict[str, Path] = {}
    for class_dir in sorted(val_root.iterdir()):
        if class_dir.is_dir():
            val_map.update(collect_hashes(class_dir))

    overlap = set(train_map) & set(val_map)
    return overlap, train_map, val_map


def stratified_split(
    files: list[Path],
    val_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    """
    Randomly split file list into train/val with reproducible shuffling.

    Args:
        files:     List of file paths for one class.
        val_ratio: Fraction to use for validation.
        seed:      Random seed for reproducibility.

    Returns:
        (train_files, val_files)
    """
    rng = random.Random(seed)
    shuffled = files.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


def rebuild_split(
    source_classes: dict[str, Path],
    train_dir: Path,
    val_dir: Path,
    val_ratio: float,
    seed: int,
    dry_run: bool,
) -> None:
    """
    Remove existing train/val dirs and rebuild them from source_classes.

    Args:
        source_classes: {class_name: source_folder} mapping.
        train_dir:      Target train root directory.
        val_dir:        Target val root directory.
        val_ratio:      Fraction of each class to put in val.
        seed:           Random seed.
        dry_run:        If True, only print — do not touch the filesystem.
    """
    if not dry_run:
        for d in (train_dir, val_dir):
            if d.exists():
                shutil.rmtree(d)
                print(f"  Removed: {d}")

    for class_name, src_folder in source_classes.items():
        all_files = sorted(src_folder.glob("*.jpg"))
        if not all_files:
            print(f"  WARNING: No .jpg files found in {src_folder}")
            continue

        train_files, val_files = stratified_split(all_files, val_ratio, seed)

        for split_name, files, split_dir in [
            ("train", train_files, train_dir),
            ("val",   val_files,   val_dir),
        ]:
            dest_dir = split_dir / class_name
            print(
                f"  {split_name}/{class_name}: {len(files)} images"
                f"{' (dry run)' if dry_run else ''}"
            )
            if not dry_run:
                dest_dir.mkdir(parents=True, exist_ok=True)
                for src in files:
                    shutil.copy2(src, dest_dir / src.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect and fix train/val data leakage."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report overlap and planned split without modifying files.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Step 1: Detect existing overlap
    # ------------------------------------------------------------------
    print("\n── Step 1: Overlap detection ──────────────────────────────────")
    if TRAIN_DIR.exists() and VAL_DIR.exists():
        overlap, train_map, val_map = detect_overlap(TRAIN_DIR, VAL_DIR)
        print(f"  Train images : {len(train_map)}")
        print(f"  Val images   : {len(val_map)}")
        print(f"  Overlapping  : {len(overlap)}")

        if overlap:
            print("  ⚠  DATA LEAKAGE DETECTED — overlapping images:")
            for h in sorted(overlap)[:10]:  # show first 10
                print(f"     train: {train_map[h].name}  ↔  val: {val_map[h].name}")
            if len(overlap) > 10:
                print(f"     … and {len(overlap) - 10} more.")
        else:
            print("  ✓  No overlap found — train/val splits are clean.")
    else:
        print("  train/ or val/ does not exist yet — skipping overlap check.")

    # ------------------------------------------------------------------
    # Step 2: Verify source folders
    # ------------------------------------------------------------------
    print("\n── Step 2: Source folder check ────────────────────────────────")
    for cls, folder in SOURCE_CLASSES.items():
        count = len(list(folder.glob("*.jpg"))) if folder.exists() else 0
        status = "✓" if count > 0 else "✗ MISSING"
        print(f"  {status}  {cls}: {count} images  ({folder})")

    missing = [k for k, v in SOURCE_CLASSES.items() if not list(v.glob("*.jpg"))]
    if missing:
        raise FileNotFoundError(
            f"Source folders are empty or missing for classes: {missing}. "
            "Cannot rebuild split."
        )

    # ------------------------------------------------------------------
    # Step 3: Rebuild clean split
    # ------------------------------------------------------------------
    print(f"\n── Step 3: Rebuild split (val_ratio={VAL_RATIO}, seed={RANDOM_SEED}) ──")
    rebuild_split(
        source_classes=SOURCE_CLASSES,
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        val_ratio=VAL_RATIO,
        seed=RANDOM_SEED,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        print("\n── Done ────────────────────────────────────────────────────────")
        print(f"  ✓  Clean train/val split written to {DATASET_ROOT.resolve()}")
        print("  Run your training script normally now.")
    else:
        print("\n  Dry-run complete. No files were modified.")


if __name__ == "__main__":
    main()
