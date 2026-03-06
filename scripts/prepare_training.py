"""Prepare expanded training dataset by merging sort_keep + sort_maybe.

Creates symlinks (or copies on Windows) from both directories into a single
training directory. Only includes images that have matching .txt captions.

Usage:
    python scripts/prepare_training.py --persona ann
    python scripts/prepare_training.py --persona ann --copy  # force copy instead of symlink
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

log = logging.getLogger(__name__)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def prepare_dataset(persona: str, force_copy: bool = False) -> dict:
    """Merge sort_keep + sort_maybe into a combined training directory."""
    project_root = Path(__file__).parent.parent
    base = project_root / "persona" / "training" / persona

    keep_dir = base / "sort_keep"
    maybe_dir = base / "sort_maybe"
    combined_dir = base / "combined"

    if not keep_dir.exists():
        log.error("sort_keep not found: %s", keep_dir)
        return {"error": "sort_keep not found"}

    combined_dir.mkdir(parents=True, exist_ok=True)

    stats = {"keep": 0, "maybe": 0, "skipped_no_caption": 0, "errors": 0}

    for source_dir, label in [(keep_dir, "keep"), (maybe_dir, "maybe")]:
        if not source_dir.exists():
            log.warning("%s not found, skipping", source_dir)
            continue

        for img_path in sorted(source_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue

            caption_path = img_path.with_suffix(".txt")
            if not caption_path.exists():
                stats["skipped_no_caption"] += 1
                continue

            # Prefix filename with source to avoid collisions
            prefix = "k_" if label == "keep" else "m_"
            dest_img = combined_dir / f"{prefix}{img_path.name}"
            dest_txt = combined_dir / f"{prefix}{img_path.stem}.txt"

            try:
                if not dest_img.exists():
                    if force_copy or sys.platform == "win32":
                        shutil.copy2(img_path, dest_img)
                        shutil.copy2(caption_path, dest_txt)
                    else:
                        os.symlink(img_path.resolve(), dest_img)
                        os.symlink(caption_path.resolve(), dest_txt)
                stats[label] += 1
            except Exception as e:
                log.warning("Error copying %s: %s", img_path.name, e)
                stats["errors"] += 1

    total = stats["keep"] + stats["maybe"]
    log.info("Combined dataset: %d images (%d keep + %d maybe), %d skipped, %d errors",
             total, stats["keep"], stats["maybe"], stats["skipped_no_caption"], stats["errors"])
    log.info("Output: %s", combined_dir)

    return {**stats, "total": total, "output_dir": str(combined_dir)}


def main():
    parser = argparse.ArgumentParser(description="Prepare combined LoRA training dataset")
    parser.add_argument("--persona", required=True, help="Persona name")
    parser.add_argument("--copy", action="store_true", help="Force copy instead of symlink")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = prepare_dataset(args.persona, force_copy=args.copy)

    if result.get("error"):
        print(f"ERROR: {result['error']}")
        sys.exit(1)

    print(f"\nDataset ready: {result['total']} paired images in {result['output_dir']}")
    print(f"  sort_keep: {result['keep']}")
    print(f"  sort_maybe: {result['maybe']}")
    if result["skipped_no_caption"]:
        print(f"  Skipped (no caption): {result['skipped_no_caption']}")


if __name__ == "__main__":
    main()
