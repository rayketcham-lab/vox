"""Curate and sort training images for LoRA training.

Stage 1: EXIF extraction + face detection + quality scoring + auto-sort
Stage 2: Vision-model captioning focused on subject's physical appearance

Usage:
    python scripts/curate_training_images.py --source "~/Downloads/training_photos"
    python scripts/curate_training_images.py --caption   # run vision captioning on curated images
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import struct
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ExifTags

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_BASE = PROJECT_ROOT / "persona" / "training" / "ann"
CURATED_DIR = TRAINING_BASE / "curated"
KEEP_DIR = TRAINING_BASE / "sort_keep"
MAYBE_DIR = TRAINING_BASE / "sort_maybe"
SKIP_DIR = TRAINING_BASE / "sort_skip"
METADATA_FILE = TRAINING_BASE / "metadata.csv"
REPORT_FILE = TRAINING_BASE / "curation_report.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# OpenCV face detector (Haar cascade — fast, good enough for sorting)
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
PROFILE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)


def extract_exif(img_path: Path) -> dict:
    """Extract useful EXIF metadata from an image."""
    meta = {
        "filename": img_path.name,
        "file_size_kb": img_path.stat().st_size // 1024,
        "file_date": datetime.fromtimestamp(img_path.stat().st_mtime).isoformat(),
    }
    try:
        img = Image.open(img_path)
        meta["width"] = img.width
        meta["height"] = img.height
        meta["megapixels"] = round(img.width * img.height / 1_000_000, 1)

        exif_data = img._getexif()
        if exif_data:
            tag_names = {v: k for k, v in ExifTags.TAGS.items()}
            for tag_id, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                if tag_name == "DateTimeOriginal":
                    meta["date_taken"] = str(value)
                elif tag_name == "Make":
                    meta["camera_make"] = str(value)
                elif tag_name == "Model":
                    meta["camera_model"] = str(value)
                elif tag_name == "FocalLength":
                    if hasattr(value, "numerator"):
                        meta["focal_length"] = f"{value.numerator / value.denominator}mm"
                    else:
                        meta["focal_length"] = str(value)
                elif tag_name == "FNumber":
                    if hasattr(value, "numerator"):
                        meta["aperture"] = f"f/{value.numerator / value.denominator:.1f}"
                    else:
                        meta["aperture"] = str(value)
                elif tag_name == "ISOSpeedRatings":
                    meta["iso"] = str(value)
                elif tag_name == "Orientation":
                    meta["orientation"] = int(value)
        img.close()
    except Exception as e:
        meta["exif_error"] = str(e)
    return meta


def detect_faces(img_path: Path) -> dict:
    """Detect faces in an image using OpenCV Haar cascades.

    Returns dict with face_count, largest_face_pct, face_positions.
    """
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return {"face_count": 0, "error": "could not read image"}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Try frontal faces first
        faces = FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )
        if len(faces) == 0:
            # Try profile faces
            faces = PROFILE_CASCADE.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )

        face_count = len(faces)
        largest_face_pct = 0.0
        face_positions = []

        for (x, y, fw, fh) in faces:
            face_pct = (fw * fh) / (w * h) * 100
            largest_face_pct = max(largest_face_pct, face_pct)
            face_positions.append({
                "x": int(x), "y": int(y),
                "w": int(fw), "h": int(fh),
                "pct_of_frame": round(face_pct, 1),
            })

        return {
            "face_count": face_count,
            "largest_face_pct": round(largest_face_pct, 1),
            "face_positions": face_positions,
            "image_w": w,
            "image_h": h,
        }
    except Exception as e:
        return {"face_count": 0, "error": str(e)}


def score_sharpness(img_path: Path) -> float:
    """Laplacian variance — higher = sharper image."""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Resize for consistent scoring (don't penalize small images)
        gray = cv2.resize(gray, (512, 512))
        return round(cv2.Laplacian(gray, cv2.CV_64F).var(), 2)
    except Exception:
        return 0.0


def score_image(meta: dict, faces: dict, sharpness: float) -> tuple[float, str]:
    """Score an image 0-100 for LoRA training suitability.

    Returns (score, tier) where tier is 'keep', 'maybe', or 'skip'.
    """
    score = 50.0  # baseline
    reasons = []

    # Face detection (biggest factor)
    fc = faces.get("face_count", 0)
    if fc == 0:
        score -= 30
        reasons.append("no face detected")
    elif fc == 1:
        score += 20
        reasons.append("single face")
        # Face size bonus — bigger face = more detail to learn
        face_pct = faces.get("largest_face_pct", 0)
        if face_pct > 15:
            score += 15
            reasons.append(f"large face ({face_pct}%)")
        elif face_pct > 5:
            score += 8
            reasons.append(f"medium face ({face_pct}%)")
        elif face_pct > 2:
            score += 2
            reasons.append(f"small face ({face_pct}%)")
        else:
            score -= 10
            reasons.append(f"tiny face ({face_pct}%)")
    elif fc == 2:
        score -= 5
        reasons.append(f"{fc} faces — may need crop")
    else:
        score -= 20
        reasons.append(f"{fc} faces — group photo")

    # Resolution
    mp = meta.get("megapixels", 0)
    if mp >= 4:
        score += 10
        reasons.append(f"high res ({mp}MP)")
    elif mp >= 2:
        score += 5
    elif mp < 0.5:
        score -= 15
        reasons.append(f"very low res ({mp}MP)")

    # Sharpness
    if sharpness > 200:
        score += 10
        reasons.append("sharp")
    elif sharpness > 50:
        score += 5
    elif sharpness < 15:
        score -= 15
        reasons.append("blurry")

    # File size (proxy for quality/compression)
    size_kb = meta.get("file_size_kb", 0)
    if size_kb > 2000:
        score += 5
    elif size_kb < 50:
        score -= 10
        reasons.append("very compressed")

    # Clamp
    score = max(0, min(100, score))

    # Tier assignment
    if score >= 65:
        tier = "keep"
    elif score >= 40:
        tier = "maybe"
    else:
        tier = "skip"

    return score, tier, reasons


def analyze_collection(source_dir: Path) -> list[dict]:
    """Analyze all images in source directory."""
    images = sorted([
        f for f in source_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTS
    ])
    total = len(images)
    log.info("Analyzing %d images in %s...", total, source_dir)

    results = []
    for i, img_path in enumerate(images):
        if (i + 1) % 25 == 0 or i == 0:
            log.info("  [%d/%d] %s", i + 1, total, img_path.name)

        meta = extract_exif(img_path)
        faces = detect_faces(img_path)
        sharpness = score_sharpness(img_path)
        score, tier, reasons = score_image(meta, faces, sharpness)

        results.append({
            **meta,
            **faces,
            "sharpness": sharpness,
            "score": score,
            "tier": tier,
            "reasons": reasons,
            "source_path": str(img_path),
        })

    return results


def sort_images(results: list[dict]):
    """Copy images into tier folders."""
    for d in [KEEP_DIR, MAYBE_DIR, SKIP_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    tier_dirs = {"keep": KEEP_DIR, "maybe": MAYBE_DIR, "skip": SKIP_DIR}
    counts = {"keep": 0, "maybe": 0, "skip": 0}

    for r in results:
        src = Path(r["source_path"])
        dest_dir = tier_dirs[r["tier"]]
        dest = dest_dir / src.name
        if not dest.exists():
            shutil.copy2(src, dest)
        counts[r["tier"]] += 1

    return counts


def write_metadata(results: list[dict]):
    """Write metadata CSV for review."""
    TRAINING_BASE.mkdir(parents=True, exist_ok=True)
    fields = [
        "filename", "score", "tier", "face_count", "largest_face_pct",
        "megapixels", "sharpness", "date_taken", "file_date",
        "camera_make", "camera_model", "focal_length", "aperture", "iso",
        "width", "height", "file_size_kb", "reasons",
    ]
    with open(METADATA_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(results, key=lambda x: -x["score"]):
            row = dict(r)
            row["reasons"] = "; ".join(r.get("reasons", []))
            writer.writerow(row)
    log.info("Metadata saved: %s", METADATA_FILE)


def write_report(results: list[dict], counts: dict):
    """Write JSON summary report."""
    dates = [r.get("date_taken", r.get("file_date", "")) for r in results if r.get("date_taken") or r.get("file_date")]
    cameras = set(
        f"{r.get('camera_make', '?')} {r.get('camera_model', '?')}"
        for r in results
        if r.get("camera_make")
    )
    avg_score = sum(r["score"] for r in results) / len(results) if results else 0
    avg_sharpness = sum(r["sharpness"] for r in results) / len(results) if results else 0

    report = {
        "total_images": len(results),
        "sort_counts": counts,
        "average_score": round(avg_score, 1),
        "average_sharpness": round(avg_sharpness, 1),
        "cameras_found": sorted(cameras),
        "date_range": {
            "earliest": min(dates) if dates else "unknown",
            "latest": max(dates) if dates else "unknown",
        },
        "resolution_breakdown": {
            "high_4mp+": sum(1 for r in results if r.get("megapixels", 0) >= 4),
            "medium_2-4mp": sum(1 for r in results if 2 <= r.get("megapixels", 0) < 4),
            "low_under_2mp": sum(1 for r in results if r.get("megapixels", 0) < 2),
        },
        "face_detection": {
            "single_face": sum(1 for r in results if r.get("face_count") == 1),
            "multiple_faces": sum(1 for r in results if r.get("face_count", 0) > 1),
            "no_face_detected": sum(1 for r in results if r.get("face_count", 0) == 0),
        },
        "top_20_images": [
            {"filename": r["filename"], "score": r["score"], "reasons": r["reasons"]}
            for r in sorted(results, key=lambda x: -x["score"])[:20]
        ],
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info("Report saved: %s", REPORT_FILE)
    return report


def caption_with_vision(persona_name: str = "ann"):
    """Caption curated images using the local vision model.

    Generates detailed physical descriptions focused on the subject.
    """
    import ollama

    client = ollama.Client()
    trigger = f"ohwx {persona_name}"

    # Look for curated images first, fall back to keep
    img_dir = CURATED_DIR if CURATED_DIR.exists() and any(CURATED_DIR.iterdir()) else KEEP_DIR
    if not img_dir.exists():
        log.error("No images found. Run --source first to sort images.")
        return

    images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS])
    log.info("Captioning %d images in %s with vision model...", len(images), img_dir)

    caption_prompt = (
        "Describe this person in detail for an AI image generation caption. "
        "Focus on physical appearance: face shape, eye color, eye shape, hair color, "
        "hair style, hair length, skin tone, skin texture, body type, body shape. "
        "Describe their expression, pose, and what they're wearing. "
        "If they have makeup, describe the makeup style in detail (eye shadow color, "
        "lip color, eyeliner style, blush, foundation). "
        "Describe the lighting and camera angle. "
        "Be factual, specific, and detailed. Use comma-separated descriptive phrases. "
        "Do NOT use the person's name. Start with physical descriptors. "
        "Example format: woman, brunette, long wavy brown hair, green eyes, "
        "fair skin, round face, soft features, smiling, wearing black tank top, "
        "natural makeup, indoor lighting, selfie angle, head and shoulders"
    )

    captioned = 0
    skipped = 0
    for i, img_path in enumerate(images):
        caption_file = img_path.with_suffix(".txt")
        if caption_file.exists():
            existing = caption_file.read_text(encoding="utf-8").strip()
            if existing and existing != f"{trigger}, a photo of a person":
                skipped += 1
                continue

        log.info("  [%d/%d] Captioning: %s", i + 1, len(images), img_path.name)

        try:
            # Read image as base64 for vision model
            import base64
            img_b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")

            response = client.chat(
                model="llava-llama3:8b",
                messages=[{
                    "role": "user",
                    "content": caption_prompt,
                    "images": [img_b64],
                }],
            )
            raw_caption = response["message"]["content"].strip()

            # Clean up: remove sentences, keep comma-separated phrases
            # Remove "The person" / "This person" / "In this image" prefixes
            import re
            raw_caption = re.sub(r"^(The |This |In this |Here we see |The image shows )", "", raw_caption)
            # Convert periods to commas for SD-style captioning
            raw_caption = raw_caption.replace(". ", ", ").rstrip(".")

            # Prepend trigger word
            full_caption = f"{trigger}, {raw_caption}"
            caption_file.write_text(full_caption, encoding="utf-8")
            captioned += 1
            log.info("    Caption: %s", full_caption[:120])

        except Exception as e:
            log.error("    Failed: %s", e)

    log.info("Done! Captioned: %d, Skipped: %d (already had captions)", captioned, skipped)


def main():
    parser = argparse.ArgumentParser(description="Curate training images for LoRA")
    parser.add_argument("--source", type=str, help="Source directory of raw images")
    parser.add_argument("--caption", action="store_true", help="Run vision captioning on curated/keep images")
    parser.add_argument("--persona", type=str, default="ann", help="Persona name (default: ann)")
    args = parser.parse_args()

    if args.source:
        source = Path(args.source)
        if not source.exists():
            log.error("Source directory not found: %s", source)
            sys.exit(1)

        # Stage 1: Analyze, score, sort
        results = analyze_collection(source)
        counts = sort_images(results)
        write_metadata(results)
        report = write_report(results, counts)

        log.info("")
        log.info("=" * 60)
        log.info("CURATION REPORT")
        log.info("=" * 60)
        log.info("Total images analyzed: %d", report["total_images"])
        log.info("")
        log.info("Sort results:")
        log.info("  KEEP:  %d images (score >= 65)", counts["keep"])
        log.info("  MAYBE: %d images (score 40-65)", counts["maybe"])
        log.info("  SKIP:  %d images (score < 40)", counts["skip"])
        log.info("")
        log.info("Face detection:")
        fd = report["face_detection"]
        log.info("  Single face: %d", fd["single_face"])
        log.info("  Multiple faces: %d", fd["multiple_faces"])
        log.info("  No face detected: %d", fd["no_face_detected"])
        log.info("")
        log.info("Resolution breakdown:")
        rb = report["resolution_breakdown"]
        log.info("  High (4MP+):    %d", rb["high_4mp+"])
        log.info("  Medium (2-4MP): %d", rb["medium_2-4mp"])
        log.info("  Low (<2MP):     %d", rb["low_under_2mp"])
        log.info("")
        log.info("Cameras found: %s", ", ".join(report["cameras_found"]) or "none detected")
        log.info("Date range: %s to %s", report["date_range"]["earliest"], report["date_range"]["latest"])
        log.info("")
        log.info("Files saved:")
        log.info("  Metadata CSV: %s", METADATA_FILE)
        log.info("  Full report:  %s", REPORT_FILE)
        log.info("")
        log.info("NEXT STEPS:")
        log.info("  1. Review KEEP folder: %s", KEEP_DIR)
        log.info("     Move your favorites to: %s", CURATED_DIR)
        log.info("  2. Check MAYBE folder for any good ones: %s", MAYBE_DIR)
        log.info("  3. Target: 150-200 curated images")
        log.info("  4. Run captioning: python scripts/curate_training_images.py --caption")

    elif args.caption:
        caption_with_vision(args.persona)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
