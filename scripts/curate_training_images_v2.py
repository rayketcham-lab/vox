"""Deep curation pipeline for LoRA training images (v2).

Replaces the fast Haar-cascade sort with:
1. DNN-based face detection (ResNet SSD — far more accurate than Haar)
2. Per-face quality analysis (sharpness on face region, size, centering)
3. Auto-crop faces from group photos with generous padding
4. Date-aware scoring (recent photos weighted higher)
5. Strict tier system for staged training:
   - stage1_faces: single face, close-up (face >= 15% of frame), sharp
   - stage1_crops: auto-cropped faces from multi-face images (review these)
   - stage2_body: single-person full/medium body shots
   - skip: blurry, tiny, no face, distant

Stage 1 training uses ONLY stage1_faces (+ reviewed crops) — face identity first.
Stage 2 adds body shots after face identity is locked in.

Usage:
    python scripts/curate_training_images_v2.py --source "path/to/photos"
    python scripts/curate_training_images_v2.py --source "path/to/photos" --recent-years 5
    python scripts/curate_training_images_v2.py --caption  # re-caption stage1 images
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import logging
import re
import shutil
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ExifTags, ImageFilter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_BASE = PROJECT_ROOT / "persona" / "training" / "ann"
STAGE1_FACES = TRAINING_BASE / "stage1_faces"
STAGE1_CROPS = TRAINING_BASE / "stage1_crops"
STAGE2_BODY = TRAINING_BASE / "stage2_body"
SKIP_DIR = TRAINING_BASE / "sort_skip_v2"
METADATA_FILE = TRAINING_BASE / "metadata_v2.csv"
REPORT_FILE = TRAINING_BASE / "curation_report_v2.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# DNN face detector model files (downloaded on first run)
MODEL_DIR = PROJECT_ROOT / "models" / "face_detection"
PROTOTXT = MODEL_DIR / "deploy.prototxt"
CAFFEMODEL = MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

# Thresholds
DNN_CONFIDENCE = 0.6       # Min confidence for face detection
FACE_CLOSE_UP_PCT = 15.0   # Face >= 15% of frame = close-up
FACE_MEDIUM_PCT = 5.0      # Face >= 5% of frame = medium shot
MIN_FACE_PX = 80           # Min face size in pixels
SHARPNESS_GOOD = 100.0     # Face region Laplacian variance threshold
SHARPNESS_OK = 40.0
CROP_PADDING = 1.8         # Multiply face bbox by this for crop (1.8 = generous head+shoulders)


def _ensure_dnn_model():
    """Download the DNN face detection model if not present."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not PROTOTXT.exists():
        log.info("Downloading DNN face detector prototxt...")
        url = ("https://raw.githubusercontent.com/opencv/opencv/master/"
               "samples/dnn/face_detector/deploy.prototxt")
        urllib.request.urlretrieve(url, str(PROTOTXT))

    if not CAFFEMODEL.exists():
        log.info("Downloading DNN face detector model (~10MB)...")
        url = ("https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
               "dnn_samples_face_detector_20170830/"
               "res10_300x300_ssd_iter_140000.caffemodel")
        urllib.request.urlretrieve(url, str(CAFFEMODEL))

    return cv2.dnn.readNetFromCaffe(str(PROTOTXT), str(CAFFEMODEL))


def detect_faces_dnn(img: np.ndarray, net) -> list[dict]:
    """Detect faces using OpenCV DNN ResNet-SSD.

    Returns list of face dicts with bbox, confidence, size info.
    Much more accurate than Haar cascades, especially for angled/partial faces.
    """
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False,
    )
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < DNN_CONFIDENCE:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        # Clamp to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        fw = x2 - x1
        fh = y2 - y1
        if fw < MIN_FACE_PX or fh < MIN_FACE_PX:
            continue

        face_pct = (fw * fh) / (w * h) * 100
        faces.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "w": fw, "h": fh,
            "confidence": round(confidence, 3),
            "pct_of_frame": round(face_pct, 1),
            "center_x": (x1 + x2) / 2 / w,  # 0-1 normalized
            "center_y": (y1 + y2) / 2 / h,
        })

    # Sort by size (largest first)
    faces.sort(key=lambda f: f["w"] * f["h"], reverse=True)
    return faces


def score_face_sharpness(img: np.ndarray, face: dict) -> float:
    """Measure sharpness specifically on the face region, not the whole image."""
    x1, y1, x2, y2 = face["x1"], face["y1"], face["x2"], face["y2"]
    face_roi = img[y1:y2, x1:x2]
    if face_roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
    # Resize to consistent size for comparable scores
    gray = cv2.resize(gray, (200, 200))
    return round(cv2.Laplacian(gray, cv2.CV_64F).var(), 2)


def estimate_face_angle(face: dict) -> str:
    """Rough estimate of face angle from bbox aspect ratio and position."""
    aspect = face["w"] / max(face["h"], 1)
    if aspect < 0.6:
        return "profile"
    if aspect > 1.1:
        return "tilted"
    return "frontal"


def extract_date(img_path: Path) -> str | None:
    """Extract date taken from EXIF."""
    try:
        img = Image.open(img_path)
        exif = img._getexif()
        if exif:
            for tag_id, value in exif.items():
                if ExifTags.TAGS.get(tag_id) == "DateTimeOriginal":
                    return str(value)
        img.close()
    except Exception:
        pass
    return None


def get_year(date_str: str | None, file_path: Path) -> int:
    """Extract year from EXIF date or filename or file date."""
    if date_str:
        try:
            return int(date_str[:4])
        except (ValueError, IndexError):
            pass
    # Try filename pattern like 20180428_023434.jpg
    name = file_path.stem
    if len(name) >= 8 and name[:8].isdigit():
        try:
            year = int(name[:4])
            if 1990 <= year <= 2030:
                return year
        except ValueError:
            pass
    # Fall back to file modification time
    try:
        return datetime.fromtimestamp(file_path.stat().st_mtime).year
    except Exception:
        return 2020


def auto_crop_face(img: np.ndarray, face: dict, padding: float = CROP_PADDING) -> np.ndarray:
    """Crop image around a face with generous padding for head+shoulders."""
    h, w = img.shape[:2]
    cx = (face["x1"] + face["x2"]) / 2
    cy = (face["y1"] + face["y2"]) / 2
    fw = face["w"] * padding
    fh = face["h"] * padding

    # Shift crop slightly down to include shoulders
    cy += face["h"] * 0.15

    x1 = int(max(0, cx - fw / 2))
    y1 = int(max(0, cy - fh / 2))
    x2 = int(min(w, cx + fw / 2))
    y2 = int(min(h, cy + fh / 2))

    # Ensure minimum crop size
    if (x2 - x1) < 256 or (y2 - y1) < 256:
        return None

    return img[y1:y2, x1:x2]


def score_image(
    faces: list[dict],
    face_sharpness: float,
    megapixels: float,
    year: int,
    current_year: int,
    recent_years: int,
) -> tuple[float, str, list[str]]:
    """Score an image for LoRA training suitability.

    Returns (score, tier, reasons).
    Tiers: stage1_face, stage1_crop, stage2_body, skip
    """
    score = 50.0
    reasons = []

    fc = len(faces)

    # --- Face count scoring ---
    if fc == 0:
        return 10.0, "skip", ["no face detected"]

    if fc == 1:
        score += 25
        reasons.append("single face")
    elif fc == 2:
        score -= 5
        reasons.append(f"{fc} faces — will auto-crop")
    else:
        score -= 15
        reasons.append(f"{fc} faces — group photo, will auto-crop")

    # --- Face size (biggest factor for likeness training) ---
    best_face = faces[0]
    face_pct = best_face["pct_of_frame"]
    if face_pct >= 30:
        score += 25
        reasons.append(f"very close-up ({face_pct}%)")
    elif face_pct >= FACE_CLOSE_UP_PCT:
        score += 18
        reasons.append(f"close-up ({face_pct}%)")
    elif face_pct >= FACE_MEDIUM_PCT:
        score += 8
        reasons.append(f"medium shot ({face_pct}%)")
    elif face_pct >= 2:
        score -= 5
        reasons.append(f"distant ({face_pct}%)")
    else:
        score -= 20
        reasons.append(f"very distant ({face_pct}%)")

    # --- Face sharpness (on face region specifically) ---
    if face_sharpness >= SHARPNESS_GOOD:
        score += 12
        reasons.append(f"sharp face ({face_sharpness:.0f})")
    elif face_sharpness >= SHARPNESS_OK:
        score += 5
        reasons.append(f"ok sharpness ({face_sharpness:.0f})")
    else:
        score -= 15
        reasons.append(f"blurry face ({face_sharpness:.0f})")

    # --- Face angle ---
    angle = estimate_face_angle(best_face)
    if angle == "frontal":
        score += 5
        reasons.append("frontal")
    elif angle == "profile":
        score -= 5
        reasons.append("profile angle")

    # --- DNN confidence ---
    conf = best_face["confidence"]
    if conf >= 0.95:
        score += 5
    elif conf < 0.7:
        score -= 10
        reasons.append(f"low confidence ({conf})")

    # --- Resolution ---
    if megapixels >= 4:
        score += 8
        reasons.append(f"high res ({megapixels}MP)")
    elif megapixels >= 2:
        score += 4
    elif megapixels < 0.5:
        score -= 12
        reasons.append(f"low res ({megapixels}MP)")

    # --- Recency bonus ---
    age = current_year - year
    if age <= 2:
        score += 8
        reasons.append(f"recent ({year})")
    elif age <= 5:
        score += 4
        reasons.append(f"fairly recent ({year})")
    elif age > recent_years:
        score -= 10
        reasons.append(f"old photo ({year}, >{recent_years}yr)")

    # --- Centering bonus ---
    cx = best_face["center_x"]
    cy = best_face["center_y"]
    if 0.25 <= cx <= 0.75 and 0.15 <= cy <= 0.65:
        score += 3
    else:
        reasons.append("face off-center")

    score = max(0, min(100, score))

    # --- Tier assignment ---
    if fc == 1 and face_pct >= FACE_CLOSE_UP_PCT and face_sharpness >= SHARPNESS_OK and score >= 70:
        tier = "stage1_face"
    elif fc == 1 and face_pct >= FACE_MEDIUM_PCT and face_sharpness >= SHARPNESS_OK and score >= 55:
        tier = "stage2_body"
    elif fc > 1 and face_pct >= FACE_MEDIUM_PCT:
        tier = "stage1_crop"  # will auto-crop the largest face
    else:
        tier = "skip"

    return score, tier, reasons


def analyze_and_sort(source_dir: Path, recent_years: int = 10):
    """Full analysis pipeline: detect, score, crop, sort."""
    source_dir = Path(source_dir)
    images = sorted([f for f in source_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS])
    total = len(images)
    if total == 0:
        log.error("No images found in %s", source_dir)
        return

    log.info("=" * 60)
    log.info("DEEP CURATION v2 — %d images", total)
    log.info("=" * 60)

    # Download/load DNN model
    log.info("Loading DNN face detector...")
    net = _ensure_dnn_model()

    # Create output dirs
    for d in [STAGE1_FACES, STAGE1_CROPS, STAGE2_BODY, SKIP_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    current_year = datetime.now().year
    results = []
    tier_counts = {"stage1_face": 0, "stage1_crop": 0, "stage2_body": 0, "skip": 0}
    crops_saved = 0
    start_time = time.time()

    for i, img_path in enumerate(images):
        elapsed = time.time() - start_time
        rate = (i + 1) / max(elapsed, 1)
        eta = (total - i - 1) / max(rate, 0.01)

        log.info("[%d/%d] %s (%.1f img/s, ETA: %s)",
                 i + 1, total, img_path.name, rate, _fmt_time(eta))

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            log.warning("  SKIP: could not read %s", img_path.name)
            results.append({"filename": img_path.name, "tier": "skip", "score": 0, "reasons": ["unreadable"]})
            tier_counts["skip"] += 1
            continue

        h, w = img.shape[:2]
        megapixels = round(w * h / 1_000_000, 1)

        # Detect faces (DNN — much better than Haar)
        faces = detect_faces_dnn(img, net)

        # Measure face sharpness on best face
        face_sharpness = score_face_sharpness(img, faces[0]) if faces else 0.0

        # Get date info
        date_str = extract_date(img_path)
        year = get_year(date_str, img_path)

        # Score
        score, tier, reasons = score_image(
            faces, face_sharpness, megapixels, year, current_year, recent_years,
        )

        log.info("  Score: %.0f | Tier: %s | Faces: %d | %s",
                 score, tier, len(faces), ", ".join(reasons[:4]))

        # Sort into tier folders
        tier_dirs = {
            "stage1_face": STAGE1_FACES,
            "stage1_crop": STAGE1_CROPS,
            "stage2_body": STAGE2_BODY,
            "skip": SKIP_DIR,
        }
        dest_dir = tier_dirs[tier]

        if tier == "stage1_crop" and faces:
            # Auto-crop the largest face with head+shoulders padding
            cropped = auto_crop_face(img, faces[0])
            if cropped is not None and cropped.shape[0] >= 256 and cropped.shape[1] >= 256:
                crop_name = f"crop_{img_path.stem}.jpg"
                crop_path = STAGE1_CROPS / crop_name
                cv2.imwrite(str(crop_path), cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
                crops_saved += 1
                log.info("  -> Cropped face saved: %s (%dx%d)",
                         crop_name, cropped.shape[1], cropped.shape[0])
            else:
                log.info("  -> Face too small to crop usefully")
            # Also copy original to crops dir for reference
            shutil.copy2(img_path, dest_dir / img_path.name)
        elif tier != "skip":
            dest = dest_dir / img_path.name
            if not dest.exists():
                shutil.copy2(img_path, dest)
        else:
            dest = dest_dir / img_path.name
            if not dest.exists():
                shutil.copy2(img_path, dest)

        tier_counts[tier] += 1
        results.append({
            "filename": img_path.name,
            "score": score,
            "tier": tier,
            "face_count": len(faces),
            "largest_face_pct": faces[0]["pct_of_frame"] if faces else 0,
            "face_sharpness": face_sharpness,
            "confidence": faces[0]["confidence"] if faces else 0,
            "megapixels": megapixels,
            "year": year,
            "date_taken": date_str or "",
            "width": w,
            "height": h,
            "reasons": reasons,
        })

    elapsed = time.time() - start_time

    # Write metadata
    _write_metadata(results)
    report = _write_report(results, tier_counts, crops_saved, elapsed)

    # Print summary
    log.info("")
    log.info("=" * 60)
    log.info("DEEP CURATION COMPLETE — %.1f minutes", elapsed / 60)
    log.info("=" * 60)
    log.info("")
    log.info("Tier breakdown:")
    log.info("  STAGE 1 FACES:  %d  (close-up, single face, sharp)", tier_counts["stage1_face"])
    log.info("  STAGE 1 CROPS:  %d  (auto-cropped from group photos → %d crops saved)",
             tier_counts["stage1_crop"], crops_saved)
    log.info("  STAGE 2 BODY:   %d  (medium/body shots for later training)", tier_counts["stage2_body"])
    log.info("  SKIP:           %d  (blurry, tiny, no face, distant)", tier_counts["skip"])
    log.info("")
    log.info("Output directories:")
    log.info("  %s  ← TRAIN WITH THESE FIRST", STAGE1_FACES)
    log.info("  %s  ← Review crops, move good ones to stage1_faces", STAGE1_CROPS)
    log.info("  %s  ← Add these in stage 2 after face identity is locked", STAGE2_BODY)
    log.info("")
    log.info("NEXT STEPS:")
    log.info("  1. Review stage1_faces — remove any bad images manually")
    log.info("  2. Review stage1_crops — move good face crops to stage1_faces")
    log.info("  3. Target: 100-200 face-focused images in stage1_faces")
    log.info("  4. Run captioning:")
    log.info("     python scripts/curate_training_images_v2.py --caption")
    log.info("  5. Train stage 1:")
    log.info("     python -m vox.train --persona ann --image-dir \"%s\" --epochs 5 --lr 1e-4 --alpha 32",
             STAGE1_FACES)

    return results


def _clean_caption(raw: str) -> str:
    """Clean BLIP caption into SD-friendly comma-separated tags.

    BLIP produces natural-language sentences like:
        "there is a woman wearing glasses and a green shirt holding a cell phone"
    We convert to SD tags:
        "wearing glasses, green shirt, holding cell phone"
    """
    raw = raw.strip('"\'.')

    # Strip BLIP sentence starters and conditional prompt echoes
    raw = re.sub(
        r"^(there\s+is\s+)?(a\s+)?(photo\s+of\s+)?(a\s+)?"
        r"(woman|girl|lady|person)\s+(who\s+is\s+|that\s+is\s+)?",
        "", raw, flags=re.IGNORECASE,
    )
    raw = re.sub(r"\bwoman\b", "", raw, flags=re.IGNORECASE)
    # Strip conditional prompt echoes: "the background shows...", "the lighting is..."
    raw = re.sub(r"^(the\s+)?(background|setting)\s+(shows?|is)\s+(a\s+|the\s+)?", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^(the\s+)?lighting\s+(in\s+this\s+photo\s+)?(is|shows?)\s+", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^(a\s+photo\s+of\s+)?(a\s+)?(woman|person)\s+(wearing|in|with)\s+", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^(the\s+)?(woman|person)\s+(is\s+)?", "", raw, flags=re.IGNORECASE)

    # Convert "and" / "with a" / "in a/the" to commas
    raw = re.sub(r"\s+and\s+", ", ", raw)
    raw = re.sub(r"\s+with\s+(a\s+|the\s+|her\s+)?", ", ", raw)
    raw = re.sub(r"\s+in\s+(a|the|an)\s+", ", ", raw)
    raw = re.sub(r"\s+on\s+(a|the|an)\s+", ", on ", raw)

    # Strip mid-sentence filler
    raw = re.sub(r"\bthere\s+(is|are)\s+", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\b(that|who)\s+is\s+", "", raw, flags=re.IGNORECASE)

    # Strip articles and filler prepositions before nouns
    raw = re.sub(r"\b(a|an|the|her|his|their)\s+", "", raw, flags=re.IGNORECASE)
    # "with glasses" → "glasses", "with smile" → "smile" (SD-friendly)
    raw = re.sub(r"\bwith\s+(glasses|smile|hat|scarf|tattoo|piercings?|makeup|earrings?|necklace|bracelet)\b",
                 r"\1", raw, flags=re.IGNORECASE)

    # Strip hair color, eye color, skin tone, age — we prepend correct ones
    strip_patterns = [
        r"(long\s+|short\s+|shoulder[- ]length\s+)?"
        r"(brown|blonde|blond|red|black|dark|light|auburn|brunette|ginger)"
        r"\s+(wavy\s+|straight\s+|curly\s+)?hair",
        r"(wavy|straight|curly)\s+hair",
        r"(blue|green|brown|hazel|dark|light|grey|gray)\s+eye(liner)?s?\b",
        r"(fair|light|pale|white|tan|dark|olive|medium)\s+skin",
    ]
    for pat in strip_patterns:
        raw = re.sub(r",?\s*\b" + pat, "", raw, flags=re.IGNORECASE)

    # Convert periods to commas, strip newlines
    raw = raw.replace(". ", ", ").rstrip(".")
    raw = re.sub(r"[\n\r]+", ", ", raw)

    # BLIP often joins clauses without commas: "green shirt holding cell phone"
    # Insert commas before participles (wearing, holding, sitting, standing, etc.)
    raw = re.sub(
        r"\b(\w+)\s+(wearing|holding|sitting|standing|looking|taking|smiling|posing|leaning|lying|eating)\b",
        r"\1, \2", raw,
    )

    # Strip orphan words left from aggressive stripping
    raw = re.sub(r"(^|,)\s*(with|and|in|on|at|for|a|an|the|is|are|to|of|it)\s*(,|$)", r"\1\3", raw)
    # May need two passes to catch adjacent orphans
    raw = re.sub(r"(^|,)\s*(with|and|in|on|at|for|a|an|the|is|are|to|of|it)\s*(,|$)", r"\1\3", raw)

    # Collapse commas/spaces
    raw = re.sub(r",\s*,+", ",", raw)
    raw = re.sub(r"\s{2,}", " ", raw)
    raw = raw.strip().strip(",").strip()
    return raw


def caption_stage1(persona_name: str = "ann", force: bool = False):
    """Caption stage1 images using BLIP (Salesforce/blip-image-captioning-large).

    BLIP is a purpose-built captioning model — more accurate than chat-based vision
    models (llava) which produce generic/templated responses. BLIP actually looks at
    each image and describes what it sees (clothing, objects, setting, pose).

    We prepend fixed identity attributes (trigger word, hair, eyes, skin) since
    BLIP sometimes gets those wrong, but its variable descriptions (clothing,
    background, pose, objects) are accurate.
    """
    import torch
    import httpx
    from PIL import Image as PILImage

    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
    except ImportError:
        log.error("transformers required: pip install transformers")
        return

    trigger = f"ohwx {persona_name}"
    img_dir = STAGE1_FACES

    if not img_dir.exists() or not any(img_dir.iterdir()):
        log.error("No stage1 images found. Run --source first.")
        return

    images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS])
    log.info("Captioning %d stage1 images with BLIP%s...",
             len(images), " (force overwrite)" if force else "")

    # Load BLIP model — ~1GB VRAM in float16
    log.info("Loading BLIP captioning model...")
    blip_id = "Salesforce/blip-image-captioning-large"
    processor = BlipProcessor.from_pretrained(blip_id)
    model = BlipForConditionalGeneration.from_pretrained(
        blip_id, dtype=torch.float16,
    ).to("cuda")
    log.info("BLIP loaded on GPU.")

    # Fixed prefix for every caption
    fixed_prefix = f"{trigger}, woman, brown wavy hair, green eyes, fair skin"

    captioned = 0
    skipped = 0

    for i, img_path in enumerate(images):
        caption_file = img_path.with_suffix(".txt")
        if not force and caption_file.exists():
            existing = caption_file.read_text(encoding="utf-8").strip()
            if existing and len(existing) > 50:
                skipped += 1
                continue

        log.info("[%d/%d] %s", i + 1, len(images), img_path.name)

        try:
            img = PILImage.open(img_path).convert("RGB")

            # --- Pass 1: BLIP quick scan for objects and actions ---
            inputs = processor(img, return_tensors="pt").to("cuda", torch.float16)
            out = model.generate(**inputs, max_new_tokens=80, num_beams=3)
            blip_raw = processor.decode(out[0], skip_special_tokens=True)
            blip_clean = _clean_caption(blip_raw)

            # --- Pass 2: Ollama vision model for targeted detail questions ---
            # Each question forces the model to look at a specific aspect
            img_b64 = base64.b64encode(img_path.read_bytes()).decode()
            detail_questions = [
                ("clothing", "What COLOR shirt or top is this person wearing? Example: 'green tank top' or 'black t-shirt'. Just the item and color, nothing else."),
                ("expression", "One or two words for this person's expression. Example: 'wide smile' or 'smirk' or 'neutral'. Just the expression."),
                ("pose", "One or two words for pose. Example: 'selfie angle' or 'looking at camera' or 'sitting'. Just the pose."),
                ("background", "What room or place is behind this person? Example: 'kitchen' or 'car interior' or 'living room'. One or two words only."),
                ("lighting", "Is the lighting warm, cool, bright, dim, natural, or flash? One or two words only."),
                ("accessories", "List visible accessories: glasses, necklace, earrings, phone, hat. Just list items separated by commas. Say 'none' if nothing."),
            ]

            detail_parts = []
            for aspect, question in detail_questions:
                try:
                    resp = httpx.post(
                        "http://127.0.0.1:11434/api/generate",
                        json={
                            "model": "llava-llama3:8b",
                            "prompt": question,
                            "images": [img_b64],
                            "stream": False,
                            "options": {"temperature": 0.1, "num_predict": 30},
                        },
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        answer = resp.json().get("response", "").strip().rstrip(".")
                        # Clean the answer
                        answer = re.sub(r"^(the\s+)?(person|woman|subject)\s+(is\s+)?", "", answer, flags=re.IGNORECASE)
                        answer = re.sub(r"^(she\s+is\s+|she'?s\s+)", "", answer, flags=re.IGNORECASE)
                        answer = answer.strip().rstrip(".")
                        if answer and answer.lower() not in ("none", "n/a", "not visible", "unknown"):
                            detail_parts.append(answer.lower())
                except Exception:
                    pass

            # --- Combine BLIP + vision detail, deduplicate ---
            all_phrases = set()
            for source in [blip_clean] + detail_parts:
                for phrase in source.split(","):
                    phrase = phrase.strip().rstrip(".")
                    if len(phrase) < 3:
                        continue
                    pl = phrase.lower()
                    if pl in {"is", "are", "the", "of", "none", "n/a", "not visible"}:
                        continue
                    all_phrases.add(phrase)

            # Deduplicate near-matches (keep longer version)
            deduped = set()
            for phrase in sorted(all_phrases, key=len, reverse=True):
                if any(phrase.lower() in kept.lower() for kept in deduped):
                    continue
                deduped.add(phrase)

            combined = ", ".join(sorted(deduped))
            caption = f"{fixed_prefix}, {combined}" if combined else fixed_prefix
            caption_file.write_text(caption, encoding="utf-8")
            captioned += 1
            log.info("  Caption: %s", caption[:200])

        except Exception as e:
            log.error("  Failed: %s", e)

    log.info("")
    log.info("Captioning complete: %d new, %d already had captions", captioned, skipped)


def _write_metadata(results: list[dict]):
    """Write detailed metadata CSV."""
    TRAINING_BASE.mkdir(parents=True, exist_ok=True)
    fields = [
        "filename", "score", "tier", "face_count", "largest_face_pct",
        "face_sharpness", "confidence", "megapixels", "year", "date_taken",
        "width", "height", "reasons",
    ]
    with open(METADATA_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(results, key=lambda x: -x["score"]):
            row = dict(r)
            row["reasons"] = "; ".join(r.get("reasons", []))
            writer.writerow(row)
    log.info("Metadata saved: %s", METADATA_FILE)


def _write_report(results: list[dict], counts: dict, crops: int, elapsed: float) -> dict:
    """Write JSON summary report."""
    years = [r.get("year", 0) for r in results if r.get("year")]
    avg_score = sum(r["score"] for r in results) / len(results) if results else 0
    avg_sharpness = sum(r.get("face_sharpness", 0) for r in results) / len(results) if results else 0

    report = {
        "total_images": len(results),
        "elapsed_seconds": round(elapsed, 1),
        "tier_counts": counts,
        "auto_crops_saved": crops,
        "average_score": round(avg_score, 1),
        "average_face_sharpness": round(avg_sharpness, 1),
        "year_range": {"min": min(years) if years else 0, "max": max(years) if years else 0},
        "face_detection": {
            "single_face": sum(1 for r in results if r.get("face_count") == 1),
            "multiple_faces": sum(1 for r in results if r.get("face_count", 0) > 1),
            "no_face": sum(1 for r in results if r.get("face_count", 0) == 0),
        },
        "top_stage1": [
            {"filename": r["filename"], "score": r["score"],
             "face_pct": r.get("largest_face_pct", 0),
             "sharpness": r.get("face_sharpness", 0),
             "reasons": r["reasons"]}
            for r in sorted(results, key=lambda x: -x["score"])
            if r["tier"] == "stage1_face"
        ][:30],
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info("Report saved: %s", REPORT_FILE)
    return report


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.1f}m"


def main():
    parser = argparse.ArgumentParser(description="Deep curation for LoRA training (v2)")
    parser.add_argument("--source", type=str, help="Source directory of raw images to analyze")
    parser.add_argument("--caption", action="store_true", help="Run vision captioning on stage1 images")
    parser.add_argument("--force", action="store_true", help="Overwrite existing captions")
    parser.add_argument("--persona", type=str, default="ann", help="Persona name (default: ann)")
    parser.add_argument("--recent-years", type=int, default=10,
                        help="Photos older than this get penalized (default: 10)")
    args = parser.parse_args()

    if args.source:
        analyze_and_sort(Path(args.source), recent_years=args.recent_years)
    elif args.caption:
        caption_stage1(args.persona, force=args.force)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
