"""LoRA training pipeline — fine-tune Stable Diffusion on reference photos.

Upload reference photos of a subject, then train a LoRA that captures their
face, body, and style. The trained LoRA is automatically loaded during image
generation so the persona looks like the subject.

Workflow:
  1. Upload 10-100 photos to persona/training/
  2. Run training: vox --train-lora
  3. LoRA saves to models/lora/<persona_name>/
  4. Image generation auto-loads the LoRA when generating persona selfies

Requires: pip install -e ".[lora]"
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from vox.config import MODELS_DIR, PROJECT_ROOT

log = logging.getLogger(__name__)

# Training directories
TRAINING_DIR = PROJECT_ROOT / "persona" / "training"
LORA_OUTPUT_DIR = MODELS_DIR / "lora"


def setup_training_dirs(persona_name: str) -> dict[str, Path]:
    """Create the directory structure for LoRA training.

    Returns dict with paths: training_images, output, config.
    """
    slug = persona_name.lower().replace(" ", "-")
    paths = {
        "training_images": TRAINING_DIR / slug,
        "output": LORA_OUTPUT_DIR / slug,
        "config": LORA_OUTPUT_DIR / slug / "training_config.json",
    }
    for p in paths.values():
        if p.suffix:  # file path, create parent
            p.parent.mkdir(parents=True, exist_ok=True)
        else:
            p.mkdir(parents=True, exist_ok=True)
    return paths


def add_training_images(persona_name: str, image_paths: list[str | Path]) -> dict:
    """Copy images into the training directory.

    Returns status dict with count and path.
    """
    paths = setup_training_dirs(persona_name)
    dest = paths["training_images"]
    added = 0
    skipped = 0

    for img_path in image_paths:
        img_path = Path(img_path)
        if not img_path.exists():
            log.warning("Training image not found: %s", img_path)
            skipped += 1
            continue
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            log.warning("Skipping non-image file: %s", img_path)
            skipped += 1
            continue
        dest_file = dest / img_path.name
        if not dest_file.exists():
            shutil.copy2(img_path, dest_file)
            added += 1
        else:
            skipped += 1

    total = len(list(dest.glob("*")))
    log.info("Training images: added=%d, skipped=%d, total=%d", added, skipped, total)
    return {"added": added, "skipped": skipped, "total": total, "path": str(dest)}


def get_training_status(persona_name: str) -> dict:
    """Check current training data and model status."""
    paths = setup_training_dirs(persona_name)
    images = list(paths["training_images"].glob("*"))
    image_count = len([f for f in images if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")])

    # Check for existing LoRA
    lora_files = list(paths["output"].glob("*.safetensors"))
    has_lora = len(lora_files) > 0
    lora_file = str(lora_files[0]) if lora_files else None

    # Check for training config/progress
    config_exists = paths["config"].exists()
    config = {}
    if config_exists:
        with open(paths["config"]) as f:
            config = json.load(f)

    return {
        "persona_name": persona_name,
        "image_count": image_count,
        "image_path": str(paths["training_images"]),
        "has_lora": has_lora,
        "lora_file": lora_file,
        "lora_output_path": str(paths["output"]),
        "training_config": config,
        "ready_to_train": image_count >= 10,
        "recommendation": _get_recommendation(image_count),
    }


def _get_recommendation(count: int) -> str:
    """Training recommendation based on image count."""
    if count == 0:
        return "No images yet. Upload 10-100 reference photos to get started."
    if count < 10:
        return f"Only {count} images. Need at least 10, recommend 20-50 for best results."
    if count < 20:
        return f"{count} images — minimum met. More variety (20-50) will improve quality."
    if count <= 100:
        return f"{count} images — great dataset. Ready to train."
    return f"{count} images — plenty. You can trim to your best 50-100 for faster training."


def generate_training_config(
    persona_name: str,
    *,
    trigger_word: str | None = None,
    resolution: int = 1024,
    train_steps: int = 1500,
    learning_rate: float = 1e-4,
    network_rank: int = 32,
    network_alpha: int = 16,
    batch_size: int = 1,
    save_every_n_steps: int = 500,
) -> dict:
    """Generate a training configuration for kohya_ss / sd-scripts.

    This creates a config file that can be used with:
    - kohya_ss GUI
    - sd-scripts CLI (accelerate launch)
    - or our built-in training wrapper

    Returns the config dict and saves it to disk.
    """
    paths = setup_training_dirs(persona_name)
    slug = persona_name.lower().replace(" ", "-")
    trigger = trigger_word or f"ohwx {slug}"

    config = {
        "persona_name": persona_name,
        "trigger_word": trigger,
        "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "training_data": str(paths["training_images"]),
        "output_dir": str(paths["output"]),
        "output_name": f"{slug}_lora",
        "resolution": resolution,
        "train_steps": train_steps,
        "learning_rate": learning_rate,
        "network_rank": network_rank,
        "network_alpha": network_alpha,
        "batch_size": batch_size,
        "save_every_n_steps": save_every_n_steps,
        "mixed_precision": "fp16",
        "optimizer": "AdamW8bit",
        "lr_scheduler": "cosine",
        "caption_extension": ".txt",
        "shuffle_caption": True,
        "keep_tokens": 1,
        "max_token_length": 225,
        "seed": 42,
    }

    with open(paths["config"], "w") as f:
        json.dump(config, f, indent=2)

    log.info("Training config saved: %s", paths["config"])
    return config


def auto_caption_images(persona_name: str, trigger_word: str | None = None,
                        use_vision: bool = True, overwrite: bool = False) -> dict:
    """Generate caption .txt files for each training image.

    When use_vision=True, uses Ollama's vision model (llava) to describe each
    image in detail, prepending the trigger word. This produces captions like:
        "ohwx ann, a woman with shoulder-length brown hair, blue eyes, slight
         smile, wearing a red tank top, standing in a kitchen with warm lighting"

    Detailed per-image captions are critical for LoRA quality — they teach the
    model to separate identity (trigger word) from pose, clothing, and setting.

    Falls back to generic captions if the vision model is unavailable.

    Returns dict with count of captions created.
    """
    paths = setup_training_dirs(persona_name)
    slug = persona_name.lower().replace(" ", "-")
    trigger = trigger_word or f"ohwx {slug}"
    image_dir = paths["training_images"]

    # Try to use vision model for detailed captions
    vision_available = False
    if use_vision:
        try:
            import httpx
            resp = httpx.get("http://127.0.0.1:11434/api/tags", timeout=3)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                vision_available = any("llava" in m or "llama" in m for m in models)
        except Exception as exc:
            log.debug("Ollama vision check failed: %s", exc)
        if vision_available:
            log.info("Vision model available — generating detailed captions")
        else:
            log.info("Vision model not available — using generic captions")

    created = 0
    skipped = 0
    vision_used = 0
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    for img_file in sorted(image_dir.iterdir()):
        if img_file.suffix.lower() not in extensions:
            continue
        caption_file = img_file.with_suffix(".txt")
        if caption_file.exists() and not overwrite:
            skipped += 1
            continue

        caption = _vision_caption(img_file, trigger) if vision_available else None
        if caption:
            vision_used += 1
        else:
            caption = f"{trigger}, a photo of a person"

        caption_file.write_text(caption, encoding="utf-8")
        created += 1
        if created % 10 == 0:
            log.info("Captioned %d images so far...", created)

    log.info("Captions: created=%d, skipped=%d, vision=%d", created, skipped, vision_used)
    return {
        "created": created,
        "skipped": skipped,
        "vision_used": vision_used,
        "trigger_word": trigger,
        "note": "Review and edit caption .txt files for better results. "
                "Each image should have a matching .txt describing the scene."
                + (" Vision model generated detailed captions." if vision_used else ""),
    }


def _vision_caption(img_path: Path, trigger: str) -> str | None:
    """Use Ollama vision model to generate a detailed caption for a training image."""
    import base64
    try:
        import httpx
    except ImportError:
        return None

    try:
        img_bytes = img_path.read_bytes()
        img_b64 = base64.b64encode(img_bytes).decode()

        prompt = (
            "Describe this photo for Stable Diffusion training in one detailed sentence. "
            "Include: hair color/style/length, eye color, skin tone, facial features "
            "(freckles, moles, dimples, wrinkles), expression, body type, clothing, pose, "
            "background/setting, lighting. Be specific and factual. "
            "Do NOT mention names. Start with: a woman"
        )

        resp = httpx.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": "llava",
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 200},
            },
            timeout=60,
        )
        if resp.status_code == 200:
            description = resp.json().get("response", "").strip()
            if description:
                # Clean up — remove quotes, "Sure!", meta-text
                description = description.strip('"\'')
                for prefix in ["Sure!", "Here's", "This photo shows", "The image shows"]:
                    if description.lower().startswith(prefix.lower()):
                        description = description[len(prefix):].lstrip(" :,")
                # Prepend trigger word
                return f"{trigger}, {description}"
    except Exception as e:
        log.debug("Vision caption failed for %s: %s", img_path.name, e)
    return None


def get_trigger_word(persona_name: str) -> str:
    """Get the trigger word for a persona's LoRA.

    Returns the trigger word from training config, or the default 'ohwx <name>'.
    """
    paths = setup_training_dirs(persona_name)
    config_path = paths["config"]
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        return config.get("trigger_word", f"ohwx {persona_name.lower()}")
    return f"ohwx {persona_name.lower()}"


def get_lora_path(persona_name: str) -> str | None:
    """Get the path to a trained LoRA for image generation.

    Looks for diffusers-format LoRA directories (containing unet/ subfolder)
    first, then falls back to loose .safetensors files.
    Returns None if no LoRA exists yet.
    """
    paths = setup_training_dirs(persona_name)
    output_dir = paths["output"]

    # Check for diffusers-format LoRA dirs (e.g., ann_lora/unet/)
    for subdir in sorted(output_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if subdir.is_dir() and (subdir / "unet").exists():
            log.info("Found diffusers-format LoRA: %s", subdir)
            return str(subdir)

    # Fallback: loose .safetensors files
    lora_files = sorted(output_dir.glob("*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True)
    if lora_files:
        return str(lora_files[0])
    return None


def train_lora(persona_name: str, on_progress: callable | None = None) -> dict:
    """Run LoRA training using diffusers' built-in training.

    This is a simplified training loop that works without kohya_ss.
    For production quality, use the generated config with kohya_ss GUI.

    Args:
        persona_name: Name of the persona to train.
        on_progress: Callback for progress updates: on_progress(step, total, loss).

    Returns dict with training results.
    """
    paths = setup_training_dirs(persona_name)
    config_path = paths["config"]

    if not config_path.exists():
        config = generate_training_config(persona_name)
    else:
        with open(config_path) as f:
            config = json.load(f)

    status = get_training_status(persona_name)
    if not status["ready_to_train"]:
        return {"error": status["recommendation"], "status": status}

    # Auto-caption if needed
    auto_caption_images(persona_name, config.get("trigger_word"))

    try:
        import importlib.util
        if not importlib.util.find_spec("torch") or not importlib.util.find_spec("diffusers"):
            raise ImportError("torch or diffusers not installed")
    except ImportError:
        return {
            "error": "Training requires: pip install -e '.[lora]'\n"
                     "You also need torch with CUDA support.",
            "alternative": "Use the generated config with kohya_ss GUI:\n"
                          f"  Config: {config_path}\n"
                          f"  Images: {paths['training_images']}",
        }

    # Run the actual training loop
    from vox.train import train_lora as run_training

    result = run_training(
        persona_name=persona_name,
        image_dir=str(paths["training_images"]),
        output_dir=str(paths["output"]),
        epochs=config.get("epochs", 3),
        learning_rate=config.get("learning_rate", 1e-4),
        lora_rank=config.get("network_rank", 32),
        lora_alpha=config.get("network_alpha", 16),
        resolution=config.get("resolution", 1024),
        batch_size=config.get("batch_size", 1),
        on_progress=on_progress,
    )
    return result
