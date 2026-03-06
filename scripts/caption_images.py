"""Auto-caption images using LLaVA 1.5 7B for LoRA training.

Generates detailed 'ohwx ann'-prefixed captions matching the sort_keep style:
  ohwx ann, Brunette, long wavy brown hair, green eyes, fair skin, round face,
  soft features, smiling, wearing black top, standing in kitchen...

Uses LLaVA 1.5 7B (~15GB VRAM) for rich multi-sentence descriptions.
Falls back to BLIP-2 6.7B (~14GB) if LLaVA unavailable.

Usage:
    python scripts/caption_images.py persona/training/ann/sort_maybe
    python scripts/caption_images.py persona/training/ann/sort_maybe --overwrite
    python scripts/caption_images.py persona/training/ann/sort_maybe --model blip2
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path

import torch
from PIL import Image

log = logging.getLogger(__name__)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

LLAVA_PROMPT = (
    "Describe this photo in detail for training an image generation model. "
    "Include: hair color/style/length, eye color, skin tone, face shape, "
    "facial expression, body type, clothing (color, style, fit), accessories, "
    "pose, setting/background, lighting. Be specific and use comma-separated tags. "
    "Start with physical features, then clothing, then setting."
)


def _load_llava(device: str):
    """Load LLaVA 1.5 7B."""
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    model_id = "llava-hf/llava-1.5-7b-hf"
    log.info("Loading LLaVA 1.5 7B: %s (~15GB VRAM)", model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(device)
    return model, processor, "llava"


def _load_blip2(device: str):
    """Load BLIP-2 OPT-6.7B as fallback."""
    from transformers import AutoProcessor, Blip2ForConditionalGeneration

    model_id = "Salesforce/blip2-opt-6.7b"
    log.info("Loading BLIP-2 OPT-6.7B: %s (~14GB VRAM)", model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(device)
    return model, processor, "blip2"


def _caption_one_llava(model, processor, image: Image.Image, device: str) -> str:
    """Generate a detailed caption using LLaVA."""
    conversation = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": LLAVA_PROMPT},
        ]},
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            temperature=1.0,
        )

    # Decode only the new tokens (skip the prompt)
    generated = output[0][inputs["input_ids"].shape[1]:]
    caption = processor.decode(generated, skip_special_tokens=True).strip()
    return caption


def _caption_one_blip2(model, processor, image: Image.Image, device: str) -> str:
    """Generate a detailed caption using BLIP-2."""
    prompt = (
        "Question: Describe this person in detail including hair color and style, "
        "eye color, skin tone, face shape, expression, clothing, pose, and background. "
        "Answer:"
    )
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=150,
            num_beams=3,
            repetition_penalty=1.5,
        )

    caption = processor.batch_decode(output, skip_special_tokens=True)[0].strip()
    return caption


def caption_directory(
    image_dir: Path,
    trigger: str = "ohwx ann",
    model_choice: str = "llava",
    overwrite: bool = False,
    device: str | None = None,
) -> dict:
    """Generate detailed captions for images in a directory."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Find images to caption
    images_to_caption = []
    skipped = 0
    for f in sorted(image_dir.iterdir()):
        if f.suffix.lower() not in IMG_EXTS:
            continue
        caption_path = f.with_suffix(".txt")
        if caption_path.exists() and not overwrite:
            skipped += 1
            continue
        images_to_caption.append(f)

    if not images_to_caption:
        log.info("No images to caption in %s (%d already captioned)", image_dir, skipped)
        return {"captioned": 0, "skipped": skipped, "errors": 0}

    log.info("Found %d images to caption in %s (%d already done)", len(images_to_caption), image_dir, skipped)

    # Load model
    try:
        if model_choice == "llava":
            model, processor, model_type = _load_llava(device)
        else:
            model, processor, model_type = _load_blip2(device)
    except Exception as e:
        log.warning("Failed to load %s: %s — trying fallback", model_choice, e)
        if model_choice == "llava":
            model, processor, model_type = _load_blip2(device)
        else:
            model, processor, model_type = _load_llava(device)

    caption_fn = _caption_one_llava if model_type == "llava" else _caption_one_blip2

    captioned = 0
    errors = 0

    for img_path in images_to_caption:
        try:
            img = Image.open(img_path).convert("RGB")
            # Resize large images to save VRAM during captioning
            max_dim = 1024
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)

            raw_caption = caption_fn(model, processor, img, device)

            # Clean up caption
            caption = raw_caption.strip()
            # Remove common LLaVA preamble
            for preamble in ("The image shows ", "In this photo, ", "This is a photo of ", "The photo shows "):
                if caption.startswith(preamble):
                    caption = caption[len(preamble):]
                    caption = caption[0].upper() + caption[1:] if caption else caption
                    break

            full_caption = f"{trigger}, {caption}"
            img_path.with_suffix(".txt").write_text(full_caption, encoding="utf-8")
            captioned += 1

            if captioned % 10 == 0:
                log.info("  [%d/%d] %s", captioned, len(images_to_caption), img_path.name)

        except Exception as e:
            log.warning("Error captioning %s: %s", img_path.name, e)
            errors += 1

    log.info("Done: %d captioned, %d errors, %d skipped", captioned, errors, skipped)

    # Cleanup
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    return {"captioned": captioned, "skipped": skipped, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description="Auto-caption images for LoRA training (LLaVA/BLIP-2)")
    parser.add_argument("image_dir", type=Path, help="Directory with images to caption")
    parser.add_argument("--trigger", default="ohwx ann", help="Trigger word prefix (default: 'ohwx ann')")
    parser.add_argument("--model", choices=["llava", "blip2"], default="llava", help="Vision model (default: llava)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing captions")
    parser.add_argument("--device", default=None, help="Device (default: auto)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not args.image_dir.exists():
        print(f"ERROR: Directory not found: {args.image_dir}")
        sys.exit(1)

    result = caption_directory(
        args.image_dir,
        trigger=args.trigger,
        model_choice=args.model,
        overwrite=args.overwrite,
        device=args.device,
    )
    print(f"\nResults: {result['captioned']} captioned, {result['skipped']} skipped, {result['errors']} errors")


if __name__ == "__main__":
    main()
