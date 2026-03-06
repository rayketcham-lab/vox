"""LoRA training loop — fine-tune SDXL on reference photos using diffusers + peft.

Trains a LoRA adapter on the UNet (and optionally text encoders) so that
the trigger word 'ohwx <name>' generates images of the target person.

Hardware: RTX 3090 24GB — uses fp16, gradient checkpointing, 8-bit Adam
to fit SDXL training in VRAM.

Usage:
    python -m vox.train --persona ann --epochs 3
    python -m vox.train --persona ann --resume  # resume from last checkpoint
"""

from __future__ import annotations

import gc
import json
import logging
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LoRADataset(Dataset):
    """Image + caption dataset for LoRA training."""

    def __init__(
        self,
        image_dir: str | Path,
        resolution: int = 1024,
        caption_ext: str = ".txt",
        center_crop: bool = True,
        random_flip: bool = True,
    ):
        self.image_dir = Path(image_dir)
        self.resolution = resolution
        self.caption_ext = caption_ext

        # Find all image files that have matching caption files
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        self.image_paths = []
        self.captions = []

        for img_path in sorted(self.image_dir.iterdir()):
            if img_path.suffix.lower() not in extensions:
                continue
            caption_path = img_path.with_suffix(caption_ext)
            if not caption_path.exists():
                continue
            self.image_paths.append(img_path)
            self.captions.append(caption_path.read_text(encoding="utf-8").strip())

        log.info("Dataset: %d image-caption pairs from %s", len(self.image_paths), image_dir)

        self.transform = self._build_transform(resolution, center_crop, random_flip)

    @staticmethod
    def _build_transform(resolution, center_crop=True, random_flip=True):
        xforms = []
        if center_crop:
            xforms.append(transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR))
            xforms.append(transforms.CenterCrop(resolution))
        else:
            xforms.append(transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR))
        if random_flip:
            xforms.append(transforms.RandomHorizontalFlip())
        xforms.append(transforms.ToTensor())
        xforms.append(transforms.Normalize([0.5], [0.5]))  # [-1, 1]
        return transforms.Compose(xforms)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            pixel_values = self.transform(img)
            return {"pixel_values": pixel_values, "caption": self.captions[idx]}
        except Exception as e:
            log.error("FAILED loading image [%d]: %s — %s: %s", idx, img_path.name, type(e).__name__, e)
            # Return a black image + caption so training doesn't crash
            fallback = torch.zeros(3, self.resolution, self.resolution)
            return {"pixel_values": fallback, "caption": self.captions[idx]}


# ---------------------------------------------------------------------------
# Prior Preservation — regularization images
# ---------------------------------------------------------------------------

def generate_regularization_images(
    output_dir: str | Path,
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    class_prompt: str = "woman, portrait, photorealistic, natural lighting",
    num_images: int = 200,
    resolution: int = 1024,
    batch_size: int = 1,
    seed: int = 0,
) -> Path:
    """Generate regularization images from base SDXL (no LoRA).

    These images represent the model's prior knowledge of 'woman' — training
    with them prevents the model from forgetting what a generic woman looks like,
    which forces the LoRA to only encode what's UNIQUE about the subject.
    """
    from diffusers import StableDiffusionXLPipeline

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if we already have enough
    existing = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))
    if len(existing) >= num_images:
        log.info("Prior preservation: %d images already exist in %s (need %d), skipping generation",
                 len(existing), output_dir, num_images)
        return output_dir

    need = num_images - len(existing)
    log.info("Generating %d regularization images (class: '%s')...", need, class_prompt)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16, variant="fp16",
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    generator = torch.Generator("cuda").manual_seed(seed)
    start_idx = len(existing)

    for i in range(need):
        img = pipe(
            prompt=class_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=resolution,
            height=resolution,
            generator=generator,
        ).images[0]

        img_path = output_dir / f"reg_{start_idx + i:04d}.jpg"
        img.save(img_path, quality=95)

        # Write matching caption
        caption_path = img_path.with_suffix(".txt")
        caption_path.write_text(class_prompt, encoding="utf-8")

        if (i + 1) % 10 == 0:
            log.info("  Generated %d/%d regularization images", i + 1, need)

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    log.info("Regularization images ready: %s (%d total)", output_dir, num_images)
    return output_dir


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_lora(
    persona_name: str,
    *,
    image_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    trigger_word: str | None = None,
    resolution: int = 1024,
    epochs: int = 3,
    batch_size: int = 1,
    gradient_accumulation: int = 4,
    learning_rate: float = 1e-4,
    lr_scheduler: str = "cosine",
    lr_warmup_ratio: float = 0.05,
    lora_rank: int = 32,
    lora_alpha: int = 16,
    train_text_encoder: bool = True,
    save_every_n_steps: int = 250,
    seed: int = 42,
    resume: bool = False,
    prior_preservation: bool = False,
    prior_loss_weight: float = 1.0,
    prior_class_prompt: str = "woman, portrait, photorealistic, natural lighting",
    prior_num_images: int = 200,
    on_progress: callable | None = None,
) -> dict:
    """Train a LoRA on SDXL for a persona.

    Args:
        persona_name: Name used for output files.
        image_dir: Directory with images + .txt captions.
        output_dir: Where to save LoRA checkpoints.
        base_model: HuggingFace model ID for SDXL base.
        trigger_word: Trigger token (default: 'ohwx <name>').
        resolution: Training resolution (1024 for SDXL).
        epochs: Number of passes over the dataset.
        batch_size: Images per step (1 for 24GB VRAM).
        gradient_accumulation: Effective batch = batch_size * this.
        learning_rate: Peak learning rate.
        lr_scheduler: 'cosine', 'constant', or 'linear'.
        lr_warmup_ratio: Fraction of steps for warmup.
        lora_rank: LoRA rank (higher = more capacity, more VRAM).
        lora_alpha: LoRA alpha scaling factor.
        train_text_encoder: Also train text encoder LoRA (better text association).
        save_every_n_steps: Checkpoint frequency.
        seed: Random seed for reproducibility.
        resume: Resume from latest checkpoint.
        on_progress: Callback(step, total_steps, loss, epoch).

    Returns:
        Dict with training results and output path.
    """
    from diffusers import AutoencoderKL, DDPMScheduler
    from peft import LoraConfig, get_peft_model
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    slug = persona_name.lower().replace(" ", "-")
    trigger = trigger_word or f"ohwx {slug}"

    # Resolve paths
    from vox.config import MODELS_DIR, PROJECT_ROOT
    if image_dir is None:
        # Prefer stage1_faces (curated, captioned) > combined > sort_keep > base
        base_training = PROJECT_ROOT / "persona" / "training" / slug
        stage1 = base_training / "stage1_faces"
        combined = base_training / "combined"
        keep = base_training / "sort_keep"
        if stage1.exists() and any(stage1.glob("*.txt")):
            image_dir = stage1
        elif combined.exists() and any(combined.glob("*.txt")):
            image_dir = combined
        elif keep.exists():
            image_dir = keep
        else:
            image_dir = base_training
    image_dir = Path(image_dir)

    if output_dir is None:
        output_dir = MODELS_DIR / "lora" / slug
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("LoRA Training: %s", persona_name)
    log.info("  Trigger: %s", trigger)
    log.info("  Images: %s", image_dir)
    log.info("  Output: %s", output_dir)
    log.info("  Model: %s", base_model)
    log.info("  Epochs: %d, LR: %s, Rank: %d", epochs, learning_rate, lora_rank)
    if prior_preservation:
        log.info("  Prior preservation: ON (weight=%.2f, %d reg images)", prior_loss_weight, prior_num_images)
    log.info("=" * 60)

    # Set seeds
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16

    # --- Load dataset ---
    # Pre-flight: validate all images before starting (catches truncated/corrupt files)
    log.info("Pre-flight: validating %d images...", len(list(image_dir.glob("*.txt"))))
    bad_images = []
    for img_path in sorted(image_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue
        try:
            with Image.open(img_path) as img:
                img.verify()  # Checks file integrity without fully loading
        except Exception as e:
            log.warning("BAD IMAGE: %s — %s: %s", img_path.name, type(e).__name__, e)
            bad_images.append((img_path.name, str(e)))
    if bad_images:
        log.warning("Found %d bad images (will use black fallback during training):", len(bad_images))
        for name, err in bad_images:
            log.warning("  %s: %s", name, err)
    else:
        log.info("Pre-flight: all images valid")

    dataset = LoRADataset(image_dir, resolution=resolution)
    if len(dataset) == 0:
        return {"error": f"No image-caption pairs found in {image_dir}"}

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=True,
    )

    # --- Prior preservation: generate + load regularization images ---
    prior_dataloader = None
    if prior_preservation:
        reg_dir = output_dir / "regularization"
        generate_regularization_images(
            output_dir=reg_dir,
            base_model=base_model,
            class_prompt=prior_class_prompt,
            num_images=prior_num_images,
            resolution=resolution,
            seed=seed + 1000,
        )
        prior_dataset = LoRADataset(reg_dir, resolution=resolution, random_flip=True)
        if len(prior_dataset) == 0:
            log.warning("Prior preservation: no reg images found, disabling")
            prior_preservation = False
        else:
            prior_dataloader = DataLoader(
                prior_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
            )
            log.info("Prior preservation: %d regularization images loaded", len(prior_dataset))

    total_steps = math.ceil(len(dataset) / batch_size) * epochs
    effective_steps = total_steps // gradient_accumulation
    log.info("Dataset: %d images, %d steps/epoch, %d total steps (%d effective)",
             len(dataset), math.ceil(len(dataset) / batch_size), total_steps, effective_steps)

    # --- Load SDXL components ---
    log.info("Loading SDXL components (this may download ~6.5GB on first run)...")

    # Load tokenizers
    tokenizer_1 = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer_2")

    # Load text encoders
    text_encoder_1 = CLIPTextModel.from_pretrained(
        base_model, subfolder="text_encoder", torch_dtype=weight_dtype
    ).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        base_model, subfolder="text_encoder_2", torch_dtype=weight_dtype
    ).to(device)

    # Load VAE in fp32 — SDXL VAE produces NaN in fp16 (known issue)
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=torch.float32).to(device)

    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet", torch_dtype=weight_dtype).to(device)

    noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

    # Freeze everything first
    vae.requires_grad_(False)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)

    # Enable gradient checkpointing to save VRAM
    unet.enable_gradient_checkpointing()
    if train_text_encoder:
        text_encoder_1.gradient_checkpointing_enable()

    # --- Apply LoRA to UNet ---
    unet_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=[
            "to_k", "to_q", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "ff.net.0.proj", "ff.net.2",
        ],
    )
    unet = get_peft_model(unet, unet_lora_config)
    unet.print_trainable_parameters()

    # --- Optionally apply LoRA to text encoder 1 ---
    te_lora_config = None
    if train_text_encoder:
        te_lora_config = LoraConfig(
            r=lora_rank // 2,  # Smaller rank for text encoder
            lora_alpha=lora_alpha // 2,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_1 = get_peft_model(text_encoder_1, te_lora_config)
        text_encoder_1.print_trainable_parameters()

    # --- Optimizer ---
    trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    if train_text_encoder:
        trainable_params += list(filter(lambda p: p.requires_grad, text_encoder_1.parameters()))

    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(trainable_params, lr=learning_rate, weight_decay=1e-2)
        log.info("Using 8-bit AdamW optimizer (saves ~2GB VRAM)")
    except ImportError:
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-2)
        log.info("Using standard AdamW optimizer")

    # --- LR Scheduler ---
    warmup_steps = int(effective_steps * lr_warmup_ratio)
    if lr_scheduler == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=effective_steps - warmup_steps, eta_min=learning_rate * 0.1)
    else:
        scheduler = None

    # --- Resume from checkpoint ---
    start_step = 0
    start_epoch = 0
    if resume:
        # Load weights from final LoRA if it exists, else latest checkpoint
        final_lora = output_dir / f"{slug}_lora"
        checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))

        if (final_lora / "unet").exists():
            log.info("Loading weights from final LoRA: %s", final_lora)
            unet.load_adapter(str(final_lora / "unet"), adapter_name="default")
            if train_text_encoder and (final_lora / "text_encoder").exists():
                text_encoder_1.load_adapter(str(final_lora / "text_encoder"), adapter_name="default")
            log.info("Weights loaded — starting fresh epoch count (refinement mode)")
        elif checkpoints:
            latest_ckpt = checkpoints[-1]
            log.info("Resuming from %s", latest_ckpt)
            unet.load_adapter(str(latest_ckpt / "unet_lora"), adapter_name="default")
            if train_text_encoder and (latest_ckpt / "te1_lora").exists():
                text_encoder_1.load_adapter(str(latest_ckpt / "te1_lora"), adapter_name="default")
            log.info("Weights loaded — starting fresh epoch count (refinement mode)")
        # NOTE: We always reset step/epoch to 0 on resume. The --epochs flag means
        # "run N new epochs" from the loaded weights, not "continue to epoch N".
        log.info("Refinement: %d new epochs at LR %s", epochs, learning_rate)

    # --- Mixed precision scaler (prevents NaN in fp16) ---
    scaler = torch.amp.GradScaler("cuda")

    # --- Training loop ---
    log.info("Starting training...")
    global_step = start_step
    best_loss = float("inf")
    loss_history = []
    epoch_losses = []
    start_time = time.time()

    # Cast UNet trainable params to float32 for stable training
    for param in unet.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    if train_text_encoder:
        for param in text_encoder_1.parameters():
            if param.requires_grad:
                param.data = param.data.float()

    unet.train()
    if train_text_encoder:
        text_encoder_1.train()

    def _vram_mb():
        """Get current GPU VRAM usage in MB."""
        try:
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024
            return allocated, reserved
        except Exception:
            return 0.0, 0.0

    alloc, res = _vram_mb()
    log.info("VRAM before training: %.0fMB allocated, %.0fMB reserved (of 24576MB)", alloc, res)

    # Prior preservation: create an infinite iterator over reg images
    _prior_iter = None
    if prior_preservation and prior_dataloader is not None:
        def _cycle(dl):
            while True:
                yield from dl
        _prior_iter = iter(_cycle(prior_dataloader))

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for step, batch in enumerate(dataloader):
            # Skip already-done steps on resume
            absolute_step = epoch * math.ceil(len(dataset) / batch_size) + step
            if absolute_step < start_step:
                continue

            # Track which image we're on for crash diagnostics
            current_img_idx = step  # DataLoader batch index
            current_caption = batch["caption"][0][:80] if batch["caption"] else "?"

            try:
                pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
                captions = batch["caption"]

                # Encode images to latents in fp32 (VAE NaN in fp16)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Noise in float32
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Encode text
                tokens_1 = tokenizer_1(captions, padding="max_length", max_length=tokenizer_1.model_max_length, truncation=True, return_tensors="pt").to(device)
                tokens_2 = tokenizer_2(captions, padding="max_length", max_length=tokenizer_2.model_max_length, truncation=True, return_tensors="pt").to(device)

                # Text encoder forward
                if train_text_encoder:
                    encoder_hidden_states_1 = text_encoder_1(tokens_1.input_ids, output_hidden_states=True).hidden_states[-2]
                else:
                    with torch.no_grad():
                        encoder_hidden_states_1 = text_encoder_1(tokens_1.input_ids, output_hidden_states=True).hidden_states[-2]

                with torch.no_grad():
                    te2_output = text_encoder_2(tokens_2.input_ids, output_hidden_states=True)
                    encoder_hidden_states_2 = te2_output.hidden_states[-2]
                    pooled_output = te2_output[0]

                # Concatenate encoder hidden states (SDXL uses both)
                encoder_hidden_states = torch.cat([encoder_hidden_states_1, encoder_hidden_states_2], dim=-1)

                # SDXL additional conditioning
                add_time_ids = _compute_time_ids(resolution, resolution, 0, 0, resolution, resolution, device, torch.float32)
                add_time_ids = add_time_ids.repeat(latents.shape[0], 1)

                added_cond_kwargs = {
                    "text_embeds": pooled_output.float(),
                    "time_ids": add_time_ids,
                }

                # Forward pass — UNet weights are mixed (frozen fp16 + LoRA fp32)
                # Use autocast so frozen layers run fp16, LoRA layers run fp32
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    model_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states,
                        added_cond_kwargs=added_cond_kwargs,
                    ).sample

                # Loss in float32 (model_pred may be fp16 from autocast)
                subject_loss = F.mse_loss(model_pred.float(), noise, reduction="mean")
                loss = subject_loss

                # Prior preservation loss — regularization against base model drift
                if prior_preservation and _prior_iter is not None:
                    prior_batch = next(_prior_iter)
                    prior_pixels = prior_batch["pixel_values"].to(device, dtype=torch.float32)
                    prior_captions = prior_batch["caption"]

                    with torch.no_grad():
                        prior_latents = vae.encode(prior_pixels).latent_dist.sample()
                        prior_latents = prior_latents * vae.config.scaling_factor

                    prior_noise = torch.randn_like(prior_latents)
                    prior_timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (prior_latents.shape[0],), device=device,
                    ).long()
                    prior_noisy = noise_scheduler.add_noise(prior_latents, prior_noise, prior_timesteps)

                    # Encode prior captions
                    pt1 = tokenizer_1(prior_captions, padding="max_length", max_length=tokenizer_1.model_max_length, truncation=True, return_tensors="pt").to(device)
                    pt2 = tokenizer_2(prior_captions, padding="max_length", max_length=tokenizer_2.model_max_length, truncation=True, return_tensors="pt").to(device)

                    with torch.no_grad():
                        phs1 = text_encoder_1(pt1.input_ids, output_hidden_states=True).hidden_states[-2]
                        pte2 = text_encoder_2(pt2.input_ids, output_hidden_states=True)
                        phs2 = pte2.hidden_states[-2]
                        p_pooled = pte2[0]

                    prior_hidden = torch.cat([phs1, phs2], dim=-1)
                    prior_add_time_ids = _compute_time_ids(resolution, resolution, 0, 0, resolution, resolution, device, torch.float32)
                    prior_add_time_ids = prior_add_time_ids.repeat(prior_latents.shape[0], 1)
                    prior_cond = {"text_embeds": p_pooled.float(), "time_ids": prior_add_time_ids}

                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        prior_pred = unet(prior_noisy, prior_timesteps, prior_hidden, added_cond_kwargs=prior_cond).sample

                    prior_loss = F.mse_loss(prior_pred.float(), prior_noise, reduction="mean")
                    loss = subject_loss + prior_loss_weight * prior_loss

                loss = loss / gradient_accumulation

                # Backward with scaler
                scaler.scale(loss).backward()

                loss_val = loss.item() * gradient_accumulation
                if not math.isnan(loss_val):
                    epoch_loss += loss_val
                epoch_steps += 1
                global_step += 1

                # Gradient accumulation step
                if global_step % gradient_accumulation == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    if scheduler:
                        scheduler.step()
                    optimizer.zero_grad()

                # Logging (every 10 steps, with VRAM)
                if global_step % 10 == 0:
                    avg_loss = epoch_loss / max(epoch_steps, 1)
                    elapsed = time.time() - start_time
                    steps_per_sec = global_step / max(elapsed, 1)
                    remaining = (total_steps - global_step) / max(steps_per_sec, 0.001)
                    current_lr = optimizer.param_groups[0]["lr"]
                    alloc, res = _vram_mb()

                    log.info(
                        "Epoch %d/%d | Step %d/%d | Loss: %.4f | LR: %.2e | %.1f steps/s | ETA: %s | VRAM: %.0f/%.0fMB",
                        epoch + 1, epochs, global_step, total_steps, avg_loss,
                        current_lr, steps_per_sec, _format_time(remaining), alloc, res,
                    )
                    loss_history.append({"step": global_step, "loss": avg_loss, "lr": current_lr})

                    if on_progress:
                        on_progress(global_step, total_steps, avg_loss, epoch + 1)

                # Save checkpoint
                if global_step % save_every_n_steps == 0:
                    _save_checkpoint(
                        output_dir, global_step, unet, text_encoder_1 if train_text_encoder else None,
                        optimizer, loss_history, slug,
                    )

                # Track best loss
                if not math.isnan(loss_val) and loss_val < best_loss:
                    best_loss = loss_val

            except torch.cuda.OutOfMemoryError:
                alloc, res = _vram_mb()
                log.error(
                    "!!! OOM CRASH at epoch %d, step %d (global %d) | VRAM: %.0fMB/%.0fMB | Caption: %s",
                    epoch + 1, step, global_step, alloc, res, current_caption,
                )
                log.error("Clearing CUDA cache and skipping this batch...")
                torch.cuda.empty_cache()
                gc.collect()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                epoch_steps += 1
                continue

            except Exception as e:
                log.error(
                    "!!! CRASH at epoch %d, step %d (global %d) | %s: %s | Caption: %s",
                    epoch + 1, step, global_step, type(e).__name__, e, current_caption,
                )
                import traceback
                log.error("Full traceback:\n%s", traceback.format_exc())
                raise

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        epoch_losses.append(avg_epoch_loss)
        log.info("Epoch %d complete — avg loss: %.4f", epoch + 1, avg_epoch_loss)

    # --- Save final LoRA ---
    final_path = output_dir / f"{slug}_lora.safetensors"
    _save_final_lora(output_dir, unet, text_encoder_1 if train_text_encoder else None, slug)

    elapsed = time.time() - start_time
    log.info("=" * 60)
    log.info("Training complete in %s", _format_time(elapsed))
    log.info("Best loss: %.4f", best_loss)
    log.info("Final LoRA: %s", output_dir)
    log.info("=" * 60)

    # Save training log
    training_log = {
        "persona_name": persona_name,
        "trigger_word": trigger,
        "base_model": base_model,
        "epochs": epochs,
        "total_steps": global_step,
        "best_loss": best_loss,
        "epoch_losses": epoch_losses,
        "loss_history": loss_history[-50:],  # last 50 entries
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "learning_rate": learning_rate,
        "resolution": resolution,
        "image_count": len(dataset),
        "training_time_seconds": elapsed,
        "train_text_encoder": train_text_encoder,
        "prior_preservation": prior_preservation,
        "prior_loss_weight": prior_loss_weight if prior_preservation else 0,
    }
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    # Cleanup
    del unet, text_encoder_1, text_encoder_2, vae, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "status": "complete",
        "output_dir": str(output_dir),
        "lora_file": str(output_dir / f"{slug}_lora"),
        "trigger_word": trigger,
        "epochs": epochs,
        "total_steps": global_step,
        "best_loss": best_loss,
        "epoch_losses": epoch_losses,
        "training_time": _format_time(elapsed),
        "image_count": len(dataset),
    }


def _compute_time_ids(
    original_h: int, original_w: int,
    crop_top: int, crop_left: int,
    target_h: int, target_w: int,
    device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Compute SDXL micro-conditioning time_ids."""
    return torch.tensor(
        [original_h, original_w, crop_top, crop_left, target_h, target_w],
        dtype=dtype, device=device,
    ).unsqueeze(0)


def _save_checkpoint(
    output_dir: Path, step: int,
    unet, text_encoder_1,
    optimizer, loss_history: list,
    slug: str,
):
    """Save a training checkpoint."""
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA weights
    unet.save_pretrained(str(ckpt_dir / "unet_lora"))
    if text_encoder_1 is not None:
        text_encoder_1.save_pretrained(str(ckpt_dir / "te1_lora"))

    # Save training state
    state = {"step": step, "loss_history": loss_history[-20:]}
    with open(ckpt_dir / "training_state.json", "w") as f:
        json.dump(state, f, indent=2)

    log.info("Checkpoint saved: %s", ckpt_dir)


def _save_final_lora(output_dir: Path, unet, text_encoder_1, slug: str):
    """Save the final merged LoRA weights."""
    final_dir = output_dir / f"{slug}_lora"
    final_dir.mkdir(parents=True, exist_ok=True)

    unet.save_pretrained(str(final_dir / "unet"))
    if text_encoder_1 is not None:
        text_encoder_1.save_pretrained(str(final_dir / "text_encoder"))

    log.info("Final LoRA saved: %s", final_dir)


def _format_time(seconds: float) -> str:
    """Format seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train LoRA for VOX persona")
    parser.add_argument("--persona", required=True, help="Persona name (e.g., 'ann')")
    parser.add_argument("--image-dir", help="Override image directory")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank (default: 32)")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha (default: 16)")
    parser.add_argument("--resolution", type=int, default=1024, help="Training resolution (default: 1024)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation (default: 4)")
    parser.add_argument("--save-every", type=int, default=250, help="Save checkpoint every N steps")
    parser.add_argument("--no-text-encoder", action="store_true", help="Skip text encoder training")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--prior-preservation", action="store_true", help="Enable prior preservation (regularization)")
    parser.add_argument("--prior-weight", type=float, default=1.0, help="Prior preservation loss weight (default: 1.0)")
    parser.add_argument("--prior-class-prompt", default="woman, portrait, photorealistic, natural lighting",
                        help="Class prompt for regularization images")
    parser.add_argument("--prior-num-images", type=int, default=200, help="Number of regularization images (default: 200)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Force unbuffered output so crash logs are visible
    import sys
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
    sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    handler.flush = lambda: sys.stderr.flush()
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    def progress_cb(step, total, loss, epoch):
        pct = step / total * 100
        print(f"\r  [{pct:5.1f}%] Step {step}/{total} | Epoch {epoch} | Loss: {loss:.4f}", end="", flush=True)

    result = train_lora(
        persona_name=args.persona,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.lr,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        resolution=args.resolution,
        batch_size=args.batch_size,
        gradient_accumulation=args.grad_accum,
        save_every_n_steps=args.save_every,
        train_text_encoder=not args.no_text_encoder,
        resume=args.resume,
        prior_preservation=args.prior_preservation,
        prior_loss_weight=args.prior_weight,
        prior_class_prompt=args.prior_class_prompt,
        prior_num_images=args.prior_num_images,
        seed=args.seed,
        on_progress=progress_cb,
    )

    print("\n")
    if result.get("error"):
        print(f"ERROR: {result['error']}")
    else:
        print("Training complete!")
        print(f"  Output: {result['output_dir']}")
        print(f"  Trigger: {result['trigger_word']}")
        print(f"  Epochs: {result['epochs']}")
        print(f"  Best loss: {result['best_loss']:.4f}")
        print(f"  Epoch losses: {[f'{l:.4f}' for l in result['epoch_losses']]}")
        print(f"  Time: {result['training_time']}")
        print(f"\nTo use: add 'ohwx {args.persona}' to your image prompts")


if __name__ == "__main__":
    main()
