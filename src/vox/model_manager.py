"""Modular model manager — load on demand, unload when done.

The RTX 3090 has 24GB VRAM. Instead of keeping every model loaded permanently,
this manager loads models when a tool needs them and unloads them after use.

Chat mode is lean: just LLM (via Ollama, external process) + STT (~2GB) + TTS (~2GB).
When a tool needs SDXL (~12GB), it loads, generates, and unloads back to lean mode.

Usage:
    with model_lease("sdxl") as pipe:
        image = pipe(prompt, ...).images[0]
    # SDXL automatically unloaded, VRAM freed

    # Or manual control:
    pipe = acquire("sdxl")
    image = pipe(prompt, ...).images[0]
    release("sdxl")
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from contextlib import contextmanager

log = logging.getLogger(__name__)

# Model registry: name → {loader, unloader, instance, last_used, vram_mb, keep_alive_sec}
_models: dict[str, dict] = {}
_lock = threading.Lock()

# Default: unload after 60 seconds of inactivity (0 = unload immediately after use)
DEFAULT_KEEP_ALIVE = 60


def register(
    name: str,
    loader: callable,
    vram_mb: int = 0,
    keep_alive: int = DEFAULT_KEEP_ALIVE,
):
    """Register a model that can be loaded on demand.

    Args:
        name: Unique model identifier (e.g., "sdxl", "sdxl:lora:ann")
        loader: Callable that returns the loaded model/pipeline
        vram_mb: Estimated VRAM usage in MB (for budgeting)
        keep_alive: Seconds to keep loaded after last use (0 = unload immediately)
    """
    with _lock:
        _models[name] = {
            "loader": loader,
            "instance": None,
            "vram_mb": vram_mb,
            "keep_alive": keep_alive,
            "last_used": 0,
            "loading": False,
        }


def acquire(name: str) -> object:
    """Load a model (or return cached instance) and mark it as in-use."""
    with _lock:
        entry = _models.get(name)
        if entry is None:
            raise KeyError(f"Unknown model: {name}. Register it first with model_manager.register()")

        if entry["instance"] is not None:
            entry["last_used"] = time.time()
            log.info("Model '%s' already loaded, reusing", name)
            return entry["instance"]

        entry["loading"] = True

    # Load outside lock (can be slow)
    log.info("Loading model '%s' (~%dMB VRAM)...", name, entry["vram_mb"])
    start = time.time()
    try:
        instance = entry["loader"]()
    except Exception:
        with _lock:
            entry["loading"] = False
        raise

    with _lock:
        entry["instance"] = instance
        entry["last_used"] = time.time()
        entry["loading"] = False

    elapsed = time.time() - start
    log.info("Model '%s' loaded in %.1fs", name, elapsed)
    return instance


def release(name: str, force: bool = False):
    """Release a model. If keep_alive > 0 and not force, keeps it cached temporarily."""
    with _lock:
        entry = _models.get(name)
        if entry is None or entry["instance"] is None:
            return

        if force or entry["keep_alive"] == 0:
            _unload(name, entry)
        else:
            entry["last_used"] = time.time()
            log.debug("Model '%s' marked for deferred unload (%ds keep-alive)",
                      name, entry["keep_alive"])


def _unload(name: str, entry: dict):
    """Actually unload a model and free VRAM. Must hold _lock."""
    instance = entry["instance"]
    entry["instance"] = None

    # Move to CPU first if it has a .to() method, then delete
    if hasattr(instance, "to"):
        try:
            instance.to("cpu")
        except Exception:
            log.debug("Failed to move model to CPU before unload")
    del instance
    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    log.info("Model '%s' unloaded (~%dMB VRAM freed)", name, entry["vram_mb"])


def release_all():
    """Force-unload all loaded models."""
    with _lock:
        for name, entry in _models.items():
            if entry["instance"] is not None:
                _unload(name, entry)


def cleanup_expired():
    """Unload models that have exceeded their keep_alive timeout."""
    now = time.time()
    with _lock:
        for name, entry in _models.items():
            if (entry["instance"] is not None
                    and entry["keep_alive"] > 0
                    and now - entry["last_used"] > entry["keep_alive"]):
                log.info("Model '%s' expired (idle %.0fs > %ds keep-alive)",
                         name, now - entry["last_used"], entry["keep_alive"])
                _unload(name, entry)


def status() -> dict[str, dict]:
    """Get status of all registered models."""
    with _lock:
        result = {}
        for name, entry in _models.items():
            result[name] = {
                "loaded": entry["instance"] is not None,
                "vram_mb": entry["vram_mb"],
                "keep_alive": entry["keep_alive"],
                "idle_seconds": int(time.time() - entry["last_used"]) if entry["last_used"] else None,
            }
        return result


def vram_loaded() -> int:
    """Total estimated VRAM used by loaded models (MB)."""
    with _lock:
        return sum(e["vram_mb"] for e in _models.values() if e["instance"] is not None)


@contextmanager
def model_lease(name: str, unload_after: bool = True):
    """Context manager: load model, yield it, then release.

    Args:
        name: Model name to acquire
        unload_after: If True, force-unload after use. If False, use keep_alive.
    """
    instance = acquire(name)
    try:
        yield instance
    finally:
        release(name, force=unload_after)
