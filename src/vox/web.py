"""FastAPI backend for VOX web UI — WebSocket chat with streaming."""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

from vox.config import DOWNLOADS_DIR, WEB_HOST, WEB_PORT
from vox.llm import _history, chat, chat_with_vision

log = logging.getLogger(__name__)

app = FastAPI(title="VOX Web UI")

# Static files
_STATIC_DIR = Path(__file__).parent / "static"
_UPLOADS_DIR = DOWNLOADS_DIR / "uploads"
_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
app.mount("/downloads", StaticFiles(directory=str(DOWNLOADS_DIR)), name="downloads")

# Per-session state
_sessions: dict[str, list[dict]] = {}
_session_locks: dict[str, asyncio.Lock] = {}


@app.get("/")
async def index():
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get("/api/images")
async def list_images():
    """Return list of PNGs in downloads/ sorted newest-first."""
    images = sorted(DOWNLOADS_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"filename": img.name, "url": f"/downloads/{img.name}"} for img in images]


@app.delete("/api/images/{filename}")
async def delete_image(filename: str):
    """Delete a single image from downloads/."""
    if "/" in filename or "\\" in filename or ".." in filename:
        return JSONResponse({"error": "Invalid filename"}, status_code=400)
    filepath = DOWNLOADS_DIR / filename
    if not filepath.exists() or not filepath.suffix == ".png":
        return JSONResponse({"error": "Image not found"}, status_code=404)
    filepath.unlink()
    log.info("Deleted image: %s", filename)
    return {"deleted": filename}


# --- LoRA Training Endpoints ---


@app.get("/api/lora/status")
async def lora_status():
    """Get LoRA training status for the current persona."""
    from vox.config import VOX_PERSONA_NAME
    from vox.lora import get_training_status
    from vox.persona import get_card

    card = get_card()
    name = card["name"] if card else VOX_PERSONA_NAME or "vox"
    return get_training_status(name)


@app.post("/api/lora/upload")
async def lora_upload_images():
    """Upload training images via multipart form."""
    # WebSocket handles uploads — see _handle_training_upload
    return JSONResponse(
        {"error": "Use WebSocket upload — send type: 'training_image' with base64 data"},
        status_code=501,
    )


@app.post("/api/lora/caption")
async def lora_auto_caption():
    """Auto-generate captions for training images."""
    from vox.config import VOX_PERSONA_NAME
    from vox.lora import auto_caption_images
    from vox.persona import get_card

    card = get_card()
    name = card["name"] if card else VOX_PERSONA_NAME or "vox"
    return auto_caption_images(name)


@app.get("/api/lora/config")
async def lora_get_config():
    """Generate/get training config."""
    from vox.config import VOX_PERSONA_NAME
    from vox.lora import generate_training_config
    from vox.persona import get_card

    card = get_card()
    name = card["name"] if card else VOX_PERSONA_NAME or "vox"
    return generate_training_config(name)


@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    await ws.accept()

    session_id = str(uuid.uuid4())
    _sessions[session_id] = []
    _session_locks[session_id] = asyncio.Lock()

    await ws.send_json({"type": "session", "id": session_id})
    log.info("WebSocket connected: session %s", session_id)

    try:
        while True:
            data = await ws.receive_json()

            # Handle training image uploads
            if data.get("type") == "training_image":
                await _handle_training_upload(ws, data)
                continue

            if data.get("type") != "message":
                continue

            user_text = data.get("text", "").strip()
            user_images = data.get("images", [])  # list of base64-encoded images

            if not user_text and not user_images:
                continue

            lock = _session_locks[session_id]
            if lock.locked():
                await ws.send_json({"type": "busy"})
                continue

            async with lock:
                await _handle_chat(ws, session_id, user_text, user_images)
    except WebSocketDisconnect:
        log.info("WebSocket disconnected: session %s", session_id)
    finally:
        _sessions.pop(session_id, None)
        _session_locks.pop(session_id, None)


def _find_generated_images(history: list[dict], before_len: int) -> list[str]:
    """Scan new history entries for generated image paths."""
    filenames = []
    for entry in history[before_len:]:
        if entry.get("role") == "tool":
            content = entry.get("content", "")
            match = re.search(r"saved to (.+\.png)", content)
            if match:
                image_path = Path(match.group(1))
                filenames.append(image_path.name)
    return filenames


async def _handle_training_upload(ws: WebSocket, data: dict):
    """Handle training image upload via WebSocket."""
    from vox.config import VOX_PERSONA_NAME
    from vox.lora import get_training_status, setup_training_dirs
    from vox.persona import get_card

    card = get_card()
    name = card["name"] if card else VOX_PERSONA_NAME or "vox"

    images_b64 = data.get("images", [])
    if not images_b64:
        await ws.send_json({"type": "training_status", "error": "No images provided"})
        return

    paths = setup_training_dirs(name)
    saved_paths = []
    for i, img_b64 in enumerate(images_b64):
        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]
        try:
            img_bytes = base64.b64decode(img_b64)
            import datetime as _dt
            ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"ref_{ts}_{i}.png"
            fpath = paths["training_images"] / fname
            fpath.write_bytes(img_bytes)
            saved_paths.append(str(fpath))
            log.info("Saved training image: %s (%d bytes)", fpath, len(img_bytes))
        except Exception:
            log.exception("Failed to save training image")

    status = get_training_status(name)
    await ws.send_json({"type": "training_status", **status})


async def _handle_chat(ws: WebSocket, session_id: str, user_text: str, user_images: list[str] | None = None):
    """Run chat() in a thread, streaming chunks back over WebSocket."""
    _SENTINEL = None
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_chunk(text: str):
        loop.call_soon_threadsafe(queue.put_nowait, text)

    # Save uploaded images to disk and get paths for display
    uploaded_paths: list[str] = []
    if user_images:
        for i, img_b64 in enumerate(user_images):
            # Strip data URL prefix if present
            if "," in img_b64:
                img_b64 = img_b64.split(",", 1)[1]
            try:
                img_bytes = base64.b64decode(img_b64)
                import datetime
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"upload_{ts}_{i}.png"
                fpath = _UPLOADS_DIR / fname
                fpath.write_bytes(img_bytes)
                uploaded_paths.append(f"/downloads/uploads/{fname}")
                log.info("Saved uploaded image: %s (%d bytes)", fpath, len(img_bytes))
            except Exception:
                log.exception("Failed to save uploaded image")

    def run_chat() -> str:
        session_history = _sessions[session_id]
        _history.clear()
        _history.extend(session_history)
        history_len_before = len(_history)
        try:
            if user_images:
                result = chat_with_vision(
                    user_text or "What's in this image?",
                    images=user_images,
                    on_chunk=on_chunk,
                )
            else:
                result = chat(user_text, on_chunk=on_chunk)
        finally:
            # Find any generated images in new history entries
            generated = _find_generated_images(_history, history_len_before)
            _sessions[session_id] = list(_history)
            # Stash generated filenames for the caller
            _sessions[f"{session_id}__images"] = generated
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)
        return result

    task = asyncio.get_running_loop().run_in_executor(None, run_chat)

    # Stream chunks as they arrive
    try:
        while True:
            chunk = await queue.get()
            if chunk is _SENTINEL:
                break
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.send_json({"type": "chunk", "text": chunk})
    except Exception:
        log.exception("Error streaming chat response")

    try:
        full_response = await task
    except Exception:
        log.exception("Chat task failed")
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.send_json({"type": "done", "text": "Sorry, something went wrong."})
        return

    # Check for generated images — first from session history (reliable),
    # then fallback to regex on the response text
    generated_images = _sessions.pop(f"{session_id}__images", [])
    if not generated_images:
        # Fallback: check the response text
        for match in re.finditer(r"saved to (.+\.png)", full_response):
            image_path = Path(match.group(1))
            generated_images.append(image_path.name)

    for filename in generated_images:
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.send_json({
                "type": "image",
                "url": f"/downloads/{filename}",
                "filename": filename,
            })

    if ws.client_state == WebSocketState.CONNECTED:
        await ws.send_json({"type": "done", "text": full_response})


def start_server(host: str | None = None, port: int | None = None):
    """Start the uvicorn server."""
    import uvicorn

    uvicorn.run(
        app,
        host=host or WEB_HOST,
        port=port or WEB_PORT,
        log_level="info",
    )
