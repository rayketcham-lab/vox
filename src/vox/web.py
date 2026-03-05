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
