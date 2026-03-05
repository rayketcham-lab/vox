"""FastAPI backend for VOX web UI — WebSocket chat with streaming."""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

from vox.config import DOWNLOADS_DIR, WEB_HOST, WEB_PORT
from vox.llm import _history, chat

log = logging.getLogger(__name__)

app = FastAPI(title="VOX Web UI")

# Static files
_STATIC_DIR = Path(__file__).parent / "static"
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
    # Sanitize — only allow simple filenames, no path traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "Invalid filename"}, status_code=400)
    filepath = DOWNLOADS_DIR / filename
    if not filepath.exists() or not filepath.suffix == ".png":
        from fastapi.responses import JSONResponse
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
            if not user_text:
                continue

            lock = _session_locks[session_id]
            if lock.locked():
                await ws.send_json({"type": "busy"})
                continue

            async with lock:
                await _handle_chat(ws, session_id, user_text)
    except WebSocketDisconnect:
        log.info("WebSocket disconnected: session %s", session_id)
    finally:
        _sessions.pop(session_id, None)
        _session_locks.pop(session_id, None)


async def _handle_chat(ws: WebSocket, session_id: str, user_text: str):
    """Run chat() in a thread, streaming chunks back over WebSocket."""
    _SENTINEL = None
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_chunk(text: str):
        loop.call_soon_threadsafe(queue.put_nowait, text)

    def run_chat() -> str:
        session_history = _sessions[session_id]
        # Swap session history into the global _history
        _history.clear()
        _history.extend(session_history)
        try:
            result = chat(user_text, on_chunk=on_chunk)
        finally:
            # Snapshot back and signal completion
            _sessions[session_id] = list(_history)
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)
        return result

    # Start chat in background thread
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

    # Get the full response from the task
    try:
        full_response = await task
    except Exception:
        log.exception("Chat task failed")
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.send_json({"type": "done", "text": "Sorry, something went wrong."})
        return

    # Check for generated images in the response
    image_match = re.search(r"saved to (.+\.png)", full_response)
    if image_match:
        image_path = Path(image_match.group(1))
        filename = image_path.name
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
