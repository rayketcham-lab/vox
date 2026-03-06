"""FastAPI backend for VOX web UI — WebSocket chat with streaming."""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import uuid
from pathlib import Path

import hmac
import secrets

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.websockets import WebSocketState

from vox.config import DOWNLOADS_DIR, WEB_AUTH_PASS, WEB_AUTH_USER, WEB_HOST, WEB_PORT
from vox.llm import _history, chat, chat_with_vision

log = logging.getLogger(__name__)

app = FastAPI(title="VOX Web UI")


# --- HTTP Basic Auth middleware (optional — enabled when WEB_AUTH_USER is set) ---

class _BasicAuthMiddleware(BaseHTTPMiddleware):
    """Simple HTTP Basic Auth. Skips auth if WEB_AUTH_USER is empty."""

    async def dispatch(self, request: Request, call_next):
        if not WEB_AUTH_USER:
            return await call_next(request)

        # Allow health check without auth
        if request.url.path == "/health":
            return await call_next(request)

        auth = request.headers.get("authorization", "")
        if auth.startswith("Basic "):
            try:
                decoded = base64.b64decode(auth[6:]).decode("utf-8")
                user, passwd = decoded.split(":", 1)
                if (hmac.compare_digest(user, WEB_AUTH_USER)
                        and hmac.compare_digest(passwd, WEB_AUTH_PASS)):
                    return await call_next(request)
            except Exception:
                pass

        return Response(
            "Unauthorized",
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="VOX"'},
        )


if WEB_AUTH_USER:
    app.add_middleware(_BasicAuthMiddleware)
    log.info("HTTP Basic Auth enabled (user: %s)", WEB_AUTH_USER)


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


def _check_ws_auth(ws: WebSocket) -> bool:
    """Validate Basic Auth on WebSocket upgrade if auth is enabled."""
    if not WEB_AUTH_USER:
        return True
    auth = ws.headers.get("authorization", "")
    if auth.startswith("Basic "):
        try:
            decoded = base64.b64decode(auth[6:]).decode("utf-8")
            user, passwd = decoded.split(":", 1)
            return (hmac.compare_digest(user, WEB_AUTH_USER)
                    and hmac.compare_digest(passwd, WEB_AUTH_PASS))
        except Exception:
            pass
    return False


@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    if not _check_ws_auth(ws):
        await ws.close(code=4001, reason="Unauthorized")
        return
    await ws.accept()

    session_id = str(uuid.uuid4())
    _sessions[session_id] = []
    _session_locks[session_id] = asyncio.Lock()

    await ws.send_json({"type": "session", "id": session_id})
    log.info("WebSocket connected: session %s", session_id)

    try:
        while True:
            data = await ws.receive_json()

            # Handle ping keep-alive + push pending notifications
            if data.get("type") == "ping":
                await ws.send_json({"type": "pong"})
                # Push any fired reminders
                while _pending_reminders:
                    reminder_msg = _pending_reminders.pop(0)
                    await ws.send_json({"type": "reminder", "text": reminder_msg})
                # Push any proactive messages from the persona
                while _pending_proactive:
                    proactive_msg = _pending_proactive.pop(0)
                    await ws.send_json({"type": "proactive", "text": proactive_msg})
                continue

            # Handle training image uploads
            if data.get("type") == "training_image":
                await _handle_training_upload(ws, data)
                continue

            if data.get("type") != "message":
                continue

            user_text = data.get("text", "").strip()[:4000]  # cap at 4K chars
            user_images = data.get("images", [])[:5]  # max 5 images per message

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
        # Keep session alive for 5 min so mobile reconnects don't lose history
        _session_locks.pop(session_id, None)
        asyncio.get_running_loop().call_later(300, _sessions.pop, session_id, None)


def _find_generated_images(history: list[dict], before_len: int) -> list[str]:
    """Scan new history entries for generated image/map files."""
    filenames = []
    for entry in history[before_len:]:
        if entry.get("role") == "tool":
            content = entry.get("content", "")
            # Match any "saved to <filename>.png" pattern (images, maps, etc.)
            for match in re.finditer(r"saved to (\S+\.png)", content):
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

    # Send uploaded image URLs back for confirmation
    if uploaded_paths and ws.client_state == WebSocketState.CONNECTED:
        await ws.send_json({"type": "uploaded_images", "urls": uploaded_paths})

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

    # Stream chunks as they arrive — buffer to strip drive-letter paths
    _chunk_buffer = ""
    try:
        while True:
            chunk = await queue.get()
            if chunk is _SENTINEL:
                # Flush remaining buffer — strip paths before sending
                if _chunk_buffer:
                    flushed = re.sub(r"[A-Z]:\\[\w\\]+\.\w+", "", _chunk_buffer)
                    flushed = re.sub(r"(?:saved?\s+(?:it\s+)?to|stored\s+(?:at|in)?)\s+\S+\.png", "", flushed, flags=re.IGNORECASE)
                    flushed = re.sub(r"\bvox_(?:image|map)_\S+\.png\b", "", flushed, flags=re.IGNORECASE)
                    flushed = re.sub(r"\s{2,}", " ", flushed)
                    if flushed.strip() and ws.client_state == WebSocketState.CONNECTED:
                        await ws.send_json({"type": "chunk", "text": flushed})
                break
            _chunk_buffer += chunk
            # Buffer until we have enough to detect/strip paths (drive letters need context)
            if len(_chunk_buffer) > 80 or chunk.endswith((".", "!", "?", "\n")):
                cleaned = re.sub(r"[A-Z]:\\[\w\\]+\.\w+", "", _chunk_buffer)
                cleaned = re.sub(r"(?:saved?\s+(?:it\s+)?to|stored\s+(?:at|in)?)\s+\S+\.png", "", cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r"\bvox_(?:image|map)_\S+\.png\b", "", cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r"\s{2,}", " ", cleaned)
                if cleaned.strip() and ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_json({"type": "chunk", "text": cleaned})
                _chunk_buffer = ""
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
        for match in re.finditer(r"saved to (\S+\.png)", full_response):
            image_path = Path(match.group(1))
            generated_images.append(image_path.name)

    # Send generated images inline BEFORE the done message
    for filename in generated_images:
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.send_json({
                "type": "image",
                "url": f"/downloads/{filename}",
                "filename": filename,
            })

    # Strip file paths from response text — the image is shown inline,
    # no need for the LLM to mention paths (it hallucinates D: paths anyway)
    clean_response = full_response
    # Always strip drive-letter paths and filenames from LLM output
    clean_response = re.sub(
        r"[A-Z]:\\[\w\\]+\.\w+\b", "", clean_response, flags=re.IGNORECASE,
    )
    clean_response = re.sub(
        r"(?:saved?\s+(?:it\s+)?to|(?:check|find|look)\s+(?:out|at|in)?)\s+\S*[\\/]\S+\.\w+\b",
        "", clean_response, flags=re.IGNORECASE,
    )
    clean_response = re.sub(
        r"(?:saved?\s+(?:it\s+)?to|stored\s+(?:it\s+)?(?:at|in)?)\s+\S+\.png\b",
        "", clean_response, flags=re.IGNORECASE,
    )
    clean_response = re.sub(
        r"\bvox_(?:image|map)_\S+\.png\b", "", clean_response, flags=re.IGNORECASE,
    )
    clean_response = re.sub(
        r"\s+for the image\.?", "", clean_response, flags=re.IGNORECASE,
    )
    clean_response = re.sub(r"\s{2,}", " ", clean_response).strip()

    if ws.client_state == WebSocketState.CONNECTED:
        await ws.send_json({"type": "done", "text": clean_response})


@app.on_event("startup")
async def _startup():
    """Load persona card and start background tasks on server startup."""
    from vox.config import VOX_PERSONA_CARD
    if VOX_PERSONA_CARD:
        from vox.persona import load_card
        load_card(VOX_PERSONA_CARD)
        log.info("Persona card loaded: %s", VOX_PERSONA_CARD)

    # Start background loops
    asyncio.create_task(_reminder_loop())
    asyncio.create_task(_proactive_loop())


async def _reminder_loop():
    """Background loop that checks for due reminders every 30 seconds."""
    from vox.reminders import check_and_fire

    while True:
        await asyncio.sleep(30)
        try:
            fired = check_and_fire()
            if fired:
                # Push to all connected WebSocket sessions
                for sid, history in list(_sessions.items()):
                    if sid.endswith("__images"):
                        continue
                    lock = _session_locks.get(sid)
                    if lock is None:
                        continue
                    # Find the WebSocket for this session — we'll broadcast
                    # via a stashed reference (see below)
                for msg in fired:
                    log.info("Reminder broadcast: %s", msg)
                    # Stash fired reminders for next WebSocket message
                    _pending_reminders.extend(fired)
        except Exception:
            log.exception("Reminder loop error")


# Pending reminder notifications to push on next WebSocket activity
_pending_reminders: list[str] = []

# Pending proactive messages (LLM-generated, pushed on next ping)
_pending_proactive: list[str] = []


async def _proactive_loop():
    """Background loop that generates proactive persona messages every ~60 seconds."""
    from vox.proactive import get_proactive_message

    await asyncio.sleep(120)  # Wait 2 min after startup before first check
    while True:
        await asyncio.sleep(60)
        try:
            prompt = get_proactive_message()
            if prompt and not _pending_proactive:
                # Generate via LLM in a thread (blocking call)
                log.info("Proactive message triggered")
                response = await asyncio.to_thread(chat, prompt)
                if response:
                    _pending_proactive.append(response)
                    log.info("Proactive message queued: %s", response[:100])
        except Exception:
            log.exception("Proactive loop error")


def start_server(host: str | None = None, port: int | None = None):
    """Start the uvicorn server."""
    import uvicorn

    uvicorn.run(
        app,
        host=host or WEB_HOST,
        port=port or WEB_PORT,
        log_level="info",
    )
