"""Vector memory — semantic search over conversation history and facts.

Tiered memory architecture (inspired by MemGPT/Letta):
  1. Core memory: Explicit facts ("remember X") — always in context (existing memory.py)
  2. Recall memory: Recent conversation turns — sliding window in prompt
  3. Archival memory: Full conversation history in ChromaDB — semantic search

This module handles archival memory. On each exchange, we store the user message
and assistant response. When building context, we retrieve the most relevant
past exchanges for the current conversation.

Storage: data/vector_memory/ (ChromaDB persistent, gitignored)
Embedding: all-MiniLM-L6-v2 (22M params, ~80MB, runs on CPU)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)

_VECTOR_DIR = Path(__file__).parent.parent.parent / "data" / "vector_memory"
_collection = None
_embed_fn = None
_initialized = False

# Track recent stores for deduplication (message_hash -> timestamp)
_recent_stores: dict[int, float] = {}
_DEDUP_WINDOW = 60  # seconds — ignore same message within this window


def _is_recent_duplicate(message: str) -> bool:
    """Check if we already stored a very similar message recently."""
    msg_hash = hash(message.strip().lower()[:200])
    now = time.time()
    # Clean old entries
    expired = [k for k, v in _recent_stores.items() if now - v > _DEDUP_WINDOW]
    for k in expired:
        del _recent_stores[k]
    if msg_hash in _recent_stores:
        return True
    _recent_stores[msg_hash] = now
    return False


def _ensure_init():
    """Lazy-initialize ChromaDB and embedding model on first use."""
    global _collection, _embed_fn, _initialized
    if _initialized:
        return

    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        _VECTOR_DIR.mkdir(parents=True, exist_ok=True)

        _embed_fn = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            device="cpu",  # Small model, CPU is fine — saves GPU for LLM/TTS
        )

        client = chromadb.PersistentClient(path=str(_VECTOR_DIR))
        _collection = client.get_or_create_collection(
            name="conversations",
            embedding_function=_embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

        log.info("Vector memory initialized: %d entries in %s", _collection.count(), _VECTOR_DIR)
        _initialized = True

    except ImportError as e:
        log.warning("Vector memory unavailable (missing dependency): %s", e)
        _initialized = True  # Don't retry
    except Exception as e:
        log.warning("Vector memory init failed: %s", e)


def store(user_message: str, assistant_response: str, metadata: dict | None = None):
    """Store a conversation exchange in vector memory.

    Skips near-duplicate messages (same user message within 60 seconds).
    """
    _ensure_init()
    if _collection is None:
        return

    # Dedup: skip if we stored a very similar message recently
    if _is_recent_duplicate(user_message):
        log.debug("Skipping duplicate vector memory entry: %s", user_message[:50])
        return

    # Combine user + assistant for richer embedding
    combined = f"User: {user_message}\nAssistant: {assistant_response}"

    # Use timestamp as ID for uniqueness
    doc_id = f"conv_{int(time.time() * 1000)}"

    meta = {
        "user_message": user_message[:500],  # Truncate for metadata storage limits
        "assistant_response": assistant_response[:500],
        "timestamp": int(time.time()),
        "type": "conversation",
    }
    if metadata:
        meta.update(metadata)

    try:
        _collection.add(
            ids=[doc_id],
            documents=[combined],
            metadatas=[meta],
        )
    except Exception as e:
        log.warning("Failed to store in vector memory: %s", e)


def store_fact(fact: str, category: str = "general"):
    """Store an explicit fact in vector memory (in addition to JSON core memory)."""
    _ensure_init()
    if _collection is None:
        return

    doc_id = f"fact_{int(time.time() * 1000)}"
    try:
        _collection.add(
            ids=[doc_id],
            documents=[fact],
            metadatas=[{
                "type": "fact",
                "category": category,
                "timestamp": int(time.time()),
            }],
        )
    except Exception as e:
        log.warning("Failed to store fact in vector memory: %s", e)


def search(query: str, n_results: int = 5, type_filter: str | None = None) -> list[dict]:
    """Semantic search over conversation history and facts.

    Args:
        query: Natural language query
        n_results: Max results to return
        type_filter: Optional "conversation" or "fact"

    Returns:
        List of {document, metadata, distance} dicts, sorted by relevance.
    """
    _ensure_init()
    if _collection is None or _collection.count() == 0:
        return []

    try:
        where = {"type": type_filter} if type_filter else None
        results = _collection.query(
            query_texts=[query],
            n_results=min(n_results, _collection.count()),
            where=where,
        )

        hits = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                hits.append({
                    "document": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                })
        return hits

    except Exception as e:
        log.warning("Vector search failed: %s", e)
        return []


def _time_decay_score(distance: float, timestamp: int) -> float:
    """Combine cosine distance with time decay — recent results rank higher.

    Returns a score where LOWER is better (like cosine distance).
    Time decay: halves the recency bonus every 7 days.
    """
    age_seconds = max(0, time.time() - timestamp)
    age_days = age_seconds / 86400
    # Recency bonus: 0.0 (today) to ~0.2 (very old), subtracted from distance
    recency_bonus = 0.2 * (0.5 ** (age_days / 7))
    return distance - recency_bonus


def build_context_block(current_message: str, max_entries: int = 5) -> str:
    """Build a prompt block with relevant past context for the current message.

    Only retrieves if the message is substantive (>10 chars) and there's
    history to search. Returns empty string if nothing relevant found.

    Uses time-decay scoring so recent conversations rank higher than old ones.
    """
    if len(current_message.strip()) < 10:
        return ""

    _ensure_init()
    if _collection is None or _collection.count() < 3:
        return ""

    # Fetch more than we need, then re-rank with time decay
    hits = search(current_message, n_results=max_entries * 2)
    if not hits:
        return ""

    # Filter by relevance — cosine distance < 0.8 means reasonably similar
    relevant = [h for h in hits if h["distance"] < 0.8]
    if not relevant:
        return ""

    # Re-rank with time decay (lower score = better)
    for h in relevant:
        ts = h["metadata"].get("timestamp", 0)
        h["_score"] = _time_decay_score(h["distance"], ts)
    relevant.sort(key=lambda h: h["_score"])

    lines = []
    for h in relevant[:max_entries]:
        meta = h["metadata"]
        if meta.get("type") == "conversation":
            user_msg = meta.get("user_message", "")
            asst_msg = meta.get("assistant_response", "")
            if user_msg and asst_msg:
                lines.append(f"- Previously discussed: \"{user_msg[:150]}\" → \"{asst_msg[:150]}\"")
        elif meta.get("type") == "fact":
            lines.append(f"- Known fact: {h['document'][:200]}")

    if not lines:
        return ""

    return (
        "\nRelevant context from past conversations (reference naturally if useful):\n"
        + "\n".join(lines)
    )


def count() -> int:
    """Number of entries in vector memory."""
    _ensure_init()
    return _collection.count() if _collection else 0


def clear():
    """Clear all vector memory (for testing/reset)."""
    _ensure_init()
    if _collection is None:
        return
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(_VECTOR_DIR))
        client.delete_collection("conversations")
        global _initialized
        _initialized = False
        log.info("Vector memory cleared")
    except Exception as e:
        log.warning("Failed to clear vector memory: %s", e)
