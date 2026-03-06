"""RAG — Retrieval-Augmented Generation for local documents.

Index PDFs, text files, and code from a configurable directory.
Uses ChromaDB for vector storage and sentence-transformers for embeddings.
Retrieves relevant chunks and feeds them to the LLM as context.

Depends on: pip install -e ".[memory]" (chromadb, sentence-transformers)
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from pathlib import Path

log = logging.getLogger(__name__)

_DOCS_DIR = Path(os.environ.get("RAG_DOCS_DIR", "")).expanduser() if os.environ.get("RAG_DOCS_DIR") else None
_CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "500"))
_CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "50"))
_COLLECTION_NAME = "documents"

_collection = None
_embed_fn = None
_initialized = False

# Supported file extensions
_TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".csv", ".json", ".yaml", ".yml", ".log", ".ini", ".cfg"}
_CODE_EXTENSIONS = {".py", ".js", ".ts", ".html", ".css", ".sh", ".bat", ".ps1", ".sql", ".toml"}
_PDF_EXTENSIONS = {".pdf"}
_ALL_EXTENSIONS = _TEXT_EXTENSIONS | _CODE_EXTENSIONS | _PDF_EXTENSIONS


def _ensure_init():
    """Lazy-initialize ChromaDB collection for documents."""
    global _collection, _embed_fn, _initialized
    if _initialized:
        return

    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        vector_dir = Path(__file__).parent.parent.parent / "data" / "rag_index"
        vector_dir.mkdir(parents=True, exist_ok=True)

        _embed_fn = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            device="cpu",
        )

        client = chromadb.PersistentClient(path=str(vector_dir))
        _collection = client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=_embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

        log.info("RAG index initialized: %d chunks in %s", _collection.count(), vector_dir)
        _initialized = True

    except ImportError as e:
        log.warning("RAG unavailable (missing dependency): %s — pip install -e '.[memory]'", e)
        _initialized = True
    except Exception as e:
        log.warning("RAG init failed: %s", e)


def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by sentence boundaries."""
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) > chunk_size and current:
            chunks.append(current.strip())
            # Keep overlap from end of current chunk
            words = current.split()
            overlap_words = words[-overlap:] if len(words) > overlap else words
            current = " ".join(overlap_words) + " " + sentence
        else:
            current += (" " if current else "") + sentence

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if len(c) > 20]  # Skip tiny chunks


def _extract_text(path: Path) -> str:
    """Extract text content from a file."""
    ext = path.suffix.lower()

    if ext in _TEXT_EXTENSIONS | _CODE_EXTENSIONS:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            log.warning("Failed to read %s: %s", path, e)
            return ""

    if ext in _PDF_EXTENSIONS:
        return _extract_pdf(path)

    return ""


def _extract_pdf(path: Path) -> str:
    """Extract text from PDF using PyPDF2 or pdfminer."""
    # Try PyPDF2 first
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(str(path))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except ImportError:
        pass
    except Exception as e:
        log.warning("PyPDF2 failed for %s: %s", path, e)

    # Try pdfminer
    try:
        from pdfminer.high_level import extract_text
        return extract_text(str(path))
    except ImportError:
        pass
    except Exception as e:
        log.warning("pdfminer failed for %s: %s", path, e)

    log.warning("No PDF reader available for %s — install PyPDF2 or pdfminer", path)
    return ""


def _file_hash(path: Path) -> str:
    """Generate a hash of file content for change detection."""
    h = hashlib.md5()  # noqa: S324
    h.update(str(path).encode())
    try:
        h.update(str(path.stat().st_mtime).encode())
        h.update(str(path.stat().st_size).encode())
    except OSError:
        pass
    return h.hexdigest()[:12]


def index_file(path: Path) -> int:
    """Index a single file, returning number of chunks indexed."""
    _ensure_init()
    if _collection is None:
        return 0

    path = Path(path)
    if not path.exists() or path.suffix.lower() not in _ALL_EXTENSIONS:
        return 0

    file_id = _file_hash(path)

    # Check if already indexed (same version)
    existing = _collection.get(where={"file_hash": file_id})
    if existing and existing["ids"]:
        log.debug("File already indexed: %s", path.name)
        return 0

    # Extract and chunk text
    text = _extract_text(path)
    if not text.strip():
        return 0

    chunks = _chunk_text(text)
    if not chunks:
        return 0

    # Remove old versions of this file
    try:
        old = _collection.get(where={"source_file": str(path)})
        if old and old["ids"]:
            _collection.delete(ids=old["ids"])
    except Exception as e:
        log.debug("Failed to remove old index for %s: %s", path.name, e)

    # Index chunks
    ids = [f"doc_{file_id}_{i}" for i in range(len(chunks))]
    metadatas = [{
        "source_file": str(path),
        "file_name": path.name,
        "file_hash": file_id,
        "chunk_index": i,
        "total_chunks": len(chunks),
        "indexed_at": int(time.time()),
    } for i in range(len(chunks))]

    try:
        _collection.add(ids=ids, documents=chunks, metadatas=metadatas)
        log.info("Indexed %s: %d chunks", path.name, len(chunks))
        return len(chunks)
    except Exception as e:
        log.warning("Failed to index %s: %s", path.name, e)
        return 0


def index_directory(directory: Path | str | None = None) -> int:
    """Index all supported files in a directory recursively.

    Returns total number of chunks indexed.
    """
    directory = Path(directory) if directory else _DOCS_DIR
    if not directory or not directory.exists():
        return 0

    total = 0
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in _ALL_EXTENSIONS:
            total += index_file(path)

    log.info("Directory indexing complete: %d total chunks from %s", total, directory)
    return total


def search(query: str, n_results: int = 5) -> list[dict]:
    """Search indexed documents for relevant chunks.

    Returns list of {text, source, score} dicts.
    """
    _ensure_init()
    if _collection is None or _collection.count() == 0:
        return []

    try:
        results = _collection.query(
            query_texts=[query],
            n_results=min(n_results, _collection.count()),
        )

        hits = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 1.0
                hits.append({
                    "text": doc,
                    "source": meta.get("file_name", "unknown"),
                    "source_path": meta.get("source_file", ""),
                    "score": 1.0 - distance,  # Convert distance to similarity
                    "chunk_index": meta.get("chunk_index", 0),
                    "total_chunks": meta.get("total_chunks", 1),
                })

        # Filter by minimum relevance
        return [h for h in hits if h["score"] > 0.2]

    except Exception as e:
        log.warning("RAG search failed: %s", e)
        return []


def build_rag_context(query: str, max_chunks: int = 3) -> str:
    """Build a context block from relevant document chunks for the LLM.

    Returns formatted text to inject into the system prompt.
    """
    hits = search(query, n_results=max_chunks)
    if not hits:
        return ""

    lines = []
    for h in hits:
        source = h["source"]
        text = h["text"][:400]
        lines.append(f"[From {source}]: {text}")

    return (
        "\nRelevant information from your documents:\n"
        + "\n\n".join(lines)
        + "\n\nUse this information to answer the user's question."
    )


def get_stats() -> dict:
    """Get RAG index statistics."""
    _ensure_init()
    if _collection is None:
        return {"chunks": 0, "available": False}

    try:
        all_meta = _collection.get()
        files = set()
        if all_meta and all_meta["metadatas"]:
            for m in all_meta["metadatas"]:
                if m.get("file_name"):
                    files.add(m["file_name"])

        return {
            "chunks": _collection.count(),
            "files": len(files),
            "file_list": sorted(files),
            "available": True,
        }
    except Exception:
        return {"chunks": 0, "available": False}
