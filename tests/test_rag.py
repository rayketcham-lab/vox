"""Tests for RAG module — document chunking and search."""

import pytest

from vox.rag import _chunk_text, _extract_text, _file_hash


def test_chunk_text_basic():
    text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
    chunks = _chunk_text(text, chunk_size=50, overlap=2)
    assert len(chunks) >= 1
    assert all(len(c) > 20 for c in chunks)


def test_chunk_text_short():
    text = "Short."
    chunks = _chunk_text(text, chunk_size=500, overlap=5)
    assert chunks == []  # Too short (<20 chars)


def test_chunk_text_overlap():
    text = " ".join(f"Sentence number {i}." for i in range(20))
    chunks = _chunk_text(text, chunk_size=100, overlap=3)
    assert len(chunks) > 1
    # Overlap means some words appear in consecutive chunks
    if len(chunks) >= 2:
        words_0 = set(chunks[0].split()[-3:])
        words_1 = set(chunks[1].split()[:6])
        assert words_0 & words_1, "Expected overlap between consecutive chunks"


def test_extract_text_txt(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("Hello world, this is a test document.")
    text = _extract_text(f)
    assert "Hello world" in text


def test_extract_text_unsupported(tmp_path):
    f = tmp_path / "test.xyz"
    f.write_text("data")
    text = _extract_text(f)
    assert text == ""


def test_file_hash_consistent(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("content")
    h1 = _file_hash(f)
    h2 = _file_hash(f)
    assert h1 == h2
    assert len(h1) == 12


def test_file_hash_changes(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("content v1")
    h1 = _file_hash(f)
    import time
    time.sleep(0.1)
    f.write_text("content v2")
    h2 = _file_hash(f)
    assert h1 != h2
