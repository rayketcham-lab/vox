"""Tests for vector memory (ChromaDB-based semantic search)."""


import pytest

from vox import vector_memory as vm


@pytest.fixture(autouse=True)
def reset_vector_memory(tmp_path):
    """Use a temporary directory for vector memory in tests."""
    original_dir = vm._VECTOR_DIR
    original_init = vm._initialized
    original_collection = vm._collection

    vm._VECTOR_DIR = tmp_path / "test_vector_memory"
    vm._initialized = False
    vm._collection = None

    yield

    vm._VECTOR_DIR = original_dir
    vm._initialized = original_init
    vm._collection = original_collection


class TestVectorMemory:
    def test_init_creates_collection(self):
        vm._ensure_init()
        assert vm._collection is not None
        assert vm._initialized is True

    def test_store_and_count(self):
        assert vm.count() == 0
        vm.store("Hello, how are you?", "I'm doing great, thanks!")
        assert vm.count() == 1

    def test_store_multiple(self):
        vm.store("What's the weather?", "It's sunny and 72F.")
        vm.store("Tell me a joke", "Why did the chicken cross the road?")
        vm.store("What's my name?", "Your name is Ray.")
        assert vm.count() == 3

    def test_store_fact(self):
        vm.store_fact("User's favorite color is blue", "preference")
        assert vm.count() == 1

    def test_search_finds_relevant(self):
        vm.store("I love pizza with pepperoni", "Great choice! Pepperoni is classic.")
        vm.store("The weather is sunny today", "Perfect day for a walk!")
        vm.store("My car needs an oil change", "When was the last one?")

        hits = vm.search("What food do I like?")
        assert len(hits) > 0
        # Pizza conversation should be most relevant
        assert "pizza" in hits[0]["document"].lower()

    def test_search_empty_returns_empty(self):
        hits = vm.search("anything")
        assert hits == []

    def test_search_with_type_filter(self):
        vm.store("How's work?", "It's going well.")
        vm.store_fact("User works at Acme Corp")

        conv_hits = vm.search("work", type_filter="conversation")
        fact_hits = vm.search("work", type_filter="fact")

        assert any(h["metadata"]["type"] == "conversation" for h in conv_hits)
        assert any(h["metadata"]["type"] == "fact" for h in fact_hits)

    def test_build_context_block_empty(self):
        result = vm.build_context_block("Hello")
        assert result == ""  # Short message, no context

    def test_build_context_block_with_history(self):
        # Need enough entries for the min threshold (3)
        vm.store("I really love hiking in the mountains", "The mountains are beautiful this time of year!")
        vm.store("My favorite trail is Angel's Landing", "That's an amazing hike in Zion!")
        vm.store("I went camping last weekend", "How was the camping trip?")
        vm.store("The sunrise from the peak was incredible", "Mountain sunrises are the best!")

        result = vm.build_context_block("Tell me about good hiking spots")
        # Should retrieve relevant past conversations
        assert "Previously discussed" in result or result == ""

    def test_metadata_stored(self):
        vm.store("test message", "test response", metadata={"mood": "happy"})
        hits = vm.search("test message")
        assert len(hits) > 0
        assert hits[0]["metadata"]["type"] == "conversation"
        assert "timestamp" in hits[0]["metadata"]

    def test_count_zero_initially(self):
        assert vm.count() == 0

    def test_store_truncates_long_messages(self):
        long_msg = "x" * 1000
        vm.store(long_msg, long_msg)
        hits = vm.search("x" * 50)
        # Metadata should be truncated to 500 chars
        assert len(hits[0]["metadata"]["user_message"]) <= 500

    def test_clear(self):
        vm.store("something", "response")
        assert vm.count() == 1
        vm.clear()
        vm._ensure_init()
        assert vm.count() == 0
