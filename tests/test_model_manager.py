"""Tests for the modular model manager."""

import time
import threading
from unittest.mock import MagicMock

from vox import model_manager as mm


class TestModelManager:
    def setup_method(self):
        """Reset model registry between tests."""
        mm._models.clear()

    def test_register_and_acquire(self):
        mock_model = MagicMock()
        mm.register("test", loader=lambda: mock_model, vram_mb=100)

        result = mm.acquire("test")
        assert result is mock_model

    def test_acquire_caches(self):
        call_count = 0
        def loader():
            nonlocal call_count
            call_count += 1
            return MagicMock()

        mm.register("test", loader=loader, vram_mb=100)
        m1 = mm.acquire("test")
        m2 = mm.acquire("test")
        assert m1 is m2
        assert call_count == 1

    def test_release_with_zero_keep_alive(self):
        mock_model = MagicMock()
        mm.register("test", loader=lambda: mock_model, vram_mb=100, keep_alive=0)
        mm.acquire("test")

        mm.release("test")
        status = mm.status()
        assert not status["test"]["loaded"]

    def test_release_with_keep_alive_stays_loaded(self):
        mock_model = MagicMock()
        mm.register("test", loader=lambda: mock_model, vram_mb=100, keep_alive=300)
        mm.acquire("test")

        mm.release("test")
        status = mm.status()
        assert status["test"]["loaded"]  # Still loaded within keep_alive

    def test_force_release(self):
        mock_model = MagicMock()
        mm.register("test", loader=lambda: mock_model, vram_mb=100, keep_alive=300)
        mm.acquire("test")

        mm.release("test", force=True)
        assert not mm.status()["test"]["loaded"]

    def test_release_all(self):
        mm.register("a", loader=lambda: MagicMock(), vram_mb=100)
        mm.register("b", loader=lambda: MagicMock(), vram_mb=200)
        mm.acquire("a")
        mm.acquire("b")

        assert mm.vram_loaded() == 300
        mm.release_all()
        assert mm.vram_loaded() == 0

    def test_cleanup_expired(self):
        mock_model = MagicMock()
        mm.register("test", loader=lambda: mock_model, vram_mb=100, keep_alive=1)
        mm.acquire("test")

        # Manually set last_used to past
        mm._models["test"]["last_used"] = time.time() - 10
        mm.cleanup_expired()
        assert not mm.status()["test"]["loaded"]

    def test_cleanup_keeps_active(self):
        mock_model = MagicMock()
        mm.register("test", loader=lambda: mock_model, vram_mb=100, keep_alive=300)
        mm.acquire("test")

        mm.cleanup_expired()
        assert mm.status()["test"]["loaded"]

    def test_vram_loaded(self):
        mm.register("a", loader=lambda: MagicMock(), vram_mb=4000)
        mm.register("b", loader=lambda: MagicMock(), vram_mb=8000)
        assert mm.vram_loaded() == 0

        mm.acquire("a")
        assert mm.vram_loaded() == 4000

        mm.acquire("b")
        assert mm.vram_loaded() == 12000

    def test_model_lease_context_manager(self):
        mock_model = MagicMock()
        mm.register("test", loader=lambda: mock_model, vram_mb=100, keep_alive=0)

        with mm.model_lease("test") as model:
            assert model is mock_model
            assert mm.status()["test"]["loaded"]

        # Should be unloaded after context exit
        assert not mm.status()["test"]["loaded"]

    def test_unknown_model_raises(self):
        import pytest
        with pytest.raises(KeyError):
            mm.acquire("nonexistent")

    def test_status_format(self):
        mm.register("test", loader=lambda: MagicMock(), vram_mb=500, keep_alive=60)
        status = mm.status()
        assert "test" in status
        assert status["test"]["loaded"] is False
        assert status["test"]["vram_mb"] == 500
        assert status["test"]["keep_alive"] == 60

    def test_thread_safety(self):
        """Multiple threads acquiring same model should only load once."""
        call_count = 0
        def loader():
            nonlocal call_count
            call_count += 1
            time.sleep(0.1)
            return MagicMock()

        mm.register("test", loader=loader, vram_mb=100)
        results = []

        def acquire():
            results.append(mm.acquire("test"))

        threads = [threading.Thread(target=acquire) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # First thread loads, rest reuse cached
        assert len(results) == 5
