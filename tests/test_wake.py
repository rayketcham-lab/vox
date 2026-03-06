"""Tests for wake word and activation modes."""



def test_always_mode_returns_immediately(monkeypatch):
    monkeypatch.setattr("vox.wake.LISTEN_MODE", "always")
    from vox.wake import wait_for_activation
    # Should return immediately without blocking
    wait_for_activation()


def test_ptt_state():
    from vox.wake import _ptt_active
    assert not _ptt_active.is_set()


def test_config_defaults():
    from vox.config import LISTEN_MODE, WAKE_SENSITIVITY, WAKE_WORD
    assert isinstance(WAKE_WORD, str)
    assert isinstance(WAKE_SENSITIVITY, float)
    assert 0.0 <= WAKE_SENSITIVITY <= 1.0
    assert LISTEN_MODE in ("wake", "ptt", "always")
