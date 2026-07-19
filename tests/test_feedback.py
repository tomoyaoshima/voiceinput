import wave

import numpy as np

from voiceinput import feedback


def test_log_sweep_rises_in_frequency():
    pcm = feedback._log_sweep(200.0, 2000.0, 0.2)
    assert pcm.dtype == np.int16
    assert pcm.size > 0
    # ピーク振幅が 0.35 * 32767 を超えない
    assert int(np.max(np.abs(pcm))) <= int(0.4 * 32767)


def test_ensure_sound_creates_cached_wav(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, "_CACHE_DIR", tmp_path)
    feedback._GENERATED_PATHS.clear()
    path = feedback._ensure_sound("start")
    assert path is not None
    assert path.exists()
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 44100
        assert wf.getnframes() > 0


def test_ensure_sound_unknown_name_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, "_CACHE_DIR", tmp_path)
    feedback._GENERATED_PATHS.clear()
    assert feedback._ensure_sound("nonexistent") is None


def test_play_unknown_name_is_noop():
    feedback.play("nonexistent_sound_name")  # 例外なく no-op


def test_play_known_name_generates_file(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, "_CACHE_DIR", tmp_path)
    feedback._GENERATED_PATHS.clear()
    feedback._NSSOUND_CACHE.clear()
    feedback.play("start")
    # 副作用として wav が生成されている
    assert (tmp_path / "start.wav").exists()
