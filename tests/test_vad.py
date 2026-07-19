"""silero-vad は実音声サンプルで訓練されているため合成音では検出できない。

ライブラリ自体の検出精度は外部依存として信頼し、ここでは
- 空入力・全無音入力での short-circuit
- silero-vad が返した timestamps をどう連結するか (gap 挿入含む)
- 不変条件 (min_silence >= 2*speech_pad)
だけをユニットテストする。実音声での検出は実機 (録音) で確認する。
"""

import sys
import types

import numpy as np
import pytest

import voiceinput.vad as vad_mod
from voiceinput.vad import split_speech_chunks, trim_silence


def _install_fake_silero(monkeypatch, timestamps_or_fn):
    monkeypatch.setattr(vad_mod, "_ensure_model", lambda: object())
    fake = types.ModuleType("silero_vad")
    if callable(timestamps_or_fn):
        fake.get_speech_timestamps = timestamps_or_fn
    else:
        fake.get_speech_timestamps = lambda audio, model, **kw: timestamps_or_fn
    monkeypatch.setitem(sys.modules, "silero_vad", fake)


def test_empty_audio_returns_empty():
    out = trim_silence(np.zeros(0, dtype=np.float32))
    assert out.size == 0


def test_no_timestamps_returns_empty(monkeypatch):
    """silero が「発話なし」と判定したら長さ 0 を返す。"""
    _install_fake_silero(monkeypatch, [])
    audio = np.ones(16000 * 3, dtype=np.float32) * 0.01
    out = trim_silence(audio)
    assert out.size == 0


def test_concatenates_with_gap_between_segments(monkeypatch):
    """2 区間の間に gap_ms 分のゼロ無音が入る (直結しない)。"""
    _install_fake_silero(
        monkeypatch,
        [{"start": 1000, "end": 3000}, {"start": 5000, "end": 7000}],
    )
    audio = np.arange(10000, dtype=np.float32)
    out = trim_silence(audio, sample_rate=16000, gap_ms=400)
    gap_samples = int(16000 * 400 / 1000)  # 6400
    assert out.size == 4000 + gap_samples
    # 前半 2000 サンプルは区間 1
    np.testing.assert_array_equal(out[:2000], np.arange(1000, 3000, dtype=np.float32))
    # 中央はゼロ無音
    np.testing.assert_array_equal(
        out[2000 : 2000 + gap_samples], np.zeros(gap_samples, dtype=np.float32)
    )
    # 後半 2000 サンプルは区間 2
    np.testing.assert_array_equal(
        out[2000 + gap_samples :], np.arange(5000, 7000, dtype=np.float32)
    )


def test_gap_zero_restores_legacy_concat(monkeypatch):
    """gap_ms=0 で従来の直結動作に戻る (rollback パス)。"""
    _install_fake_silero(
        monkeypatch,
        [{"start": 1000, "end": 3000}, {"start": 5000, "end": 7000}],
    )
    audio = np.arange(10000, dtype=np.float32)
    out = trim_silence(audio, gap_ms=0)
    expected = np.concatenate(
        [np.arange(1000, 3000, dtype=np.float32), np.arange(5000, 7000, dtype=np.float32)]
    )
    assert out.size == 4000
    np.testing.assert_array_equal(out, expected)


def test_single_segment_has_no_gap(monkeypatch):
    """区間が 1 つなら gap は入らない。"""
    _install_fake_silero(monkeypatch, [{"start": 1000, "end": 3000}])
    audio = np.arange(10000, dtype=np.float32)
    out = trim_silence(audio, gap_ms=400)
    assert out.size == 2000
    np.testing.assert_array_equal(out, np.arange(1000, 3000, dtype=np.float32))


def test_passes_voicing_parameters_to_silero(monkeypatch):
    """min_speech / min_silence / speech_pad / threshold が silero に渡る。"""
    captured: dict = {}

    def fake_get(audio, model, **kw):
        captured.update(kw)
        return []

    _install_fake_silero(monkeypatch, fake_get)

    audio = np.zeros(16000, dtype=np.float32)
    trim_silence(
        audio,
        sample_rate=16000,
        min_speech_ms=150,
        min_silence_ms=300,
        speech_pad_ms=100,
        threshold=0.45,
    )
    assert captured["sampling_rate"] == 16000
    assert captured["min_speech_duration_ms"] == 150
    assert captured["min_silence_duration_ms"] == 300
    assert captured["speech_pad_ms"] == 100
    assert captured["threshold"] == 0.45


def test_default_params_satisfy_invariant(monkeypatch):
    """デフォルト値 (700/300) が不変条件を満たしている。"""
    captured: dict = {}

    def fake_get(audio, model, **kw):
        captured.update(kw)
        return []

    _install_fake_silero(monkeypatch, fake_get)
    trim_silence(np.zeros(16000, dtype=np.float32))
    assert captured["min_silence_duration_ms"] == 700
    assert captured["speech_pad_ms"] == 300
    assert captured["min_silence_duration_ms"] >= 2 * captured["speech_pad_ms"]


def test_invariant_violation_raises(monkeypatch):
    """min_silence < 2*speech_pad は AssertionError。

    silero は区間をマージせず無音折半で密着させるため、この条件が崩れると
    発話の途中に人工ポーズ (gap) を注入する事故になる。
    """
    _install_fake_silero(monkeypatch, [])
    with pytest.raises(AssertionError):
        trim_silence(
            np.zeros(16000, dtype=np.float32),
            min_silence_ms=400,
            speech_pad_ms=300,  # 400 < 600 で違反
        )


# --- split_speech_chunks (Phase G: チャンク分割転写) ---


def test_chunks_empty_audio_returns_empty_list():
    assert split_speech_chunks(np.zeros(0, dtype=np.float32)) == []


def test_chunks_no_speech_returns_empty_list(monkeypatch):
    _install_fake_silero(monkeypatch, [])
    assert split_speech_chunks(np.ones(16000, dtype=np.float32) * 0.01) == []


def test_chunks_short_speech_single_chunk(monkeypatch):
    """max_chunk_sec 以内なら 1 チャンク (trim_silence と同一の中身)。"""
    _install_fake_silero(
        monkeypatch,
        [{"start": 1000, "end": 3000}, {"start": 5000, "end": 7000}],
    )
    audio = np.arange(10000, dtype=np.float32)
    chunks = split_speech_chunks(audio, sample_rate=16000, gap_ms=400)
    assert len(chunks) == 1
    expected = trim_silence(audio, sample_rate=16000, gap_ms=400)
    np.testing.assert_array_equal(chunks[0], expected)


def test_chunks_split_at_vad_boundary(monkeypatch):
    """max_chunk_sec を超えそうな区間は次のチャンクへ折り返す。"""
    sr = 16000
    # 各 2 秒の発話 3 区間。max_chunk_sec=5 なら 2 区間 (2+gap0.4+2=4.4s) +
    # 1 区間に分かれる。
    _install_fake_silero(
        monkeypatch,
        [
            {"start": 0, "end": 2 * sr},
            {"start": 3 * sr, "end": 5 * sr},
            {"start": 6 * sr, "end": 8 * sr},
        ],
    )
    audio = np.arange(10 * sr, dtype=np.float32)
    chunks = split_speech_chunks(
        audio, sample_rate=sr, max_chunk_sec=5.0, gap_ms=400
    )
    assert len(chunks) == 2
    gap = int(sr * 0.4)
    assert chunks[0].size == 2 * sr + gap + 2 * sr
    assert chunks[1].size == 2 * sr


def test_chunks_oversized_single_segment_kept_whole(monkeypatch):
    """1 区間単体が max_chunk_sec 超でも切らず 1 チャンクにする。"""
    sr = 16000
    _install_fake_silero(monkeypatch, [{"start": 0, "end": 10 * sr}])
    audio = np.arange(10 * sr, dtype=np.float32)
    chunks = split_speech_chunks(audio, sample_rate=sr, max_chunk_sec=5.0)
    assert len(chunks) == 1
    assert chunks[0].size == 10 * sr


def test_chunks_silero_missing_returns_whole_audio(monkeypatch):
    """silero 未導入なら [audio] の 1 チャンク (no-op)。"""

    def raise_import(*a, **kw):
        raise RuntimeError("no silero")

    monkeypatch.setattr(vad_mod, "_ensure_model", raise_import)
    audio = np.ones(16000, dtype=np.float32)
    chunks = split_speech_chunks(audio)
    assert len(chunks) == 1
    np.testing.assert_array_equal(chunks[0], audio)
