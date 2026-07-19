"""WhisperSTT のデコードパラメータと反復ガードのテスト。

mlx_whisper.transcribe を monkeypatch して、実モデルをロードせずに
- kwargs が意図通りか (temperature tuple / best_of / beam_size 不在 など)
- initial_prompt の伝搬 (空なら渡さない)
- 反復ガード (temperature>=0.4 かつ compression_ratio>2.0 のセグメント破棄)
を検証する。
"""

import numpy as np

import voiceinput.stt as stt_mod
from voiceinput.stt import WhisperSTT, _drop_repetition_segments


def _capture_transcribe(monkeypatch, result=None):
    captured: dict = {}

    def fake_transcribe(audio, **kwargs):
        captured["audio"] = audio
        captured.update(kwargs)
        return result or {"text": "こんにちは", "segments": []}

    monkeypatch.setattr(stt_mod.mlx_whisper, "transcribe", fake_transcribe)
    return captured


def test_decode_kwargs(monkeypatch):
    captured = _capture_transcribe(monkeypatch)
    stt = WhisperSTT("test-model")
    stt.transcribe(np.zeros(16000, dtype=np.float32))
    assert captured["path_or_hf_repo"] == "test-model"
    assert captured["language"] == "ja"
    assert captured["condition_on_previous_text"] is False
    assert captured["no_speech_threshold"] == 0.6
    assert captured["compression_ratio_threshold"] == 2.0
    assert captured["logprob_threshold"] == -1.0
    # 高温リトライは 0.4 で打ち切り (幻覚抑止)
    assert captured["temperature"] == (0.0, 0.2, 0.4)
    # fallback 時のみ効く。greedy パスでは mlx_whisper が自動無視
    assert captured["best_of"] == 3
    # beam_size は mlx_whisper 未実装 (渡すと即例外) — 絶対に渡さない
    assert "beam_size" not in captured


def test_initial_prompt_passed_when_nonempty(monkeypatch):
    captured = _capture_transcribe(monkeypatch)
    stt = WhisperSTT("m")
    stt.transcribe(np.zeros(100, dtype=np.float32), initial_prompt="山田、Codex。")
    assert captured["initial_prompt"] == "山田、Codex。"


def test_initial_prompt_omitted_when_empty(monkeypatch):
    captured = _capture_transcribe(monkeypatch)
    stt = WhisperSTT("m")
    stt.transcribe(np.zeros(100, dtype=np.float32), initial_prompt="")
    assert "initial_prompt" not in captured


def test_empty_audio_short_circuits(monkeypatch):
    captured = _capture_transcribe(monkeypatch)
    stt = WhisperSTT("m")
    assert stt.transcribe(np.zeros(0, dtype=np.float32)) == ""
    assert "audio" not in captured  # transcribe 自体呼ばれない


# --- 反復ガード (_drop_repetition_segments) ---


def test_guard_keeps_normal_segments():
    result = {
        "text": "全体テキスト",
        "segments": [
            {"text": "こんにちは。", "temperature": 0.0, "compression_ratio": 1.2},
            {"text": "元気です。", "temperature": 0.0, "compression_ratio": 1.1},
        ],
    }
    text, max_temp = _drop_repetition_segments(result)
    assert text == "こんにちは。元気です。"
    assert max_temp == 0.0


def test_guard_drops_repetition_suspect():
    """最終温度 (0.4) まで fallback して compression_ratio>2.0 のセグメントは破棄。"""
    result = {
        "text": "x",
        "segments": [
            {"text": "正常な文。", "temperature": 0.0, "compression_ratio": 1.3},
            {
                "text": "ありがとうございますありがとうございます",
                "temperature": 0.4,
                "compression_ratio": 3.5,
            },
        ],
    }
    text, max_temp = _drop_repetition_segments(result)
    assert text == "正常な文。"
    assert max_temp == 0.4


def test_guard_keeps_fallback_segment_with_ok_compression():
    """温度が上がっても compression が正常なら残す (fallback 成功ケース)。"""
    result = {
        "text": "x",
        "segments": [
            {"text": "難しい文。", "temperature": 0.4, "compression_ratio": 1.5},
        ],
    }
    text, max_temp = _drop_repetition_segments(result)
    assert text == "難しい文。"
    assert max_temp == 0.4


def test_guard_falls_back_to_text_without_segments():
    """segments が無い / 空なら result['text'] をそのまま使う (defensive)。"""
    assert _drop_repetition_segments({"text": "本文"}) == ("本文", 0.0)
    assert _drop_repetition_segments({"text": "本文", "segments": []}) == (
        "本文",
        0.0,
    )


def test_guard_integrated_in_transcribe(monkeypatch):
    """transcribe 経由でも反復セグメントが落ちる。"""
    result = {
        "text": "x",
        "segments": [
            {"text": "本題です。", "temperature": 0.0, "compression_ratio": 1.0},
            {"text": "繰り返し繰り返し", "temperature": 0.4, "compression_ratio": 9.9},
        ],
    }
    _capture_transcribe(monkeypatch, result=result)
    stt = WhisperSTT("m")
    assert stt.transcribe(np.zeros(100, dtype=np.float32)) == "本題です。"


# --- transcribe_chunks (Phase G: チャンク分割転写) ---


def test_chunks_empty_list_returns_empty():
    stt = WhisperSTT("m")
    assert stt.transcribe_chunks([]) == ""


def test_chunks_single_chunk_same_as_transcribe(monkeypatch):
    calls: list = []

    def fake_transcribe(audio, **kwargs):
        calls.append(kwargs)
        return {"text": "こんにちは", "segments": []}

    monkeypatch.setattr(stt_mod.mlx_whisper, "transcribe", fake_transcribe)
    stt = WhisperSTT("m")
    out = stt.transcribe_chunks(
        [np.zeros(100, dtype=np.float32)], initial_prompt="山田。"
    )
    assert out == "こんにちは"
    assert len(calls) == 1
    assert calls[0]["initial_prompt"] == "山田。"


def test_chunks_each_gets_same_prompt_and_texts_joined(monkeypatch):
    """複数チャンクの各 transcribe に同じ prompt が渡り、結果が連結される。"""
    calls: list = []
    outputs = iter(["前半です。", "後半です。"])

    def fake_transcribe(audio, **kwargs):
        calls.append(kwargs.get("initial_prompt"))
        return {"text": next(outputs), "segments": []}

    monkeypatch.setattr(stt_mod.mlx_whisper, "transcribe", fake_transcribe)
    stt = WhisperSTT("m")
    out = stt.transcribe_chunks(
        [np.zeros(100, dtype=np.float32), np.zeros(100, dtype=np.float32)],
        initial_prompt="語彙。",
    )
    assert out == "前半です。後半です。"
    # チャンクごとに同じ prompt (長文の後半にも語彙ヒントを効かせる目的)
    assert calls == ["語彙。", "語彙。"]
