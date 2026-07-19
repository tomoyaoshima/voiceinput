"""silero-vad で録音バッファから無音区間を削除する。

録音時間に無音が多いと Whisper の処理時間が無駄に伸びるため、
発話と判定された区間だけを連結して STT に渡す。

silero-vad のモデル本体は ~2MB、初回 import 時に内部キャッシュへロード。
"""

from __future__ import annotations

import threading

import numpy as np

_LOCK = threading.Lock()
_MODEL = None  # silero_vad model singleton


def _ensure_model():
    """silero-vad モデルを 1 回だけ読み込んで返す。"""
    global _MODEL
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                from silero_vad import load_silero_vad
                _MODEL = load_silero_vad()
    return _MODEL


def warmup() -> None:
    """モデルロード + ダミー推論を走らせて初回呼び出しの遅延を消す。"""
    try:
        model = _ensure_model()
        # silero-vad は 16kHz、512 samples を 1 chunk で処理
        import torch
        with torch.no_grad():
            _ = model(torch.zeros(512, dtype=torch.float32), 16000)
    except Exception:
        pass


def _speech_timestamps(
    audio: np.ndarray,
    sample_rate: int,
    *,
    min_speech_ms: int,
    min_silence_ms: int,
    speech_pad_ms: int,
    threshold: float,
) -> list[dict] | None:
    """silero-vad の発話区間タイムスタンプを返す。未導入なら None。

    不変条件 ``min_silence_ms >= 2 * speech_pad_ms`` を要求する。
    silero は重なった区間を「マージ」せず、隣接区間の間の無音を折半して
    区間を密着させる。この条件が崩れると、密着した区間の間に呼び出し側が
    無音 gap を挿入した時、発話の途中に人工ポーズを注入する事故になる。
    """
    assert min_silence_ms >= 2 * speech_pad_ms, (
        f"min_silence_ms ({min_silence_ms}) must be >= 2 * speech_pad_ms "
        f"({speech_pad_ms}) — silero pads without merging, so shorter "
        f"silences would make segments touch and gap insertion would inject "
        f"artificial pauses mid-speech"
    )
    try:
        model = _ensure_model()
        from silero_vad import get_speech_timestamps
        import torch
    except Exception:
        return None

    audio_tensor = torch.from_numpy(audio.astype(np.float32))
    return get_speech_timestamps(
        audio_tensor,
        model,
        sampling_rate=sample_rate,
        threshold=threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=False,
    )


def trim_silence(
    audio: np.ndarray,
    sample_rate: int = 16000,
    *,
    min_speech_ms: int = 200,
    min_silence_ms: int = 700,
    speech_pad_ms: int = 300,
    gap_ms: int = 400,
    threshold: float = 0.5,
) -> np.ndarray:
    """発話区間を連結して返す。発話がなければ長さ 0 の配列。

    Parameters
    ----------
    audio : float32 mono numpy
    min_speech_ms : これ未満の発話区間は破棄 (誤検知抑制)
    min_silence_ms : これ未満の無音は「区間途中の間」として連結したまま残す。
        700ms (旧 400ms) — 短いポーズを保持して Whisper に自然な間を渡す
    speech_pad_ms : 各発話区間の前後に padding。300ms (旧 200ms) —
        語頭の無声子音の切れを軽減
    gap_ms : 区間と区間の間に挿入する無音の長さ。旧実装は区間を直結して
        いたため、離れた発話が不自然に密着して単語誤認識を誘発していた。
        400ms のゼロ無音を挟むことで Whisper がセグメント境界を自然に
        検出できる。0 で従来の直結動作。
        (接合部で幻覚が観測された場合、ゼロでなく微小ノイズ 1e-4 振幅に
        変えるという逃げ道もある)
    threshold : silero の発話確率しきい値 (デフォルト 0.5 = silero 既定)。
        下げると感度が上がり語頭を拾いやすくなるが、ブレス・環境音の
        混入も増える。変更はベンチ (実発話セット) で比較してから。
    """
    if audio.size == 0:
        return audio
    timestamps = _speech_timestamps(
        audio,
        sample_rate,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
        threshold=threshold,
    )
    if timestamps is None:
        # silero-vad が未導入なら no-op (生 audio をそのまま返す)
        return audio
    if not timestamps:
        return np.zeros(0, dtype=np.float32)
    gap = np.zeros(int(sample_rate * gap_ms / 1000), dtype=np.float32)
    parts: list[np.ndarray] = []
    for i, ts in enumerate(timestamps):
        if i > 0 and gap.size > 0:
            parts.append(gap)
        parts.append(audio[ts["start"] : ts["end"]])
    return np.concatenate(parts).astype(np.float32)


def split_speech_chunks(
    audio: np.ndarray,
    sample_rate: int = 16000,
    *,
    max_chunk_sec: float = 28.0,
    min_speech_ms: int = 200,
    min_silence_ms: int = 700,
    speech_pad_ms: int = 300,
    gap_ms: int = 400,
    threshold: float = 0.5,
) -> list[np.ndarray]:
    """発話区間を VAD 境界で ≤ max_chunk_sec のチャンク束に分割して返す。

    なぜチャンク分割するか: Whisper の initial_prompt は「最初の 30 秒窓」に
    しか効かない (condition_on_previous_text=False だと各窓処理後に prompt が
    リセットされる — mlx_whisper transcribe.py で検証済み)。長い録音を 1 回の
    transcribe に渡すと、30 秒を超えた部分は語彙ヒントなしで認識されて
    しまう。VAD 境界で分割し、チャンクごとに transcribe + prompt を与える
    ことで、長文の後半にも語彙ヒントを効かせる。

    - 発話区間を貪欲に詰め、gap 込みで max_chunk_sec を超えそうなら次の
      チャンクへ折り返す (区間の途中では絶対に切らない = VAD 境界のみ)。
    - 1 区間単体が max_chunk_sec を超える場合はそのまま 1 チャンクにする
      (Whisper 側の 30 秒窓処理に任せる。無理に切ると発話中で切れる)。
    - 発話なしなら空リスト。silero 未導入なら [audio] (1 チャンク、no-op)。
    - チャンク内の区間連結は trim_silence と同じ gap 挿入方式。
    """
    if audio.size == 0:
        return []
    timestamps = _speech_timestamps(
        audio,
        sample_rate,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
        threshold=threshold,
    )
    if timestamps is None:
        return [audio]
    if not timestamps:
        return []

    gap = np.zeros(int(sample_rate * gap_ms / 1000), dtype=np.float32)
    max_samples = int(max_chunk_sec * sample_rate)

    chunks: list[np.ndarray] = []
    bucket: list[np.ndarray] = []
    bucket_samples = 0

    def flush() -> None:
        nonlocal bucket, bucket_samples
        if bucket:
            chunks.append(np.concatenate(bucket).astype(np.float32))
            bucket = []
            bucket_samples = 0

    for ts in timestamps:
        seg = audio[ts["start"] : ts["end"]]
        added = seg.size + (gap.size if bucket else 0)
        if bucket and bucket_samples + added > max_samples:
            flush()
            added = seg.size
        if bucket and gap.size > 0:
            bucket.append(gap)
        bucket.append(seg)
        bucket_samples += added
    flush()
    return chunks
