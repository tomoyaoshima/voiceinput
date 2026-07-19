"""録音状態のフィードバック音を再生する。

macOS 標準サウンドは似たり寄ったりで「開始」「停止」を直感的に区別しづらいため、
numpy で対数スイープ波形を生成して使う:

- start: 低 → 高 への上昇 sweep ("ひゅっ" と立ち上がる感じ)
- stop:  高 → 低 への下降 sweep (落ち着く / 閉じる感じ)
- error: 急下降 sweep (失敗を示す)

生成した WAV は OS の一時ディレクトリにキャッシュし、NSSound 経由で鳴らす。
NSSound は別スレッドから呼んでも安全。
"""

import tempfile
import wave
from pathlib import Path

import numpy as np

try:
    from AppKit import NSSound
except Exception:  # pragma: no cover
    NSSound = None  # type: ignore

_CACHE_DIR = Path(tempfile.gettempdir()) / "voiceinput-sounds"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _log_sweep(
    start_hz: float,
    end_hz: float,
    duration: float,
    *,
    sample_rate: int = 44100,
    amplitude: float = 0.35,
    fade_ms: float = 15.0,
) -> np.ndarray:
    """対数(指数)周波数スイープを生成して int16 PCM の numpy 配列を返す。"""
    n = int(duration * sample_rate)
    t = np.linspace(0.0, duration, n, endpoint=False)
    k = np.log(end_hz / start_hz) / duration
    # 瞬時周波数 f0 * exp(k*t) の積分が位相
    phase = 2 * np.pi * start_hz * (np.exp(k * t) - 1.0) / k
    samples = np.sin(phase)

    fade = max(int(fade_ms / 1000.0 * sample_rate), 1)
    envelope = np.ones_like(samples)
    envelope[:fade] = np.linspace(0.0, 1.0, fade)
    envelope[-fade:] = np.linspace(1.0, 0.0, fade)
    samples *= envelope * amplitude

    return (samples * 32767).astype(np.int16)


def _write_wav(path: Path, pcm: np.ndarray, sample_rate: int = 44100) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


# 各サウンドの周波数プロファイル (start_hz, end_hz, duration_sec)
_PROFILES: dict[str, tuple[float, float, float]] = {
    "start": (440.0, 1320.0, 0.16),  # 上昇 (440Hz → 1320Hz、ひゅっと立ち上がり)
    "stop":  (1320.0, 480.0, 0.22),  # 下降 (緩やかに落ちる、閉じる感)
    "error": (660.0, 110.0, 0.32),   # 急下降 (失敗、深く落ちる)
}

_GENERATED_PATHS: dict[str, Path] = {}
_NSSOUND_CACHE: dict[str, "NSSound"] = {}


def _ensure_sound(name: str) -> Path | None:
    if name not in _PROFILES:
        return None
    if name in _GENERATED_PATHS and _GENERATED_PATHS[name].exists():
        return _GENERATED_PATHS[name]
    start_hz, end_hz, duration = _PROFILES[name]
    pcm = _log_sweep(start_hz, end_hz, duration)
    path = _CACHE_DIR / f"{name}.wav"
    _write_wav(path, pcm)
    _GENERATED_PATHS[name] = path
    return path


def play(name: str) -> None:
    """`start` / `stop` / `error` のいずれかを再生する。"""
    if NSSound is None:
        return
    path = _ensure_sound(name)
    if path is None:
        return
    sound = _NSSOUND_CACHE.get(name)
    if sound is None:
        sound = NSSound.alloc().initWithContentsOfFile_byReference_(str(path), True)
        if sound is None:
            return
        _NSSOUND_CACHE[name] = sound
    sound.stop()
    sound.play()
