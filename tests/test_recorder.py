"""Recorder の堅牢性テスト。

PortAudioError などで sd.InputStream の生成が失敗した時に、Recorder の
内部状態が "未録音" のまま残ることを保証する。app.py 側はこの不変条件を
頼りに「start が失敗したら _toggle_active を rollback」する。
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import sounddevice as sd

from voiceinput import recorder as recorder_mod
from voiceinput.recorder import Recorder


class _FailingInputStream:
    """sd.InputStream の生成自体が失敗するケース (PortAudioError 互換)。"""

    def __init__(self, *args, **kwargs) -> None:
        raise sd.PortAudioError("simulated PortAudio failure", -9986)


class _FakeStream:
    """成功パスで使う最小限のスタブ。"""

    def __init__(self, *args, **kwargs) -> None:
        self.callback = kwargs.get("callback")
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.closed = True


def _patch_sounddevice(monkeypatch: pytest.MonkeyPatch, stream_cls) -> None:
    """recorder.sd を最小スタブで差し替える。

    実 sd を呼ぶと CI でデバイスがなくて落ちるので、必要な属性
    (InputStream / PortAudioError / _terminate / _initialize) だけを
    用意した SimpleNamespace を渡す。
    """
    fake_sd = types.SimpleNamespace(
        InputStream=stream_cls,
        PortAudioError=sd.PortAudioError,  # 例外クラスは本物を流用
        _terminate=lambda: None,
        _initialize=lambda: None,
    )
    monkeypatch.setattr(recorder_mod, "sd", fake_sd)


def test_start_failure_keeps_stream_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PortAudioError が 2 回続けて起きると最終的に再 raise される。

    その時 is_recording は False のまま (= app.py 側の _toggle_active
    rollback と整合する)。
    """
    _patch_sounddevice(monkeypatch, _FailingInputStream)
    # recycle で sd._terminate / _initialize は無視させる
    monkeypatch.setattr(recorder_mod, "_recycle_portaudio", lambda: None)
    rec = Recorder()
    with pytest.raises(sd.PortAudioError):
        rec.start()
    assert rec.is_recording is False
    assert rec._stream is None


def test_stop_after_failed_start_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sounddevice(monkeypatch, _FailingInputStream)
    monkeypatch.setattr(recorder_mod, "_recycle_portaudio", lambda: None)
    rec = Recorder()
    with pytest.raises(sd.PortAudioError):
        rec.start()
    out = rec.stop()
    assert isinstance(out, np.ndarray)
    assert out.size == 0


def test_start_recovers_after_one_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """1 度目の InputStream で PortAudioError → recycle 後の retry で成功。

    AirPods 切替などで PortAudio キャッシュが陳腐化しているケースを
    再現する。recycle が呼ばれ、retry で _FakeStream が返るシナリオ。
    """
    attempts: list[int] = []
    recycle_calls: list[int] = []

    class FlipStream:
        def __init__(self, *args, **kwargs) -> None:
            attempts.append(1)
            if len(attempts) == 1:
                raise sd.PortAudioError("first time fails", -9986)
            self.callback = kwargs.get("callback")

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def close(self) -> None:
            pass

    _patch_sounddevice(monkeypatch, FlipStream)
    monkeypatch.setattr(
        recorder_mod, "_recycle_portaudio", lambda: recycle_calls.append(1)
    )

    rec = Recorder()
    rec.start()
    assert rec.is_recording is True
    assert len(attempts) == 2  # 1 回失敗 + 1 回成功
    assert len(recycle_calls) == 1  # recycle が間に挟まる


def test_start_then_stop_returns_recorded_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    """成功パスの sanity check。callback で渡した波形が返ってくる。"""
    _patch_sounddevice(monkeypatch, _FakeStream)
    rec = Recorder()
    rec.start()
    assert rec.is_recording is True
    # callback を直接叩いて録音バッファに足す
    sample = np.ones((400, 1), dtype=np.float32)
    rec._callback(sample, frames=400, time_info=None, status=None)
    out = rec.stop()
    assert rec.is_recording is False
    assert out.shape == (400,)
    assert np.all(out == 1.0)


def test_start_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """既に録音中なら 2 回目の start は何もしない (前回の stream を捨てない)。"""
    _patch_sounddevice(monkeypatch, _FakeStream)
    rec = Recorder()
    rec.start()
    first_stream = rec._stream
    rec.start()
    assert rec._stream is first_stream
