import logging
import threading

import numpy as np
import sounddevice as sd

_logger = logging.getLogger("voiceinput.recorder")


def _recycle_portaudio() -> None:
    """PortAudio のデバイス一覧キャッシュを作り直す。

    voiceinput はメニューバー常駐で長時間動くので、AirPods 接続・
    Studio Display 接続・既定マイク変更などが起きると、PortAudio が
    プロセス起動時に取り込んだデバイス情報と OS 側の現状がズレる。
    その結果 ``Pa_OpenStream`` が ``-9986 (paInternalError)`` を返し
    InputStream の生成に失敗する。

    sounddevice は内部に ``_terminate`` / ``_initialize`` を持っており、
    これを叩くと PortAudio を一旦 Pa_Terminate → Pa_Initialize して
    デバイス列挙からやり直してくれる。失敗してもプロセスを巻き込まない
    ように個別に try する。
    """
    try:
        sd._terminate()
    except Exception:
        _logger.exception("sd._terminate() raised (ignored)")
    try:
        sd._initialize()
    except Exception:
        _logger.exception("sd._initialize() raised (ignored)")


class Recorder:
    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self._stream: sd.InputStream | None = None
        self._buffer: list[np.ndarray] = []
        self._lock = threading.Lock()

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    def _callback(self, indata, frames, time_info, status):
        with self._lock:
            self._buffer.append(indata.copy().reshape(-1))

    def _open_stream(self) -> sd.InputStream:
        return sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )

    def start(self) -> None:
        if self.is_recording:
            return
        with self._lock:
            self._buffer = []
        try:
            stream = self._open_stream()
        except sd.PortAudioError as first_err:
            # AirPods 切替・既定デバイス変更などで PortAudio の内部キャッシュが
            # 陳腐化しているケース。1 度だけ recycle して再試行する。
            _logger.warning(
                "InputStream open failed (%s); recycling PortAudio and retrying",
                first_err,
            )
            _recycle_portaudio()
            stream = self._open_stream()
        self._stream = stream
        self._stream.start()

    def stop(self) -> np.ndarray:
        if not self.is_recording:
            return np.zeros(0, dtype=np.float32)
        stream = self._stream
        self._stream = None
        stream.stop()
        stream.close()
        with self._lock:
            return (
                np.concatenate(self._buffer)
                if self._buffer
                else np.zeros(0, dtype=np.float32)
            )
