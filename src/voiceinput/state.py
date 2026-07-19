import time
from enum import Enum
from typing import Callable


class RecordingMode(Enum):
    IDLE = "idle"
    HELD = "held"        # キー押下中 (push-to-talk)
    LATCHED = "latched"  # 短押しトグルで継続録音中、次の押下で停止


class RecordingStateMachine:
    """AquaVoice 風の状態遷移を管理する。

    - IDLE で press → 即録音開始 + HELD
    - HELD で release:
      - 押下時間 < short_press_threshold なら LATCHED (短押し → トグル ON)
      - そうでなければ stop + IDLE (長押し → push-to-talk 終了)
    - LATCHED で press → stop + IDLE (もう一度押すと停止)
    - LATCHED で release は無視
    """

    def __init__(
        self,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
        on_latch: Callable[[], None] = lambda: None,
        short_press_threshold: float = 0.35,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_latch = on_latch
        self.short_press_threshold = short_press_threshold
        self._time = time_fn
        self.mode = RecordingMode.IDLE
        self._press_started_at = 0.0

    def on_press(self) -> None:
        if self.mode == RecordingMode.IDLE:
            self._press_started_at = self._time()
            self.mode = RecordingMode.HELD
            self.on_start()
        elif self.mode == RecordingMode.LATCHED:
            self.mode = RecordingMode.IDLE
            self.on_stop()
        # HELD で press は重複イベント (キーリピート抑止) → 無視

    def on_release(self) -> None:
        if self.mode == RecordingMode.HELD:
            held_for = self._time() - self._press_started_at
            if held_for < self.short_press_threshold:
                self.mode = RecordingMode.LATCHED
                self.on_latch()
            else:
                self.mode = RecordingMode.IDLE
                self.on_stop()
        # LATCHED / IDLE での release は無視

    def force_idle(self) -> None:
        """on_start / on_stop を呼ばずに IDLE に戻す。

        recorder.start() が例外を投げた直後など、本来 on_start で始まる
        べき状態に到達できなかった時の rollback 用。次の on_release は
        IDLE を見て無視するので、開放感のある状態に戻る。
        """
        self.mode = RecordingMode.IDLE
        self._press_started_at = 0.0
