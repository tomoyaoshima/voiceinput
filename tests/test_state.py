from voiceinput.state import RecordingMode, RecordingStateMachine


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, sec: float) -> None:
        self.t += sec


def make_sm(threshold: float = 0.35):
    starts: list[float] = []
    stops: list[float] = []
    latches: list[float] = []
    clock = FakeClock()
    sm = RecordingStateMachine(
        on_start=lambda: starts.append(clock.t),
        on_stop=lambda: stops.append(clock.t),
        on_latch=lambda: latches.append(clock.t),
        short_press_threshold=threshold,
        time_fn=clock,
    )
    return sm, clock, starts, stops, latches


def test_long_press_starts_and_stops_on_release():
    sm, clock, starts, stops, latches = make_sm()
    sm.on_press()
    assert starts == [0.0]
    assert sm.mode == RecordingMode.HELD
    clock.advance(1.0)
    sm.on_release()
    assert stops == [1.0]
    assert latches == []
    assert sm.mode == RecordingMode.IDLE


def test_short_press_latches_recording_and_keeps_running():
    sm, clock, starts, stops, latches = make_sm()
    sm.on_press()
    clock.advance(0.1)  # 短押し
    sm.on_release()
    assert stops == []          # まだ停止していない
    assert latches == [0.1]
    assert sm.mode == RecordingMode.LATCHED


def test_press_in_latched_mode_stops_recording():
    sm, clock, starts, stops, latches = make_sm()
    sm.on_press()
    clock.advance(0.1)
    sm.on_release()  # LATCHED へ
    clock.advance(5.0)
    sm.on_press()    # 停止
    assert stops == [5.1]
    assert sm.mode == RecordingMode.IDLE


def test_release_in_latched_mode_is_ignored():
    sm, clock, starts, stops, latches = make_sm()
    sm.on_press()
    clock.advance(0.1)
    sm.on_release()  # LATCHED
    sm.on_release()  # ignore
    assert stops == []
    assert sm.mode == RecordingMode.LATCHED


def test_repeated_press_in_held_mode_ignored():
    sm, clock, starts, stops, latches = make_sm()
    sm.on_press()
    sm.on_press()  # repeat (キーリピート想定)
    sm.on_press()
    assert starts == [0.0]    # 1 回しか発火しない
    assert sm.mode == RecordingMode.HELD


def test_threshold_boundary_treated_as_long_press():
    sm, clock, starts, stops, latches = make_sm(threshold=0.35)
    sm.on_press()
    clock.advance(0.35)  # 等しい場合は long press 扱い
    sm.on_release()
    assert stops == [0.35]
    assert latches == []
    assert sm.mode == RecordingMode.IDLE


def test_force_idle_resets_without_calling_callbacks():
    """recorder.start 失敗時の rollback 用 API。

    HELD 状態で force_idle() を呼んでも on_stop は走らず、IDLE に戻る。
    続いて on_release が来ても無視される (= 二重 stop ログを生まない)。
    """
    sm, clock, starts, stops, latches = make_sm()
    sm.on_press()
    assert sm.mode == RecordingMode.HELD
    sm.force_idle()
    assert sm.mode == RecordingMode.IDLE
    # callback は呼ばれていない
    assert stops == []
    assert latches == []
    # この後に来る release は IDLE 経路で無視される
    sm.on_release()
    assert stops == []
