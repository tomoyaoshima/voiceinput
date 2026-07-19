import pytest
from pynput import keyboard

from voiceinput.hotkey import HotkeyManager


def test_parse_alt_space():
    mods, main = HotkeyManager._parse_combo("<alt>+<space>")
    assert mods == {"alt"}
    assert main == keyboard.Key.space


def test_parse_ctrl_alt_letter():
    mods, main = HotkeyManager._parse_combo("<ctrl>+<alt>+a")
    assert mods == {"ctrl", "alt"}
    assert isinstance(main, keyboard.KeyCode)
    assert main.char == "a"


def test_parse_unknown_key_raises():
    with pytest.raises(ValueError):
        HotkeyManager._parse_combo("<alt>+<nonsense>")


def test_parse_no_main_raises():
    with pytest.raises(ValueError):
        HotkeyManager._parse_combo("<alt>+<ctrl>")


def test_parse_solo_cmd_r():
    mods, main = HotkeyManager._parse_combo("<cmd_r>")
    assert mods == set()
    assert main == keyboard.Key.cmd_r


def _make_mgr(combo: str):
    mgr = HotkeyManager()
    events: list[str] = []
    mgr.register_press_release(
        combo,
        on_press=lambda: events.append("press"),
        on_release=lambda: events.append("release"),
    )
    return mgr, events


# --- standard combo ---


def test_press_release_sequence_dispatches_callbacks():
    mgr, events = _make_mgr("<alt>+<space>")
    mgr._on_key_press(keyboard.Key.alt)
    mgr._on_key_press(keyboard.Key.space)
    mgr._on_key_release(keyboard.Key.space)
    mgr._on_key_release(keyboard.Key.alt)
    assert events == ["press", "release"]


def test_releasing_modifier_first_also_fires_release():
    mgr, events = _make_mgr("<alt>+<space>")
    mgr._on_key_press(keyboard.Key.alt)
    mgr._on_key_press(keyboard.Key.space)
    mgr._on_key_release(keyboard.Key.alt)
    assert events == ["press", "release"]


def test_space_as_nbsp_still_matches():
    mgr, events = _make_mgr("<alt>+<space>")
    mgr._on_key_press(keyboard.Key.alt)
    nbsp = keyboard.KeyCode.from_char("\xa0")
    mgr._on_key_press(nbsp)
    mgr._on_key_release(nbsp)
    mgr._on_key_release(keyboard.Key.alt)
    assert events == ["press", "release"]


def test_main_key_without_modifier_is_ignored():
    mgr, events = _make_mgr("<alt>+<space>")
    mgr._on_key_press(keyboard.Key.space)
    mgr._on_key_release(keyboard.Key.space)
    assert events == []


def test_repeated_press_only_fires_once():
    mgr, events = _make_mgr("<alt>+<space>")
    mgr._on_key_press(keyboard.Key.alt)
    mgr._on_key_press(keyboard.Key.space)
    mgr._on_key_press(keyboard.Key.space)
    mgr._on_key_press(keyboard.Key.space)
    mgr._on_key_release(keyboard.Key.space)
    assert events == ["press", "release"]


# --- solo modifier (Right Cmd) ---


def test_solo_cmd_r_press_release_fires_press_then_release():
    mgr, events = _make_mgr("<cmd_r>")
    mgr._on_key_press(keyboard.Key.cmd_r)
    mgr._on_key_release(keyboard.Key.cmd_r)
    assert events == ["press", "release"]


def test_solo_cmd_r_combined_with_other_key_is_canceled():
    """Cmd+C のように他キーと組み合わさったらホットキーとして反応しない"""
    mgr, events = _make_mgr("<cmd_r>")
    mgr._on_key_press(keyboard.Key.cmd_r)
    c = keyboard.KeyCode.from_char("c")
    mgr._on_key_press(c)
    mgr._on_key_release(c)
    mgr._on_key_release(keyboard.Key.cmd_r)
    assert events == []


def test_solo_cmd_r_repeated_press_doesnt_double_fire():
    mgr, events = _make_mgr("<cmd_r>")
    mgr._on_key_press(keyboard.Key.cmd_r)
    mgr._on_key_press(keyboard.Key.cmd_r)  # OS のキーリピート想定
    mgr._on_key_release(keyboard.Key.cmd_r)
    assert events == ["press", "release"]


def test_solo_cmd_r_only_responds_to_target_modifier():
    """<cmd_r> 設定中に <cmd_l> を押しても無反応"""
    mgr, events = _make_mgr("<cmd_r>")
    mgr._on_key_press(keyboard.Key.cmd_l)
    mgr._on_key_release(keyboard.Key.cmd_l)
    assert events == []


def test_solo_cmd_r_after_canceled_can_still_be_used():
    """1 度キャンセルされても次の単独押しは正常動作する"""
    mgr, events = _make_mgr("<cmd_r>")
    # Cmd+V でキャンセル
    mgr._on_key_press(keyboard.Key.cmd_r)
    v = keyboard.KeyCode.from_char("v")
    mgr._on_key_press(v)
    mgr._on_key_release(v)
    mgr._on_key_release(keyboard.Key.cmd_r)
    assert events == []
    # 単独押し
    mgr._on_key_press(keyboard.Key.cmd_r)
    mgr._on_key_release(keyboard.Key.cmd_r)
    assert events == ["press", "release"]


def test_is_solo_modifier_property():
    mgr, _ = _make_mgr("<cmd_r>")
    assert mgr.is_solo_modifier is True
    mgr2, _ = _make_mgr("<alt>+<space>")
    assert mgr2.is_solo_modifier is False


# --- Esc cancel ---


def _make_mgr_with_cancel(combo: str):
    """register_cancel も登録した manager を返す。"""
    mgr = HotkeyManager()
    events: list[str] = []
    mgr.register_press_release(
        combo,
        on_press=lambda: events.append("press"),
        on_release=lambda: events.append("release"),
    )
    mgr.register_cancel(lambda: events.append("cancel"))
    return mgr, events


def test_esc_alone_fires_cancel():
    """修飾キーなしの Esc 単独押しで on_cancel が呼ばれる"""
    mgr, events = _make_mgr_with_cancel("<cmd_r>")
    mgr._on_key_press(keyboard.Key.esc)
    assert "cancel" in events


def test_esc_with_modifier_does_not_fire_cancel():
    """Cmd+Esc は通常 (タスク強制終了など) に使われるので無視する"""
    mgr, events = _make_mgr_with_cancel("<alt>+<space>")
    mgr._on_key_press(keyboard.Key.cmd)
    mgr._on_key_press(keyboard.Key.esc)
    assert "cancel" not in events


def test_esc_as_main_key_does_not_fire_cancel():
    """ユーザーが hotkey に Esc を割り当てている場合は cancel しない
    (press/release dispatch のみ走る、cancel と二重発火しない)。"""
    mgr = HotkeyManager()
    events: list[str] = []
    mgr.register_press_release(
        "<esc>",
        on_press=lambda: events.append("press"),
        on_release=lambda: events.append("release"),
    )
    mgr.register_cancel(lambda: events.append("cancel"))
    mgr._on_key_press(keyboard.Key.esc)
    mgr._on_key_release(keyboard.Key.esc)
    assert "cancel" not in events


def test_no_cancel_callback_registered_is_safe():
    """register_cancel を呼ばずに Esc が押されても落ちない (default は no-op)"""
    mgr = HotkeyManager()
    mgr.register_press_release(
        "<cmd_r>",
        on_press=lambda: None,
        on_release=lambda: None,
    )
    # 例外なく完了することが期待
    mgr._on_key_press(keyboard.Key.esc)


def test_esc_during_recording_combo_does_not_break_subsequent_combo():
    """Esc cancel と通常の combo dispatch は独立。Esc 後も Alt+Space は動く。"""
    mgr, events = _make_mgr_with_cancel("<alt>+<space>")
    mgr._on_key_press(keyboard.Key.esc)
    assert events == ["cancel"]
    # Esc 後に通常の Alt+Space が機能する
    mgr._on_key_press(keyboard.Key.alt)
    mgr._on_key_press(keyboard.Key.space)
    mgr._on_key_release(keyboard.Key.space)
    mgr._on_key_release(keyboard.Key.alt)
    assert events[1:] == ["press", "release"]
