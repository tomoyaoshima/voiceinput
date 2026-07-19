import logging
import traceback
from typing import Callable

from pynput import keyboard

_logger = logging.getLogger("voiceinput.hotkey")

_MOD_NAMES = {
    keyboard.Key.alt: "alt",
    keyboard.Key.alt_l: "alt",
    keyboard.Key.alt_r: "alt",
    keyboard.Key.alt_gr: "alt",
    keyboard.Key.ctrl: "ctrl",
    keyboard.Key.ctrl_l: "ctrl",
    keyboard.Key.ctrl_r: "ctrl",
    keyboard.Key.cmd: "cmd",
    keyboard.Key.cmd_l: "cmd",
    keyboard.Key.cmd_r: "cmd",
    keyboard.Key.shift: "shift",
    keyboard.Key.shift_l: "shift",
    keyboard.Key.shift_r: "shift",
}

# combo の main key として「修飾キー単独押し」も許可するキー一覧
_SOLO_MODIFIER_KEYS = {
    keyboard.Key.cmd_l,
    keyboard.Key.cmd_r,
    keyboard.Key.alt_l,
    keyboard.Key.alt_r,
    keyboard.Key.ctrl_l,
    keyboard.Key.ctrl_r,
    keyboard.Key.shift_l,
    keyboard.Key.shift_r,
}

# macOS で Option+Space を押すと Space が non-breaking space (\xa0) として届くので両方を許容する
_SPACE_CHARS = {" ", "\xa0"}


class HotkeyManager:
    """combo (例: "<alt>+<space>" や "<cmd_r>") の押下/解放をコールバックする。

    pynput の GlobalHotKeys は短押し1回のトグルしか扱えないので、
    press/release を区別したい push-to-talk 用に Listener を直接使う。

    combo が修飾キー単独 (`<cmd_r>` 等) の場合は、他のキーと組み合わされたら
    キャンセルし、単独で押して離した時にだけ on_press → on_release を順番に発火する
    トグル動作になる (Cmd+C などの通常ショートカットを邪魔しない)。
    """

    def __init__(self) -> None:
        self._listener: keyboard.Listener | None = None
        self._required_mods: set[str] = set()
        self._main_key: keyboard.Key | keyboard.KeyCode | None = None
        self._on_press: Callable[[], None] = lambda: None
        self._on_release: Callable[[], None] = lambda: None
        self._on_cancel: Callable[[], None] | None = None
        self._mods_held: set[str] = set()
        self._combo_active = False
        # solo modifier 用の状態
        self._solo_pending = False
        self._solo_canceled = False

    def register_press_release(
        self,
        combo: str,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
    ) -> None:
        mods, main = self._parse_combo(combo)
        self._required_mods = mods
        self._main_key = main
        self._on_press = on_press
        self._on_release = on_release

    def register_cancel(self, on_cancel: Callable[[], None]) -> None:
        """Esc 単独押下を on_cancel に通知する (録音キャンセル用)。

        条件:
        - Esc キーが押された
        - 修飾キーが 1 つも押されていない (Cmd+Esc 等は無視)
        - Esc 自体が register_press_release の main key になっていない
          (ユーザーが Esc を hotkey に割り当てているケースの誤発火を避ける)

        コールバックは Listener thread から呼ばれる。pynput は passive で
        Esc を「奪わない」ので、他のアプリの Esc 動作 (ダイアログ閉じる等)
        は通常通り動く。
        """
        self._on_cancel = on_cancel

    @property
    def is_solo_modifier(self) -> bool:
        return not self._required_mods and self._main_key in _SOLO_MODIFIER_KEYS

    @staticmethod
    def _parse_combo(combo: str) -> tuple[set[str], keyboard.Key | keyboard.KeyCode]:
        mods: set[str] = set()
        main: keyboard.Key | keyboard.KeyCode | None = None
        for raw in combo.split("+"):
            token = raw.strip()
            if token.startswith("<") and token.endswith(">"):
                name = token[1:-1].lower()
                if name in {"alt", "ctrl", "cmd", "shift"}:
                    mods.add(name)
                else:
                    key_attr = getattr(keyboard.Key, name, None)
                    if key_attr is None:
                        raise ValueError(f"unknown key in combo: {token}")
                    main = key_attr
            else:
                if len(token) != 1:
                    raise ValueError(f"unsupported combo token: {token}")
                main = keyboard.KeyCode.from_char(token)
        if main is None:
            raise ValueError(f"no main key in combo: {combo}")
        return mods, main

    def _matches_main(self, key) -> bool:
        if self._main_key is None:
            return False
        if key == self._main_key:
            return True
        if self._main_key == keyboard.Key.space:
            char = getattr(key, "char", None)
            if char in _SPACE_CHARS:
                return True
            if getattr(key, "vk", None) == 49:  # macOS Space VK
                return True
        return False

    # --- standard combo (modifier(s) + main key) ---

    def _on_combo_key_press(self, key) -> None:
        mod = _MOD_NAMES.get(key)
        if mod is not None:
            self._mods_held.add(mod)
            return
        if self._matches_main(key) and self._required_mods <= self._mods_held:
            if not self._combo_active:
                self._combo_active = True
                self._safe_call(self._on_press)

    def _on_combo_key_release(self, key) -> None:
        mod = _MOD_NAMES.get(key)
        if mod is not None:
            self._mods_held.discard(mod)
            if self._combo_active:
                self._combo_active = False
                self._safe_call(self._on_release)
            return
        if self._matches_main(key) and self._combo_active:
            self._combo_active = False
            self._safe_call(self._on_release)

    # --- solo modifier (e.g. <cmd_r> alone) ---

    def _on_solo_key_press(self, key) -> None:
        if key == self._main_key:
            if not self._solo_pending:
                self._solo_pending = True
                self._solo_canceled = False
            return
        # Right Cmd 押下中に他のキーが押された → 通常ショートカットとして使われた
        if self._solo_pending:
            self._solo_canceled = True

    def _on_solo_key_release(self, key) -> None:
        if key == self._main_key and self._solo_pending:
            confirmed = not self._solo_canceled
            self._solo_pending = False
            self._solo_canceled = False
            if confirmed:
                # press → release を順番に発火 (state machine で「短押し」とみなされる)
                self._safe_call(self._on_press)
                self._safe_call(self._on_release)

    # --- cancel (Esc 単独押し) ---

    def _maybe_fire_cancel(self, key) -> None:
        """Esc 単独押しなら on_cancel を発火する。

        副次的な dispatch (solo / combo) には影響しない。例えば Esc が
        通常 hotkey として登録されていれば下の dispatch も走るが、Esc を
        main key にしているのに cancel も鳴らすのは混乱の元なので明示
        スキップする。
        """
        if self._on_cancel is None:
            return
        if key != keyboard.Key.esc:
            return
        if self._main_key == keyboard.Key.esc:
            return
        if self._mods_held:
            return
        self._safe_call(self._on_cancel)

    # --- dispatch ---

    def _on_key_press(self, key) -> None:
        self._maybe_fire_cancel(key)
        if self.is_solo_modifier:
            self._on_solo_key_press(key)
        else:
            self._on_combo_key_press(key)

    def _on_key_release(self, key) -> None:
        if self.is_solo_modifier:
            self._on_solo_key_release(key)
        else:
            self._on_combo_key_release(key)

    @staticmethod
    def _safe_call(fn: Callable[[], None]) -> None:
        """listener thread を絶対に止めないが、例外は必ずログに残す。

        以前は silent swallow していたため、recorder.start の失敗が
        誰にも観測されず "0 samples" ログだけが残る厄介な silent
        failure を生んでいた。listener が落ちないという目的は維持し、
        traceback だけは必ず出す。
        """
        try:
            fn()
        except Exception:
            # 万一 logger 設定が壊れていても traceback は失わない
            try:
                _logger.exception("hotkey callback raised")
            except Exception:
                traceback.print_exc()

    def start(self) -> None:
        if self._listener is not None:
            return
        self._listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
