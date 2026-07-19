"""ユーザーがメニューバーから行った選択を永続化する。

`~/Library/Application Support/voiceinput/state.json` に保存。
保存対象は menu bar から動的に変えられる項目のみ:

- ``format_mode``: clean / mail / raw
- ``llm_model``: qwen2.5:7b / qwen2.5:14b / phi4:latest 等
- ``manual_vocabulary``: GUI から追加された語彙リスト
  (config.yaml の ``vocabulary.manual`` と合算される)

config.yaml で明示指定された項目があればそちらが state より優先される
(優先順位は ``app.py`` 側で解決)。state.json は壊れていれば無視して既定に
戻す。書き込みは tmp → rename の atomic 置換を使う。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


_SENTINEL: object = object()


@dataclass
class UserState:
    format_mode: str | None = None
    llm_model: str | None = None
    # Phase G: STT (Whisper) モデルの menu 選択。None = config デフォルト。
    whisper_model: str | None = None
    manual_vocabulary: list[str] = field(default_factory=list)
    # Phase F: 画面コンテキスト認識の ON/OFF (menu トグル)。
    # None = ユーザー未設定 → config のデフォルトに従う。
    screen_context_enabled: bool | None = None


def default_state_path() -> Path:
    return (
        Path("~/Library/Application Support/voiceinput/state.json").expanduser()
    )


class UserStateStore:
    """state.json の読み書きを担当する。

    failure-soft: 読めない / 壊れている state はすべて空 UserState() に倒す。
    書き込みは tmp ファイル経由で atomic 置換し、書き込み中の読み込みを防ぐ。
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_state_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> UserState:
        if not self.path.exists():
            return UserState()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return UserState()
        if not isinstance(data, dict):
            return UserState()
        return UserState(
            format_mode=_clean_str(data.get("format_mode")),
            llm_model=_clean_str(data.get("llm_model")),
            whisper_model=_clean_str(data.get("whisper_model")),
            manual_vocabulary=_clean_str_list(data.get("manual_vocabulary")),
            screen_context_enabled=_clean_bool(data.get("screen_context_enabled")),
        )

    def save(self, state: UserState) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(asdict(state), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def update(
        self,
        *,
        format_mode: str | None | object = _SENTINEL,
        llm_model: str | None | object = _SENTINEL,
        whisper_model: str | None | object = _SENTINEL,
        manual_vocabulary: list[str] | None = None,
        screen_context_enabled: bool | None | object = _SENTINEL,
    ) -> UserState:
        """既存値を保ったまま指定フィールドだけ書き換える。

        フィールドを省略 (``_SENTINEL`` の既定値) すると "触らない"。
        明示的に ``None`` を渡すとクリアできる。``manual_vocabulary`` は
        list で渡せば置換、``None`` で省略 (= 触らない) を意味する。
        ``screen_context_enabled`` は True/False で設定、None でクリア
        (= config デフォルトに従う)、省略で触らない。
        """
        current = self.load()
        new_state = UserState(
            format_mode=(
                current.format_mode if format_mode is _SENTINEL else format_mode  # type: ignore[arg-type]
            ),
            llm_model=(
                current.llm_model if llm_model is _SENTINEL else llm_model  # type: ignore[arg-type]
            ),
            whisper_model=(
                current.whisper_model if whisper_model is _SENTINEL else whisper_model  # type: ignore[arg-type]
            ),
            manual_vocabulary=(
                list(current.manual_vocabulary)
                if manual_vocabulary is None
                else list(manual_vocabulary)
            ),
            screen_context_enabled=(
                current.screen_context_enabled
                if screen_context_enabled is _SENTINEL
                else screen_context_enabled  # type: ignore[arg-type]
            ),
        )
        self.save(new_state)
        return new_state


def _clean_str(value: object) -> str | None:
    """JSON 値が空文字や非文字列なら None に倒す。"""
    if not isinstance(value, str):
        return None
    s = value.strip()
    return s or None


def _clean_bool(value: object) -> bool | None:
    """JSON 値が bool でなければ None に倒す (= 未設定扱い)。"""
    if isinstance(value, bool):
        return value
    return None


def _clean_str_list(value: object) -> list[str]:
    """JSON 値が list of str でなければ空リストに倒す。重複と空白も除く。"""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out
