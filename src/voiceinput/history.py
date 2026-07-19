"""音声入力の履歴を JSONL で永続化する。

`~/Library/Application Support/voiceinput/history.jsonl` に 1 行 1 エントリで append。
直近 N 件 (デフォルト 200) を超えたら古いものから削除する。

menu bar の History サブメニューに直近の整形結果を表示し、クリックで再コピー
できるようにするための裏方。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class HistoryEntry:
    timestamp: float            # epoch 秒
    mode: str                   # "raw" / "clean" / "mail"
    text: str                   # 整形後 (実際にコピー/ペーストしたもの)
    raw_text: str = ""          # 整形前 (STT 結果)
    audio_sec: float = 0.0
    stt_sec: float = 0.0
    llm_sec: float = 0.0
    extras: dict = field(default_factory=dict)


def default_history_path() -> Path:
    return (
        Path("~/Library/Application Support/voiceinput/history.jsonl").expanduser()
    )


class History:
    def __init__(self, path: Path | None = None, max_entries: int = 200) -> None:
        self.path = path or default_history_path()
        self.max_entries = max_entries
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: HistoryEntry) -> None:
        existing = self._read_all()
        # 直近 max_entries-1 件 + 今回の 1 件
        trimmed = existing[-(self.max_entries - 1) :] if self.max_entries > 1 else []
        trimmed.append(entry)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for e in trimmed:
                f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")
        tmp.replace(self.path)

    def list(self, limit: int = 50) -> list[HistoryEntry]:
        entries = self._read_all()
        if limit <= 0:
            return entries
        return entries[-limit:]

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()

    def _read_all(self) -> list[HistoryEntry]:
        if not self.path.exists():
            return []
        out: list[HistoryEntry] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    out.append(HistoryEntry(**data))
                except (json.JSONDecodeError, TypeError):
                    continue
        return out


def make_entry(
    mode: str,
    text: str,
    raw_text: str = "",
    audio_sec: float = 0.0,
    stt_sec: float = 0.0,
    llm_sec: float = 0.0,
) -> HistoryEntry:
    return HistoryEntry(
        timestamp=time.time(),
        mode=mode,
        text=text,
        raw_text=raw_text,
        audio_sec=audio_sec,
        stt_sec=stt_sec,
        llm_sec=llm_sec,
    )
