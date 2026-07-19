import json

from voiceinput.history import History, HistoryEntry, make_entry


def test_append_and_list_returns_entries_in_chronological_order(tmp_path):
    h = History(tmp_path / "history.jsonl", max_entries=10)
    h.append(make_entry(mode="clean", text="一つめ"))
    h.append(make_entry(mode="clean", text="二つめ"))
    items = h.list()
    assert [e.text for e in items] == ["一つめ", "二つめ"]


def test_list_with_limit_returns_recent_n(tmp_path):
    h = History(tmp_path / "history.jsonl", max_entries=10)
    for i in range(5):
        h.append(make_entry(mode="clean", text=f"text {i}"))
    items = h.list(limit=2)
    assert [e.text for e in items] == ["text 3", "text 4"]


def test_max_entries_drops_oldest_when_overflowing(tmp_path):
    h = History(tmp_path / "history.jsonl", max_entries=3)
    for i in range(5):
        h.append(make_entry(mode="clean", text=f"text {i}"))
    items = h.list()
    assert len(items) == 3
    assert [e.text for e in items] == ["text 2", "text 3", "text 4"]


def test_clear_empties_file(tmp_path):
    h = History(tmp_path / "history.jsonl", max_entries=10)
    h.append(make_entry(mode="clean", text="hello"))
    h.clear()
    assert h.list() == []


def test_corrupt_line_is_skipped(tmp_path):
    path = tmp_path / "history.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "not-json\n"
        + json.dumps({"timestamp": 1.0, "mode": "clean", "text": "ok"})
        + "\n"
    )
    h = History(path)
    items = h.list()
    assert len(items) == 1
    assert items[0].text == "ok"


def test_unicode_is_preserved(tmp_path):
    h = History(tmp_path / "history.jsonl")
    h.append(make_entry(mode="clean", text="日本語テキスト 🎤"))
    raw = (tmp_path / "history.jsonl").read_text(encoding="utf-8")
    # ensure_ascii=False で書き込まれている
    assert "日本語テキスト" in raw
    assert "🎤" in raw


def test_make_entry_populates_all_fields(tmp_path):
    e = make_entry(mode="mail", text="A", raw_text="B", audio_sec=1.5, stt_sec=0.4, llm_sec=0.9)
    assert e.mode == "mail"
    assert e.text == "A"
    assert e.raw_text == "B"
    assert e.audio_sec == 1.5
    assert e.stt_sec == 0.4
    assert e.llm_sec == 0.9
    assert e.timestamp > 0


def test_dataclass_optional_fields_roundtrip(tmp_path):
    """既存の旧フォーマット (extras なし) も読み込めること"""
    path = tmp_path / "history.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"timestamp": 1.0, "mode": "clean", "text": "old"})
        + "\n"
    )
    h = History(path)
    items = h.list()
    assert items[0].text == "old"
    assert items[0].extras == {}
