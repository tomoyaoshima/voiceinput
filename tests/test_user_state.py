from pathlib import Path

from voiceinput.user_state import UserState, UserStateStore


def test_load_returns_empty_when_file_missing(tmp_path: Path):
    store = UserStateStore(tmp_path / "state.json")
    state = store.load()
    assert state == UserState()


def test_save_then_load_round_trip(tmp_path: Path):
    store = UserStateStore(tmp_path / "state.json")
    store.save(UserState(format_mode="mail", llm_model="qwen2.5:14b"))
    loaded = store.load()
    assert loaded.format_mode == "mail"
    assert loaded.llm_model == "qwen2.5:14b"


def test_update_keeps_unchanged_fields(tmp_path: Path):
    store = UserStateStore(tmp_path / "state.json")
    store.save(UserState(format_mode="clean", llm_model="qwen2.5:7b"))
    # llm_model だけ更新 → format_mode は維持される
    store.update(llm_model="qwen2.5:14b")
    loaded = store.load()
    assert loaded.format_mode == "clean"
    assert loaded.llm_model == "qwen2.5:14b"


def test_corrupt_file_falls_back_to_empty(tmp_path: Path):
    path = tmp_path / "state.json"
    path.write_text("{ invalid json")
    store = UserStateStore(path)
    assert store.load() == UserState()


def test_non_dict_top_level_falls_back(tmp_path: Path):
    path = tmp_path / "state.json"
    path.write_text("[1, 2, 3]")  # JSON だが dict ではない
    store = UserStateStore(path)
    assert store.load() == UserState()


def test_empty_string_values_become_none(tmp_path: Path):
    path = tmp_path / "state.json"
    path.write_text('{"format_mode": "", "llm_model": "  "}')
    store = UserStateStore(path)
    state = store.load()
    assert state.format_mode is None
    assert state.llm_model is None


def test_manual_vocabulary_persists(tmp_path: Path):
    """GUI で追加した語が state.json に list として保存されること。"""
    store = UserStateStore(tmp_path / "state.json")
    store.update(manual_vocabulary=["voiceinput", "Codex", "山田太郎"])
    loaded = store.load()
    assert loaded.manual_vocabulary == ["voiceinput", "Codex", "山田太郎"]


def test_manual_vocabulary_dedupes_and_strips(tmp_path: Path):
    """壊れた / 重複した値は load 側で正規化される (defense in depth)。"""
    path = tmp_path / "state.json"
    path.write_text(
        '{"manual_vocabulary": ["voiceinput", " voiceinput ", "", "Codex", null, 42]}'
    )
    store = UserStateStore(path)
    state = store.load()
    assert state.manual_vocabulary == ["voiceinput", "Codex"]


def test_update_preserves_other_fields_when_only_vocabulary_changed(tmp_path: Path):
    """manual_vocabulary だけ更新しても format_mode / llm_model は維持される。"""
    store = UserStateStore(tmp_path / "state.json")
    store.save(UserState(format_mode="mail", llm_model="qwen2.5:14b"))
    store.update(manual_vocabulary=["NewWord"])
    loaded = store.load()
    assert loaded.format_mode == "mail"
    assert loaded.llm_model == "qwen2.5:14b"
    assert loaded.manual_vocabulary == ["NewWord"]


def test_update_explicit_none_clears_field(tmp_path: Path):
    """明示的に None を渡すとそのフィールドだけクリアできる (sentinel が "省略" の意味)。"""
    store = UserStateStore(tmp_path / "state.json")
    store.save(UserState(format_mode="mail", llm_model="qwen2.5:14b"))
    store.update(format_mode=None)
    loaded = store.load()
    assert loaded.format_mode is None
    assert loaded.llm_model == "qwen2.5:14b"  # 触ってないので維持


# --- Phase F: screen_context_enabled の永続化 ---


def test_screen_context_default_is_none(tmp_path: Path):
    """未設定は None (= config デフォルトに従う)。"""
    store = UserStateStore(tmp_path / "state.json")
    assert store.load().screen_context_enabled is None


def test_screen_context_persists_true_false(tmp_path: Path):
    store = UserStateStore(tmp_path / "state.json")
    store.update(screen_context_enabled=False)
    assert store.load().screen_context_enabled is False
    store.update(screen_context_enabled=True)
    assert store.load().screen_context_enabled is True


def test_screen_context_update_preserves_other_fields(tmp_path: Path):
    store = UserStateStore(tmp_path / "state.json")
    store.save(UserState(format_mode="clean", llm_model="gemma4:e4b"))
    store.update(screen_context_enabled=False)
    loaded = store.load()
    assert loaded.format_mode == "clean"
    assert loaded.llm_model == "gemma4:e4b"
    assert loaded.screen_context_enabled is False


def test_screen_context_non_bool_in_json_becomes_none(tmp_path: Path):
    path = tmp_path / "state.json"
    path.write_text('{"screen_context_enabled": "yes"}')
    store = UserStateStore(path)
    assert store.load().screen_context_enabled is None


# --- Phase G: whisper_model の永続化 ---


def test_whisper_model_default_is_none(tmp_path: Path):
    store = UserStateStore(tmp_path / "state.json")
    assert store.load().whisper_model is None


def test_whisper_model_persists(tmp_path: Path):
    store = UserStateStore(tmp_path / "state.json")
    store.update(whisper_model="mlx-community/whisper-large-v3-mlx")
    assert store.load().whisper_model == "mlx-community/whisper-large-v3-mlx"


def test_whisper_model_update_preserves_other_fields(tmp_path: Path):
    store = UserStateStore(tmp_path / "state.json")
    store.save(UserState(format_mode="clean", llm_model="gemma4:e4b"))
    store.update(whisper_model="kaiinui/kotoba-whisper-v2.0-mlx")
    loaded = store.load()
    assert loaded.format_mode == "clean"
    assert loaded.llm_model == "gemma4:e4b"
    assert loaded.whisper_model == "kaiinui/kotoba-whisper-v2.0-mlx"
