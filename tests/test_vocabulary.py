from pathlib import Path

from voiceinput.history import History, HistoryEntry
from voiceinput.vocabulary import (
    DEFAULT_MAX_CHARS,
    PROMPT_LIST_MAX_CHARS,
    VocabularyBuilder,
    build_initial_prompt,
    compose_initial_prompt,
    extract_terms,
    parse_vocabulary_lines,
    render_prompt_list,
    render_vocabulary_file,
)


def _seed_history(tmp_path: Path, entries: list[HistoryEntry]) -> History:
    """テスト用の History を一発で書き込んで返す。"""
    path = tmp_path / "history.jsonl"
    history = History(path=path, max_entries=1000)
    for e in entries:
        history.append(e)
    return history


def _entry(text: str, raw_text: str = "") -> HistoryEntry:
    return HistoryEntry(timestamp=0.0, mode="clean", text=text, raw_text=raw_text)


def test_extract_terms_picks_katakana_kanji_alnum():
    text = "voiceinput でカタカナと漢字連続を抽出する Codex_v2"
    terms = extract_terms(text)
    assert "voiceinput" in terms
    assert "カタカナ" in terms
    assert "漢字連続" in terms
    assert "抽出" in terms
    assert "Codex_v2" in terms


def test_extract_terms_skips_short_runs():
    """1 文字のカタカナ・漢字や 2 文字英数字はノイズなので拾わない。"""
    text = "ア 田 is x ab"
    assert extract_terms(text) == []


def test_extract_terms_handles_empty_input():
    assert extract_terms("") == []
    assert extract_terms(None) == []  # type: ignore[arg-type]


def test_build_initial_prompt_combines_manual_and_history(tmp_path: Path):
    history = _seed_history(
        tmp_path,
        [
            _entry("カタカナと漢字の議事録"),
            _entry("カタカナでサービス名を含む"),
            _entry("voiceinput で議事録を録る"),
        ],
    )
    out = build_initial_prompt(history, ["voiceinput", "Codex"], top_n=10)
    parts = out.split()
    # 手動リストが先頭
    assert parts[0] == "voiceinput"
    assert parts[1] == "Codex"
    # 履歴頻出の "カタカナ" / "議事録" が含まれる (順不同で OK)
    assert "カタカナ" in parts
    assert "議事録" in parts


def test_manual_list_dedupes_against_history(tmp_path: Path):
    """履歴と手動リストに同じ語がある場合、手動側 1 つだけ残す。"""
    history = _seed_history(tmp_path, [_entry("voiceinput を使う")])
    out = build_initial_prompt(history, ["voiceinput"])
    assert out.split().count("voiceinput") == 1


def test_max_chars_truncates_output(tmp_path: Path):
    """200 文字超は途中で打ち切られる。"""
    long_terms = ["カタカナ単語" + str(i) for i in range(200)]
    history = _seed_history(tmp_path, [])
    out = build_initial_prompt(history, long_terms, max_chars=80)
    assert len(out) <= 80


def test_empty_history_and_manual_returns_empty(tmp_path: Path):
    history = _seed_history(tmp_path, [])
    assert build_initial_prompt(history, []) == ""


def test_build_uses_text_not_raw_text(tmp_path: Path):
    """raw_text には誤認識が混じり得るので、整形後の text のみを抽出ソースにする。"""
    history = _seed_history(
        tmp_path,
        [
            _entry(text="正解単語", raw_text="ゴカイ単語"),
            _entry(text="正解単語", raw_text="ゴカイ単語"),
        ],
    )
    out = build_initial_prompt(history, [])
    assert "正解単語" in out.split()
    # 誤認識側は採用しない
    assert "ゴカイ単語" not in out


def test_vocabulary_builder_caches_until_history_changes(tmp_path: Path):
    history = _seed_history(tmp_path, [_entry("初期テキスト")])
    builder = VocabularyBuilder(history, manual=["手動語"])
    first = builder.build()
    cache_key = builder._cache_key
    # 2 回目は同じキャッシュキーで早期 return される
    second = builder.build()
    assert first == second
    assert builder._cache_key == cache_key


def test_vocabulary_builder_force_rebuilds(tmp_path: Path):
    history = _seed_history(tmp_path, [_entry("ソリューション開発")])
    builder = VocabularyBuilder(history, manual=[])
    out1 = builder.build()
    # force=True なら必ず再構築
    out2 = builder.build(force=True)
    assert out1 == out2  # 内容は同じだが
    # キャッシュキーは更新されているはず (再呼び出し可能であればよい)
    assert builder._cache_key is not None


def test_vocabulary_builder_picks_up_new_history(tmp_path: Path):
    """history.append 後に build() を呼ぶと、新しい語が反映される。"""
    history = _seed_history(tmp_path, [_entry("最初の議題")])
    builder = VocabularyBuilder(history, manual=[])
    out1 = builder.build()
    assert "議題" in out1.split()
    # 新たに append → mtime が変わる
    history.append(_entry("追加されたサービス名"))
    out2 = builder.build()
    assert "サービス" in out2.split()


def test_default_max_chars_is_safe_under_whisper_limit():
    # Whisper initial_prompt は ~224 token、日本語は 1-2 token/char。
    # 200 文字なら最悪 400 token 程度なので余裕を見ておきたい。
    assert DEFAULT_MAX_CHARS <= 220


# ---------------------------------------------------------------------------
# compose_initial_prompt (Phase F: 画面コンテキスト語の合成)
# ---------------------------------------------------------------------------


def test_compose_puts_screen_terms_first():
    """画面語が先頭、続けて静的 vocab。"""
    out = compose_initial_prompt("既存語A 既存語B", ["山田", "案件X"])
    assert out == "山田 案件X 既存語A 既存語B"


def test_compose_dedupes_screen_vs_static():
    """画面語と静的 vocab が重複したら 1 回だけ (先頭側を維持)。"""
    out = compose_initial_prompt("山田 既存語", ["山田", "案件X"])
    # 山田 は画面側で先に出るので 1 回だけ、既存語が続く
    assert out == "山田 案件X 既存語"


def test_compose_empty_screen_terms_returns_static():
    out = compose_initial_prompt("既存語A 既存語B", [])
    assert out == "既存語A 既存語B"


def test_compose_empty_static_returns_screen_terms():
    out = compose_initial_prompt("", ["山田", "案件X"])
    assert out == "山田 案件X"


def test_compose_skips_blank_and_dupe_screen_terms():
    out = compose_initial_prompt("", ["山田", "", "  ", "山田", "案件X"])
    assert out == "山田 案件X"


def test_compose_truncates_to_max_chars():
    """画面語を優先しつつ max_chars で切り詰める (幻覚抑制の枠)。"""
    screen = ["あいうえお", "かきくけこ", "さしすせそ"]  # 各5文字
    # max_chars=12 → "あいうえお かきくけこ" (5+1+5=11) まで、3つ目は入らない
    out = compose_initial_prompt("既存語", screen, max_chars=12)
    assert out == "あいうえお かきくけこ"
    assert len(out) <= 12


# ---------------------------------------------------------------------------
# parse_vocabulary_lines / render_vocabulary_file (GUI まとめて編集)
# ---------------------------------------------------------------------------


def test_parse_lines_basic():
    text = "サンプル商会\nPayPal\n山田"
    assert parse_vocabulary_lines(text) == ["サンプル商会", "PayPal", "山田"]


def test_parse_lines_skips_blank_and_comments():
    text = "# 使い方コメント\n\nサンプル商会\n   \n# もう一つ\nPayPal\n"
    assert parse_vocabulary_lines(text) == ["サンプル商会", "PayPal"]


def test_parse_lines_strips_whitespace():
    text = "  サンプル商会  \n\tPayPal\t"
    assert parse_vocabulary_lines(text) == ["サンプル商会", "PayPal"]


def test_parse_lines_dedupes_preserving_order():
    text = "サンプル商会\nPayPal\nサンプル商会\n山田\nPayPal"
    assert parse_vocabulary_lines(text) == ["サンプル商会", "PayPal", "山田"]


def test_parse_lines_empty():
    assert parse_vocabulary_lines("") == []
    assert parse_vocabulary_lines("# コメントだけ\n\n") == []


def test_parse_lines_keeps_hash_inside_word():
    """'#' で始まる行はコメントだが、語の途中の '#' (C#/F#) は保持する。"""
    text = "C#\nF#\nNext.js"
    assert parse_vocabulary_lines(text) == ["C#", "F#", "Next.js"]


def test_parse_lines_leading_hash_is_comment():
    """先頭が '#' の語はコメント扱いで無視される (既知の制約)。"""
    assert parse_vocabulary_lines("#hashtag\nサンプル商会") == ["サンプル商会"]


def test_render_then_parse_roundtrip():
    """render → (ユーザー無編集) → parse で元の語リストに戻る。"""
    words = ["サンプル商会", "PayPal", "山田"]
    rendered = render_vocabulary_file(words)
    # ヘッダコメントが付く
    assert rendered.startswith("#")
    assert "サンプル商会" in rendered
    # コメントを無視して元のリストに戻る
    assert parse_vocabulary_lines(rendered) == words


def test_render_empty_words_has_only_header():
    rendered = render_vocabulary_file([])
    assert rendered.startswith("#")
    assert parse_vocabulary_lines(rendered) == []


def test_render_skips_blank_words():
    rendered = render_vocabulary_file(["サンプル商会", "", "  ", "PayPal"])
    assert parse_vocabulary_lines(rendered) == ["サンプル商会", "PayPal"]


# --- render_prompt_list (Phase G: 句読点プライミング prompt) ---


def test_prompt_list_empty_returns_empty():
    assert render_prompt_list([]) == ""
    assert render_prompt_list(["", "  "]) == ""


def test_prompt_list_single_term():
    assert render_prompt_list(["山田"]) == "山田。"


def test_prompt_list_joins_with_kuten():
    """「語、語、語。」形式 (読点区切り + 終止句点)。"""
    assert render_prompt_list(["サンプル商会", "山田", "Codex"]) == "サンプル商会、山田、Codex。"


def test_prompt_list_truncates_from_front_keeping_tail():
    """予算超過時は先頭 (重要度低) から落とし、末尾 (重要度高) を残す。

    Whisper 自身も超過時は先頭を切る + 条件付けは末尾ほど強いため。
    """
    # 各5文字 x 3語 + 区切り2 + 句点1 = 18 文字。max=12 なら末尾 2 語のみ
    out = render_prompt_list(
        ["あいうえお", "かきくけこ", "さしすせそ"], max_chars=12
    )
    assert out == "かきくけこ、さしすせそ。"
    assert len(out) <= 12


def test_prompt_list_never_cuts_mid_word():
    """語境界でのみ切る (部分語の混入禁止)。"""
    out = render_prompt_list(["アクアボイス", "タイプレス"], max_chars=8)
    # 「アクアボイス、タイプレス。」は 13 文字で入らない → 末尾の 1 語だけ
    assert out == "タイプレス。"


def test_prompt_list_dedupes_keeping_last():
    """重複は後の出現 (重要側) を残す。"""
    out = render_prompt_list(["山田", "サンプル商会", "山田"])
    assert out == "サンプル商会、山田。"


def test_prompt_list_default_budget_under_whisper_window():
    """デフォルト上限 110 文字 (最悪 ~220 token で 223 token 窓に収まる)。"""
    assert PROMPT_LIST_MAX_CHARS == 110
    many = [f"語彙{i:03d}" for i in range(100)]
    out = render_prompt_list(many)
    assert len(out) <= PROMPT_LIST_MAX_CHARS
