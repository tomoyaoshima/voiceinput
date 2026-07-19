"""screen_context の純関数テスト (AX 非依存)。

実際の AX 読み取り (capture) は macOS の権限とフォーカス状態に依存するので
ユニットテストでは扱わない。ここでは手組みの ScreenContext に対する語抽出・
LLM 文脈整形・本文読み取り可否の純ロジックだけを検証する。
"""

from voiceinput.screen_context import (
    DEFAULT_DENYLIST,
    ScreenContext,
    context_terms,
    resolve_mode,
    should_read_value,
)


# --- context_terms ---


def test_context_terms_priority_selected_then_focused_then_title():
    ctx = ScreenContext(
        window_title="案件タイトル",
        focused_text="本文に山田と記載",
        selected_text="プロジェクトX",
    )
    terms = context_terms(ctx)
    # 選択テキスト由来が先頭
    assert terms[0] == "プロジェクト"
    assert "山田" in terms
    assert "案件" in terms or "案件タイトル" in terms


def test_context_terms_dedupes():
    ctx = ScreenContext(focused_text="山田 山田 田中", selected_text="山田")
    terms = context_terms(ctx)
    assert terms.count("山田") == 1


def test_context_terms_empty_context_returns_empty():
    assert context_terms(ScreenContext()) == []


def test_context_terms_only_extracts_words_not_phrases():
    """文ではなく語 (カタカナ/漢字/英数字) のみ返す = Whisper 幻覚抑制。"""
    ctx = ScreenContext(focused_text="これはテストの文章です")
    terms = context_terms(ctx)
    # 助詞・ひらがなは入らず、漢字/カタカナ語のみ
    assert "テスト" in terms
    assert all(" " not in t for t in terms)


# --- should_read_value ---


def test_should_read_value_blocks_secure_field():
    assert should_read_value("AXSecureTextField", "com.example.app", ()) is False


def test_should_read_value_blocks_denylisted_bundle():
    assert (
        should_read_value("AXTextField", "com.apple.keychainaccess", DEFAULT_DENYLIST)
        is False
    )


def test_should_read_value_allows_normal_field():
    assert should_read_value("AXTextArea", "com.apple.TextEdit", DEFAULT_DENYLIST) is True


def test_should_read_value_empty_bundle_ok():
    assert should_read_value("AXTextField", "", DEFAULT_DENYLIST) is True


def test_denylist_includes_password_managers():
    assert "com.apple.keychainaccess" in DEFAULT_DENYLIST
    assert any("1password" in b or "onepassword" in b for b in DEFAULT_DENYLIST)


# --- resolve_mode (app_mode 判定) ---

_RULES = (
    ("bundle", "com.apple.mail", "mail"),
    ("title", "Gmail", "mail_en"),
    ("bundle", "com.tinyspeck.slackmacgap", "clean"),
)


def test_resolve_mode_bundle_exact_match():
    assert resolve_mode("Mail", "com.apple.mail", "受信", _RULES) == "mail"


def test_resolve_mode_bundle_no_partial_match():
    """bundle は完全一致のみ (部分一致しない)。"""
    assert resolve_mode("Mail", "com.apple.mailbox", "", _RULES) is None


def test_resolve_mode_title_partial_case_insensitive():
    """title は部分一致 + 大文字小文字無視。ブラウザ (Chrome) で Gmail を判定。"""
    title = "受信トレイ (3) - foo@example.com - gmail"
    assert resolve_mode("Google Chrome", "com.google.Chrome", title, _RULES) == "mail_en"


def test_resolve_mode_title_skipped_when_empty():
    """window_title が空 (AX 無効) のとき title ルールはスキップされる。"""
    # bundle=Chrome は bundle ルールに無いので、title 空なら None
    assert resolve_mode("Google Chrome", "com.google.Chrome", "", _RULES) is None


def test_resolve_mode_bundle_wins_over_later_title():
    """ルールは順番に評価され、先にヒットした bundle が優先される。"""
    rules = (
        ("bundle", "com.apple.mail", "mail"),
        ("title", "Mail", "mail_en"),
    )
    assert resolve_mode("Mail", "com.apple.mail", "Inbox - Mail", rules) == "mail"


def test_resolve_mode_no_match_returns_none():
    assert resolve_mode("Notes", "com.apple.Notes", "メモ", _RULES) is None


def test_resolve_mode_empty_rules():
    assert resolve_mode("Mail", "com.apple.mail", "Inbox", ()) is None


def test_resolve_mode_unknown_match_type_skipped():
    """match_type が bundle/title 以外の不正値はサイレントにスキップされる。"""
    rules = (
        ("regex", "com.apple.mail", "mail"),   # 未対応 match_type → 無視
        ("bundle", "com.apple.mail", "mail"),  # こちらでヒット
    )
    assert resolve_mode("Mail", "com.apple.mail", "", rules) == "mail"
    # 不正 match_type しか無ければ None
    assert resolve_mode("Mail", "com.apple.mail", "", (("regex", "x", "mail"),)) is None
