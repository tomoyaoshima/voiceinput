from voiceinput.text_filter import (
    apply_replacements,
    redact_secrets,
    strip_llm_meta_commentary,
    strip_whisper_hallucinations,
)


def test_empty_passthrough():
    assert strip_whisper_hallucinations("") == ""


def test_whole_text_hallucination_becomes_empty():
    assert strip_whisper_hallucinations("ご視聴ありがとうございました。") == ""
    assert strip_whisper_hallucinations("ご清聴ありがとうございました") == ""


def test_trailing_thank_you_is_stripped():
    text = "明日の打ち合わせ、よろしくお願いします。ご視聴ありがとうございました。"
    assert (
        strip_whisper_hallucinations(text)
        == "明日の打ち合わせ、よろしくお願いします。"
    )


def test_trailing_subscribe_is_stripped():
    text = "それではまた今度。チャンネル登録もよろしくお願いします。"
    assert strip_whisper_hallucinations(text) == "それではまた今度。"


def test_repeated_trailing_hallucinations_are_all_stripped():
    text = "今日は早めに失礼します。ご視聴ありがとうございました。チャンネル登録お願いします。"
    assert strip_whisper_hallucinations(text) == "今日は早めに失礼します。"


def test_english_thanks_for_watching_is_stripped():
    text = "Today I recorded the demo. Thanks for watching."
    assert strip_whisper_hallucinations(text) == "Today I recorded the demo."


def test_real_speech_not_overcleaned():
    text = "明日の会議は10時からです。"
    assert strip_whisper_hallucinations(text) == "明日の会議は10時からです。"


def test_real_speech_without_trailing_punctuation_kept_as_is():
    text = "今日のミーティング、参加できなかったよ"
    assert (
        strip_whisper_hallucinations(text) == "今日のミーティング、参加できなかったよ"
    )


def test_trailing_next_episode_is_stripped():
    text = "本日はここまでです。次回もお楽しみに。"
    assert strip_whisper_hallucinations(text) == "本日はここまでです。"


def test_byebye_is_stripped():
    text = "じゃあね。バイバイ〜"
    assert strip_whisper_hallucinations(text) == "じゃあね。"


def test_leading_subtitle_credit_is_stripped():
    text = "字幕 by ABC\n本編開始です"
    out = strip_whisper_hallucinations(text)
    assert out == "本編開始です"


# ---------------------------------------------------------------------------
# strip_llm_meta_commentary
# ---------------------------------------------------------------------------


def test_meta_passthrough_when_no_parens():
    assert strip_llm_meta_commentary("") == ""
    assert (
        strip_llm_meta_commentary("普通の文章です。")
        == "普通の文章です。"
    )


def test_meta_legitimate_paren_kept():
    """meta keyword を含まない括弧 (普通の補足) は残す。"""
    text = "10時(月曜)に集合してください。"
    assert strip_llm_meta_commentary(text) == text
    text2 = "（参考画像を添付）"
    assert strip_llm_meta_commentary(text2) == text2


def test_meta_paren_with_keyword_is_stripped():
    text = "今日は晴れです。（ただし、午後から雨です）"
    out = strip_llm_meta_commentary(text)
    assert out == "今日は晴れです。"


def test_meta_three_paragraph_pattern_takes_last():
    """qwen2.5:14b の悪い癖: "誤 → メタ説明 → 正" の 3 段構造。

    実際のユーザー報告に基づくケース。最後の正しい段落を採用する。
    """
    text = (
        "次の問い合わせをしたいので、問い合わせ送信手間までお願いします。\n"
        "\n"
        "（ただし、この文は原文から「手前」が削除され、「手間」に変更されていますが、"
        "指示により同義語置換や意味の変更は禁止されているため、本来は"
        "「問い合わせ送信手前までお願いします」と修正すべきです。"
        "従って、以下のように修正します：）\n"
        "\n"
        "問い合わせ送信手前までお願いします。"
    )
    out = strip_llm_meta_commentary(text)
    assert out == "問い合わせ送信手前までお願いします。"


def test_meta_inline_paren_with_keyword_is_stripped_keeping_rest():
    text = (
        "明日は会議です。"
        "（ただし、参加者が変更される可能性があります）"
        "資料を準備してください。"
    )
    out = strip_llm_meta_commentary(text)
    assert "ただし" not in out
    assert "明日は会議です。" in out
    assert "資料を準備してください。" in out


def test_meta_alternative_marker_paren_is_stripped():
    text = (
        "ありがとうございました。"
        "（もしくは「ありがとうございます」が自然です）"
    )
    out = strip_llm_meta_commentary(text)
    assert out == "ありがとうございました。"


def test_meta_half_width_paren_also_handled():
    text = "OK です。(ただし、注意点あり)"
    out = strip_llm_meta_commentary(text)
    assert out == "OK です。"


# ---------------------------------------------------------------------------
# strip_llm_meta_commentary: collapse_paragraphs=False (mail / mail_en 用)
# ---------------------------------------------------------------------------


def test_meta_keep_multi_paragraph_when_not_collapsing():
    """collapse_paragraphs=False は複数段落本文を保持する (mail/mail_en)。

    既定 (True) なら最後の段落だけになるが、メール本文は段落構造を残す。
    """
    text = "Hi John,\n\nYour order has shipped.\n\nBest regards,"
    out = strip_llm_meta_commentary(text, collapse_paragraphs=False)
    assert out == "Hi John,\n\nYour order has shipped.\n\nBest regards,"


def test_meta_keep_multi_paragraph_still_strips_meta_paren():
    """段落保持モードでも、メタ括弧で空になった段落は落とす。"""
    text = (
        "Hi John,\n"
        "\n"
        "（ただし、本来は丁寧に修正すべきですが、以下のように返します：）\n"
        "\n"
        "Your order has shipped."
    )
    out = strip_llm_meta_commentary(text, collapse_paragraphs=False)
    assert out == "Hi John,\n\nYour order has shipped."
    assert "ただし" not in out


def test_meta_collapse_default_unchanged_for_clean():
    """既定 (collapse_paragraphs=True) は従来どおり最後の段落を採用 (回帰防止)。"""
    text = "誤った文。\n\n（ただし、修正します：）\n\n正しい文。"
    assert strip_llm_meta_commentary(text) == "正しい文。"


def test_meta_keep_single_paragraph_unchanged_when_not_collapsing():
    text = "今日は晴れです。"
    assert (
        strip_llm_meta_commentary(text, collapse_paragraphs=False)
        == "今日は晴れです。"
    )


def test_meta_all_paragraphs_empty_when_not_collapsing():
    """段落保持モードでメタ括弧だけの出力 → 全段落が消えても空文字で返る。

    collapse=True 側 (cleaned.strip()) と対称になることを確認 (どちらも "")。
    """
    text = "（ただし、修正します：）"
    assert strip_llm_meta_commentary(text, collapse_paragraphs=False) == ""
    assert strip_llm_meta_commentary(text, collapse_paragraphs=True) == ""


# ---------------------------------------------------------------------------
# redact_secrets (Phase F: 画面コンテキストの機密マスク)
# ---------------------------------------------------------------------------


def test_redact_empty_passthrough():
    assert redact_secrets("") == ""


def test_redact_keeps_normal_japanese_proper_nouns():
    """通常の固有名詞・文は壊さない (誤爆しない)。"""
    text = "株式会社サンプルの田中さんに見積もりを送る"
    assert redact_secrets(text) == text


def test_redact_email():
    out = redact_secrets("連絡先は taro@example.co.jp です")
    assert "taro@example.co.jp" not in out
    assert "███" in out
    assert "連絡先は" in out and "です" in out


def test_redact_password_label():
    out = redact_secrets("password: hunter2xyz をメモ")
    assert "hunter2xyz" not in out
    assert "███" in out


def test_redact_api_key_label():
    out = redact_secrets("API_KEY=sk-abcDEF123456")
    assert "sk-abcDEF123456" not in out
    assert "███" in out


def test_redact_label_with_space_variants():
    """ラベルがスペース区切りでもマスクされる (api key / access key / client secret)。"""
    for label in ("api key", "access key", "client secret"):
        out = redact_secrets(f"{label}: shortval123")
        assert "shortval123" not in out, label
        assert "███" in out, label


def test_redact_long_token():
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9abcdefg"
    out = redact_secrets(f"トークンは {token} だよ")
    assert token not in out
    assert "███" in out
    assert "トークンは" in out


def test_redact_credit_card_like():
    out = redact_secrets("カード番号 4111 1111 1111 1111 を入力")
    assert "4111 1111 1111 1111" not in out
    assert "███" in out


def test_redact_short_alnum_not_touched():
    """短い英数字 (商品コード等) は潰さない。"""
    text = "型番は ABC123 です"
    assert redact_secrets(text) == text


# ---------------------------------------------------------------------------
# apply_replacements (決定的テキスト置換)
# ---------------------------------------------------------------------------


def test_replacements_empty_rules_passthrough():
    assert apply_replacements("ペイパルで支払う", ()) == "ペイパルで支払う"


def test_replacements_empty_text_passthrough():
    assert apply_replacements("", (("ペイパル", "PayPal"),)) == ""


def test_replacements_basic_substitution():
    out = apply_replacements("ペイパルで支払う", (("ペイパル", "PayPal"),))
    assert out == "PayPalで支払う"


def test_replacements_multiple_occurrences():
    out = apply_replacements("さんぷるとさんぷる", (("さんぷる", "サンプル商会"),))
    assert out == "サンプル商会とサンプル商会"


def test_replacements_applied_in_order():
    """ルールは順に適用され、前段の結果に次段がかかる。"""
    rules = (("A", "B"), ("B", "C"))
    assert apply_replacements("A", rules) == "C"


def test_replacements_case_sensitive():
    """大文字小文字は区別する。"""
    out = apply_replacements("paypal と PayPal", (("paypal", "PayPal"),))
    assert out == "PayPal と PayPal"


def test_replacements_empty_from_is_ignored():
    """from が空のルールは無視する (全位置マッチの暴発防止)。"""
    out = apply_replacements("そのまま", (("", "X"),))
    assert out == "そのまま"


def test_replacements_empty_to_deletes_match():
    """to が空文字 (config で to 省略時) のルールは一致部分を削除する。"""
    out = apply_replacements("あー、えーと、本題です", (("あー、", ""), ("えーと、", "")))
    assert out == "本題です"
