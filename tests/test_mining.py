"""mining (履歴 diff → 置換候補) の純関数テスト。"""

from voiceinput.mining import extract_diff_pairs, mine_replacement_candidates


def test_extract_identical_returns_empty():
    assert extract_diff_pairs("こんにちは", "こんにちは") == []


def test_extract_punctuation_only_diff_is_ignored():
    """句読点付与だけの差分 (= 整形) は候補にしない。"""
    assert (
        extract_diff_pairs(
            "明日の会議は10時からですよね", "明日の会議は10時からですよね。"
        )
        == []
    )


def test_extract_word_replacement():
    pairs = extract_diff_pairs(
        "こーでっくすの件で連絡します", "Codexの件で連絡します"
    )
    assert ("こーでっくす", "Codex") in pairs


def test_extract_long_rewrite_pairs_are_bounded():
    """言い換え (整形) から出るペアも MAX_PAIR_LEN 以下に制限される。

    SequenceMatcher は共通文字をアンカーに短い replace 断片を返すことが
    あるが、全体書き換え級の長い断片は候補にならない。
    """
    from voiceinput.mining import MAX_PAIR_LEN

    pairs = extract_diff_pairs(
        "これはとても長い文章の言い換えテストでございます",
        "全然違う内容にまるごと書き換えられた文章になっています",
    )
    for src, dst in pairs:
        assert len(src) <= MAX_PAIR_LEN
        assert len(dst) <= MAX_PAIR_LEN


def test_extract_single_char_pairs_are_ignored():
    """1 文字ペア ("1"→"一" 等) は substring 誤爆リスクが高いので除外。"""
    pairs = extract_diff_pairs("会議は1時から", "会議は一時から")
    assert pairs == []


def test_extract_empty_inputs():
    assert extract_diff_pairs("", "x") == []
    assert extract_diff_pairs("x", "") == []


def test_mine_counts_and_filters_by_min_count():
    entries = [
        ("こーでっくすを使う", "Codexを使う"),
        ("こーでっくすで確認", "Codexで確認"),
        ("やまださんへ", "山田さんへ"),  # 1 回だけ
    ]
    out = mine_replacement_candidates(entries, min_count=2)
    assert out == [("こーでっくす", "Codex", 2)]


def test_mine_picks_most_frequent_target():
    """同じ from に修正揺れがある場合は最頻の to を採用する。"""
    entries = [
        ("さんぷるの商品", "サンプル商会の商品"),
        ("さんぷるの発送", "サンプル商会の発送"),
        ("さんぷるの件", "サンプル商回の件"),  # 揺れ (少数派)
    ]
    out = mine_replacement_candidates(entries, min_count=2)
    assert out == [("さんぷる", "サンプル商会", 2)]


def test_mine_empty_entries():
    assert mine_replacement_candidates([]) == []


def test_extract_preserves_original_width():
    """to は元の (非正規化) 表記を保つ。

    apply_replacements は生テキストへの str.replace なので、NFKC 正規化した
    表記を返すと全角の実テキストにマッチせず silent no-op になる。
    元表記のまま返すことを保証する。
    """
    pairs = extract_diff_pairs("ぺいぱるで支払い", "ＰａｙＰａｌで支払い")
    assert ("ぺいぱる", "ＰａｙＰａｌ") in pairs  # 全角のまま


def test_extract_width_normalization_only_is_ignored():
    """全角→半角の表記正規化だけの差分は候補にしない (NFKC 比較で同一)。"""
    assert extract_diff_pairs("ＡＰＩの設定", "APIの設定") == []
