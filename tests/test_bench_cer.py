"""metrics (CER) の純関数テスト。scripts/bench_stt.py が使用する。"""

from voiceinput.metrics import cer, levenshtein, normalize_for_cer


def test_normalize_strips_punct_and_space():
    assert normalize_for_cer("こんにちは、 世界。") == "こんにちは世界"


def test_normalize_nfkc():
    assert normalize_for_cer("ＡＢＣ１２３") == "ABC123"


def test_normalize_removes_long_vowel_mark():
    """長音の有無 (サーバ/サーバー) は表記揺れであって誤りではない。"""
    assert normalize_for_cer("サーバー") == normalize_for_cer("サーバ")


def test_levenshtein_basics():
    assert levenshtein("", "") == 0
    assert levenshtein("abc", "") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("abc", "axc") == 1
    assert levenshtein("kitten", "sitting") == 3


def test_cer_exact_match_is_zero():
    assert cer("明日の会議は10時から。", "明日の会議は、10時から") == 0.0


def test_cer_total_mismatch_is_high():
    assert cer("あいう", "xyz") == 1.0


def test_cer_partial():
    # 5 文字中 1 置換 → 0.2
    assert cer("あいうえお", "あいうえこ") == 0.2


def test_cer_empty_reference():
    assert cer("", "") == 0.0
    assert cer("", "なにか") == 1.0
