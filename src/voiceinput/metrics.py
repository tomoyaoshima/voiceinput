"""STT ベンチマーク用の CER (文字誤り率) 計算。

依存追加なし (stdlib のみ)。scripts/bench_stt.py と tests から使う。
"""

from __future__ import annotations

import re
import unicodedata

# 句読点・空白・記号は CER の対象外 (整形の差を実力差と混同しないため)
_STRIP_RE = re.compile(r"[\s。、．，！？!?・…〜「」『』()（）\-ー]+")


def normalize_for_cer(s: str) -> str:
    """CER 計算用の正規化: NFKC + 句読点/空白/記号除去。

    「ー」(長音) も除去する。長音の有無 (サーバ/サーバー) は表記揺れで
    あって認識誤りではないため。
    """
    s = unicodedata.normalize("NFKC", s)
    return _STRIP_RE.sub("", s)


def levenshtein(a: str, b: str) -> int:
    """編集距離 (挿入・削除・置換 各コスト 1)。O(len(a)*len(b)) の DP。"""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    """正規化後の文字誤り率。reference が空なら hypothesis の有無で 0/1。"""
    ref = normalize_for_cer(reference)
    hyp = normalize_for_cer(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)
