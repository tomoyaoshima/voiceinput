"""履歴の raw_text ↔ text 差分から確定置換 (replacements) の候補を掘る。

体感精度を下げる主犯は「毎回同じように誤変換される固有名詞」。これは
確率的な語彙ヒント (initial_prompt) より、text_filter.apply_replacements の
確定置換で直すのが最も精度が高い。このモジュールは履歴 200 件の
「STT 生出力 (raw_text)」と「LLM 整形後 (text)」の差分を比較し、
繰り返し出現する誤変換→修正ペアを候補として提示する。

LLM が直した箇所 = 「STT が誤り、LLM が (たまたま) 直せた」ペアなので、
これを確定置換に昇格させれば LLM の気まぐれに頼らず毎回直る。
採用判断は人間 (ユーザー) が行う前提で、候補提示までを担当する。
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from collections import Counter
from typing import Iterable

# 句読点・空白だけの差分は「整形」であって「誤変換の修正」ではないので除外
_PUNCT_RE = re.compile(r"[\s。、．，！？!?・…〜ー-]+")

# 候補ペアの最大長。長い差分は文の言い換え (整形) の可能性が高く、
# 確定置換に使うと誤爆するので除外する。
MAX_PAIR_LEN = 12

# 候補ペアの最小長。1 文字の置換 ("1"→"一", "p"→"P" など) は表記揺れ・
# 数字/アルファベットの正規化であることが多く、substring 置換に使うと
# 「他の語の中の 1 文字」まで巻き込んで誤爆するので除外する。
MIN_PAIR_LEN = 2


def _strip_punct(s: str) -> str:
    return _PUNCT_RE.sub("", s)


def extract_diff_pairs(raw: str, formatted: str) -> list[tuple[str, str]]:
    """1 件の履歴から (誤変換, 修正) 候補ペアを抽出する。

    difflib.SequenceMatcher の 'replace' オペコードを使い、raw と formatted の
    置換された部分文字列ペアを取り出す。以下は除外:
    - 句読点・空白を除くと同一 (= ただの整形)
    - どちらかが空 (挿入・削除は置換候補にならない)
    - MAX_PAIR_LEN 超 (言い換えの可能性が高い) / MIN_PAIR_LEN 未満 (誤爆リスク)

    重要: ペアは**元の (非正規化) 文字列から**抽出する。候補は最終的に
    `apply_replacements` (生テキストへの str.replace) に使われるため、
    NFKC 正規化した文字列を from にすると全角英数などの実テキストに
    一致せず、採用したのに一度も発火しない silent no-op になる。
    NFKC は「差が整形だけか」の比較にのみ使う。
    """
    if not raw or not formatted:
        return []
    if raw == formatted:
        return []
    pairs: list[tuple[str, str]] = []
    matcher = difflib.SequenceMatcher(a=raw, b=formatted, autojunk=False)
    for op, a0, a1, b0, b1 in matcher.get_opcodes():
        if op != "replace":
            continue
        src = raw[a0:a1].strip()
        dst = formatted[b0:b1].strip()
        if not src or not dst:
            continue
        if len(src) > MAX_PAIR_LEN or len(dst) > MAX_PAIR_LEN:
            continue
        # NFKC + 句読点除去で同一なら整形/表記正規化なので除外
        norm_src = _strip_punct(unicodedata.normalize("NFKC", src))
        norm_dst = _strip_punct(unicodedata.normalize("NFKC", dst))
        if norm_src == norm_dst:
            continue
        # 句読点を含むペアは境界がズレていることが多いので端を掃除
        src = src.strip("。、．，！？!? ")
        dst = dst.strip("。、．，！？!? ")
        if not src or not dst or src == dst:
            continue
        # 1 文字ペアは誤爆リスクが高いので除外 (MIN_PAIR_LEN)
        if len(src) < MIN_PAIR_LEN or len(dst) < MIN_PAIR_LEN:
            continue
        pairs.append((src, dst))
    return pairs


def mine_replacement_candidates(
    entries: Iterable[tuple[str, str]],
    *,
    min_count: int = 2,
) -> list[tuple[str, str, int]]:
    """(raw_text, text) ペアの列から置換候補を頻度順に返す。

    Returns
    -------
    list of (from, to, count) — count >= min_count のものだけ、頻度降順。
    同じ from に対して複数の to がある場合は最頻の to のみ採用する
    (揺れている修正は確定置換に向かない)。
    """
    counter: Counter[tuple[str, str]] = Counter()
    for raw, formatted in entries:
        for pair in extract_diff_pairs(raw, formatted):
            counter[pair] += 1

    # from ごとに最頻の to を選ぶ
    best_by_src: dict[str, tuple[str, int]] = {}
    for (src, dst), cnt in counter.items():
        cur = best_by_src.get(src)
        if cur is None or cnt > cur[1]:
            best_by_src[src] = (dst, cnt)

    out = [
        (src, dst, cnt)
        for src, (dst, cnt) in best_by_src.items()
        if cnt >= min_count
    ]
    out.sort(key=lambda x: (-x[2], x[0]))
    return out
