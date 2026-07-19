"""カスタム語彙ヒントを Whisper の ``initial_prompt`` に渡すためのモジュール。

履歴 (整形後テキスト) からカタカナ・漢字・英数字識別子を抽出して頻度順に
``top_n`` を取り、``config.yaml`` の ``vocabulary.manual`` と合成して 1 行の
スペース区切り文字列を返す。Whisper の initial_prompt は概ね 224 トークンが
上限なので、最終的な文字数を ``max_chars`` (デフォルト 200) で打ち切る。

設計上の注意点:

- ``raw_text`` ではなく ``text`` (LLM 整形後) を抽出ソースにする。
  raw に誤認識が混じると、それを再投入してしまう負のフィードバック
  ループになるため。
- 形態素解析ライブラリ (mecab/janome) は導入しない。voiceinput は
  uv 管理下の軽量プロトタイプなので、追加の C 拡張依存を避ける。
  正規表現で 90% は十分役に立つ。
- ``VocabularyBuilder`` は履歴 mtime + 設定パラメータをハッシュとして
  in-memory にキャッシュする。同じ入力なら再構築しない。
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from voiceinput.history import History

# カタカナ連続 (長音含む)。半濁音含めて [ァ-ヺー] に収める。
_KATAKANA_RE = re.compile(r"[ァ-ヺー]{2,}")
# 漢字連続。CJK 統合漢字の主要範囲。
_KANJI_RE = re.compile(r"[一-鿿]{2,}")
# 英数字混じり識別子。先頭が英字、3 文字以上 (2 文字だと "is" や "PR" が暴走しがち)。
_ALNUM_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")

# Whisper initial_prompt は ~224 トークン。日本語 1 文字 ≈ 1-2 トークンなので
# 文字数 200 で打ち切れば概ね安全。
DEFAULT_MAX_CHARS = 200


def extract_terms(text: str) -> list[str]:
    """1 つのテキストから語彙候補を抽出する。

    返り値は重複ありの出現順。``Counter`` に直接食わせる前提。
    """
    if not text:
        return []
    out: list[str] = []
    for regex in (_KATAKANA_RE, _KANJI_RE, _ALNUM_RE):
        out.extend(regex.findall(text))
    return out


def _join_under_limit(terms: Iterable[str], max_chars: int) -> str:
    """スペース区切りで連結しつつ、合計文字数が max_chars を超えないように切る。"""
    parts: list[str] = []
    total = 0
    for t in terms:
        added = len(t) + (1 if parts else 0)
        if total + added > max_chars:
            break
        parts.append(t)
        total += added
    return " ".join(parts)


# render_prompt_list の既定文字数上限。Whisper の initial_prompt は末尾
# 223 token 保持で、日本語は 1 文字 1-2 token。110 文字なら最悪でも
# ~220 token に収まり、Whisper 側での先頭切り捨てを実質回避できる。
# (mlx_whisper の tokenizer で実 token 数を数える厳密化は将来課題)
PROMPT_LIST_MAX_CHARS = 110


def render_prompt_list(terms: Iterable[str], *, max_chars: int = PROMPT_LIST_MAX_CHARS) -> str:
    """語リストを「{語}、{語}、{語}。」形式の句読点プライミング prompt にする。

    Whisper の initial_prompt は「直前の発話の続き」として条件付けされる。
    読点区切り + 終止句点の形式にすることで、(a) 句読点付き出力が安定する
    (b) メタ説明文と違って文体バイアス (です・ます化) を持ち込まない。

    重要な性質:
    - **入力 terms は「重要度が低い順」を前提とし、切り詰めは先頭から落とす**
      (末尾保持)。Whisper 自身も超過時は先頭を切る + 末尾ほど条件付けが
      強いため、重要語 (画面語など) は末尾に置く設計と整合させる。
    - 切り詰めは語境界のみ (部分語の混入禁止)。
    - 重複は「後の出現」を残す (重要側を保持)。
    - terms が空なら "" を返す。
    """
    # 重複排除: 後の出現を残す
    seen: set[str] = set()
    cleaned_rev: list[str] = []
    for t in reversed([t.strip() for t in terms if t and t.strip()]):
        if t in seen:
            continue
        seen.add(t)
        cleaned_rev.append(t)
    # cleaned_rev は「重要度が高い順」(元リストの末尾から)。
    # 末尾優先で予算に収まるだけ拾う。骨格「、」区切り + 終止「。」ぶんを勘定。
    picked_rev: list[str] = []
    total = 1  # 終止の「。」
    for t in cleaned_rev:
        added = len(t) + (1 if picked_rev else 0)  # 区切りの「、」
        if total + added > max_chars:
            break
        picked_rev.append(t)
        total += added
    if not picked_rev:
        return ""
    return "、".join(reversed(picked_rev)) + "。"


def compose_initial_prompt(
    static_prompt: str,
    screen_terms: Iterable[str],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """静的語彙 (履歴+手動) と画面コンテキスト語を 1 本の initial_prompt に合成する。

    Phase F: 録音開始時に画面から抽出した語 (``screen_terms``) を **先頭**に
    置く。「今まさに目の前にある語」を最優先で Whisper にヒントするため。
    続けて既存の静的 prompt (`build_initial_prompt` の出力, スペース区切り)
    の語を重複排除しつつ追加し、``max_chars`` で切り詰める。

    - ``screen_terms`` は ``extract_terms`` 由来の「語」のみを想定 (文を入れない)。
    - 重複は最初に出た順を維持。
    - ``screen_terms`` が空なら static_prompt をそのまま (max_chars 切詰のみ) 返す。
    """
    seen: set[str] = set()
    combined: list[str] = []

    for term in screen_terms:
        if not term:
            continue
        term = term.strip()
        if not term or term in seen:
            continue
        seen.add(term)
        combined.append(term)

    # static_prompt はスペース区切りの語列。分解して重複排除しつつ後ろに足す。
    for term in static_prompt.split():
        if term in seen:
            continue
        seen.add(term)
        combined.append(term)

    return _join_under_limit(combined, max_chars)


def build_initial_prompt(
    history: History,
    manual: list[str],
    *,
    history_size: int = 100,
    top_n: int = 30,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """履歴 + 手動リストから initial_prompt 文字列を組み立てる。

    手動リストが先 (優先)、その後に履歴頻出語 ``top_n`` を追加。重複は除外。
    最終的な合計文字数が ``max_chars`` を超えないように切り詰める。

    入力が完全に空 (履歴 0 件 & manual 空) の場合は空文字を返す。
    """
    seen: set[str] = set()
    combined: list[str] = []

    # 手動リスト優先 (重複は最初に出た順を維持)
    for term in manual:
        if not term:
            continue
        term = term.strip()
        if not term or term in seen:
            continue
        seen.add(term)
        combined.append(term)

    # 履歴から頻度順に追加
    counter: Counter[str] = Counter()
    if history_size > 0:
        for entry in history.list(limit=history_size):
            for term in extract_terms(entry.text or ""):
                counter[term] += 1
    for term, _count in counter.most_common(top_n):
        if term in seen:
            continue
        seen.add(term)
        combined.append(term)

    return _join_under_limit(combined, max_chars)


def parse_vocabulary_lines(text: str) -> list[str]:
    """テキストエリア (1 行 1 語) の内容を語彙リストに変換する純関数。

    GUI の「まとめて編集」で TextEdit に書いた内容をパースする。

    - 各行を strip。
    - 空行と ``#`` で始まる行 (コメント/使い方) は無視。
    - 重複は最初の出現順を保って排除。
    """
    out: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def render_vocabulary_file(words: Iterable[str]) -> str:
    """「まとめて編集」用テキストを生成する (ヘッダコメント + 1 行 1 語)。

    ``parse_vocabulary_lines`` と対。書き出し → ユーザー編集 → 読み戻しで
    往復してもコメント行は無視されるので、語彙だけが安定して残る。空白語は捨てる。
    """
    header = (
        "# voiceinput カスタム語彙 — 1 行に 1 語を書いてください。\n"
        "# '#' で始まる行と空行は無視されます。並べ替え・追加・削除は自由です。\n"
        "# 編集して保存 (Cmd+S) したら、確認ダイアログで「反映」を押してください。\n"
        "\n"
    )
    body = "\n".join(w.strip() for w in words if w and w.strip())
    return header + body + ("\n" if body else "")


class VocabularyBuilder:
    """履歴 + 手動リストから initial_prompt を組み立てるキャッシュ付きビルダー。

    ``build()`` は履歴ファイルの mtime と各設定値をキーにメモ化する。
    同じキーなら再構築せず以前の値を返す。``build(force=True)`` で
    キャッシュを無視して必ず再構築できる (menu の "Refresh now" 用)。
    """

    def __init__(
        self,
        history: History,
        manual: list[str],
        *,
        history_size: int = 100,
        top_n: int = 30,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> None:
        self.history = history
        self.manual = list(manual)
        self.history_size = history_size
        self.top_n = top_n
        self.max_chars = max_chars
        self._cache_key: tuple | None = None
        self._cache_value: str = ""

    def _history_mtime(self) -> float:
        try:
            return self.history.path.stat().st_mtime
        except OSError:
            return 0.0

    def _current_key(self) -> tuple:
        return (
            self._history_mtime(),
            tuple(self.manual),
            self.history_size,
            self.top_n,
            self.max_chars,
        )

    def build(self, force: bool = False) -> str:
        key = self._current_key()
        if not force and key == self._cache_key:
            return self._cache_value
        value = build_initial_prompt(
            self.history,
            self.manual,
            history_size=self.history_size,
            top_n=self.top_n,
            max_chars=self.max_chars,
        )
        self._cache_key = key
        self._cache_value = value
        return value
