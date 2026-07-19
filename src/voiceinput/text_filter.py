"""Whisper の典型的な幻覚出力 (hallucination) を後処理で除去する。

日本語の Whisper は無音・雑音区間で YouTube 動画の締め台詞を勝手に生成しがち
("ご視聴ありがとうございました" / "チャンネル登録お願いします" / 字幕クレジット
など)。学習データに動画キャプションが多く含まれているため。

このモジュールは、文末・文頭に出やすい既知パターンを正規表現で除去する。
判断材料は確率や音響情報ではなく "テキストパターン" だけなので、本物の発話と
偶然一致した場合も削除されてしまう点には注意 (例: 本当に「ご視聴ありがとう
ございました」と言いたい場合)。
"""

import re

# 末尾に幻覚として出やすいパターン (繰り返し連結することもあるのでループ削除する)
_TRAILING_PATTERNS = [
    # 「ご視聴/ご清聴 ありがとうございました」系
    r"(?:最後まで|本日は|今日は)?(?:ご(?:清|視)聴|ご視聴)(?:いただき|くださり|頂き)?(?:まして)?(?:、)?(?:本当に)?ありがとう(?:ございました|ございます|でした)?[。!?\s]*$",
    # 「チャンネル登録 / 高評価」系
    r"(?:[次新]の動画(?:でも)?お会いしましょう|また[次会]の動画で(?:お会いしましょう)?)[。!?\s]*$",
    r"(?:チャンネル登録|高評価)(?:と[^。\n]{0,30})?(?:も)?(?:[、,])?(?:よろしく)?(?:お?願い)?(?:します|いたします)?[。!?\s]*$",
    # 「次回もお楽しみに」系
    r"(?:それでは)?(?:また)?次回(?:も|の動画(?:でも)?)?(?:[、,])?お楽しみ(?:に|ください)[。!?\s]*$",
    # 英語キャプション
    r"(?:Thanks?\s*(?:you)?\s*for\s*watching)[\s。.!?]*$",
    r"(?:Please\s+)?(?:subscribe|like\s+and\s+subscribe)[\s。.!?]*$",
    # 単独の「バイバイ」
    r"バイバイ[ー〜]?[、。!?\s]*$",
]

# 文頭の字幕クレジット系 (1 回だけ削除)
_LEADING_PATTERNS = [
    r"^\s*(?:字幕|字幕制作|字幕翻訳|字幕作成|提供|協力)[\s:by協力者ー\-:]{0,3}[^\n。]{0,40}\s*[\n。]",
]

_TRAILING_RES = [re.compile(p) for p in _TRAILING_PATTERNS]
_LEADING_RES = [re.compile(p) for p in _LEADING_PATTERNS]

# 完全一致したら丸ごと空文字に倒すパターン (テキスト全体がこれだけの場合)
_WHOLE_TEXT_HALLUCINATIONS = {
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございました。",
    "ご清聴ありがとうございました",
    "ご清聴ありがとうございました。",
    "ありがとうございました",
    "ありがとうございました。",
}


def strip_whisper_hallucinations(text: str) -> str:
    if not text:
        return text
    cleaned = text.strip()
    if cleaned in _WHOLE_TEXT_HALLUCINATIONS:
        return ""
    # 末尾を繰り返し削る (連続して出る場合がある)。
    # rstrip は空白だけにして、元の発話末尾の句読点は残す。
    while True:
        new = cleaned
        for pat in _TRAILING_RES:
            new = pat.sub("", new).rstrip()
        if new == cleaned:
            break
        cleaned = new
    # 先頭の字幕クレジット
    for pat in _LEADING_RES:
        cleaned = pat.sub("", cleaned)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# LLM (qwen2.5:14b など) の meta-commentary を除去する
# ---------------------------------------------------------------------------

# clean / mail プロンプトで「修正理由を書くな」と禁じても LLM は時々
# 括弧に判断過程を漏らす。出力側でも防衛するための後処理。

# meta-discourse marker. これを「含む括弧ブロック」だけを strip する。
# 普通の補足括弧 (例: "10時(月曜)") は誤爆させない。
_LLM_META_KEYWORDS: tuple[str, ...] = (
    "ただし",
    "従って",
    "本来",
    "禁止",
    "指示により",
    "修正します",
    "修正すべき",
    "以下のように",
    "代替案",
    "もしくは",
    "選択肢",
    "注：",
    "注:",
    "補足：",
    "補足:",
    "備考",
    "解説",
    "理由は",
)

# 1 段階の括弧ブロック (全角・半角どちらも)。ネストは想定しない。
_PAREN_BLOCK_RE = re.compile(r"[(（][^()（）]*[)）]")

# 多めの空白行 (LLM が「修正前\n\n（meta）\n\n修正後」を出すパターン)
_BLANK_LINE_RE = re.compile(r"\n[ \t　]*\n+")


def strip_llm_meta_commentary(
    text: str, *, collapse_paragraphs: bool = True
) -> str:
    """LLM が出力に混ぜたメタ説明 (括弧の理由付け / 二択 / 注釈) を除去する。

    対策する典型パターン:
        修正前テキスト

        （ただし、…の理由で本来は…と修正すべきです。以下のように修正します：）

        修正後テキスト

    後処理:
    1. メタ keyword を含む括弧ブロックを strip。
    2. (``collapse_paragraphs=True`` のみ) 段落分割後に 2 つ以上残った場合、
       LLM の自然な構造 (誤→説明→正) にならって最後の段落を採用する。
    3. clean / mail モード両方で安全に呼べる純関数。

    raw 用途や本物の括弧 (例: "(月曜)") は keyword を含まないため温存される。

    ``collapse_paragraphs``:
        ``True`` (既定): clean / raw 系。LLM が "誤→メタ説明→正" の 3 段で
        返す癖に対応し、最後の段落だけを採用する。短い 1 ターン整形を想定。

        ``False``: mail / mail_en など **本文が複数段落になりうる** モード用。
        段落を最後の 1 つに畳むと本文が消えてしまう (例: 英文ビジネスメールの
        挨拶→本文→結びが結びだけになる)。メタ括弧ブロックは除去するが、空に
        なった段落だけを落として残りの段落構造は保持する。
    """
    if not text:
        return text

    def _maybe_strip(m: re.Match) -> str:
        body = m.group(0)
        if any(kw in body for kw in _LLM_META_KEYWORDS):
            return ""
        return body

    cleaned = _PAREN_BLOCK_RE.sub(_maybe_strip, text)

    # 段落単位に分割し、空段落 (メタ括弧除去で中身が消えたもの) を落とす。
    paragraphs = [
        p.strip() for p in _BLANK_LINE_RE.split(cleaned) if p.strip()
    ]

    if not collapse_paragraphs:
        # 複数段落を保持するモード。メタ括弧で空になった段落だけを除き、
        # 本文の段落構造はそのまま再結合して返す。全段落が消えた場合は
        # collapse=True 側と同じく cleaned.strip() に倒す (空入力の対称性)。
        return "\n\n".join(paragraphs) if paragraphs else cleaned.strip()

    if len(paragraphs) > 1:
        # 余分な段落分割は LLM が "代替案" を並べた兆候なので、最後を採用
        return paragraphs[-1]
    if paragraphs:
        return paragraphs[0]
    return cleaned.strip()


# ---------------------------------------------------------------------------
# 画面コンテキスト (Phase F) の機密リダクション
# ---------------------------------------------------------------------------
#
# Accessibility 経由で読んだ入力欄テキストには、パスワード・API キー・
# トークン・メールアドレス・クレカ番号などが混じりうる。SecureTextField
# スキップや denylist だけでは普通の AXTextArea (メール下書き等) に入った
# 機密を防げないので、Whisper/LLM に渡す前にこの純関数でマスクする。
#
# 方針: 「いかにも機密」なパターンだけを ███ に潰す。日本語の固有名詞や
# 通常の文は壊さない (誤爆を避ける)。完璧な DLP ではなく "明白な漏洩を
# 1 枚噛ませる" 安全網。

_REDACT = "███"

# メールアドレス
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# クレジットカード風 (13-16 桁、ハイフン/スペース区切り許容)
_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,16}\b")
# password / secret / token / api key ラベル + 直後の値。
# 複合語は区切り文字 (スペース / アンダースコア / ハイフン) を許容する
# ("api key" "access-key" "client_secret" 等)。
_SECRET_LABEL_RE = re.compile(
    r"(?i)\b(?:password|passwd|pwd|secret|token|api[ _-]?key|apikey|"
    r"access[ _-]?key|client[ _-]?secret|bearer|認証コード|パスワード)\b"
    r"\s*[:=]?\s*\S+"
)
# 長い英数字+記号トークン (JWT / hex / base64 / sk-… 系)。30 文字以上で
# 大文字小文字数字記号が混在するものに限定し、普通の英単語を巻き込まない。
_LONG_TOKEN_RE = re.compile(r"\b[A-Za-z0-9_\-./+=]{30,}\b")


def redact_secrets(text: str) -> str:
    """機密らしき部分文字列を ███ に置換した文字列を返す。

    Whisper の initial_prompt / LLM の context ヒントに画面テキストを渡す
    前段で噛ませる安全網。順序が重要 (ラベル付き秘密を先に潰してから
    汎用トークン/カード/メール)。空入力はそのまま返す。
    """
    if not text:
        return text
    out = _SECRET_LABEL_RE.sub(_REDACT, text)
    out = _LONG_TOKEN_RE.sub(_REDACT, out)
    out = _CARD_RE.sub(_REDACT, out)
    out = _EMAIL_RE.sub(_REDACT, out)
    return out


# ---------------------------------------------------------------------------
# 決定的テキスト置換 (replacements)
# ---------------------------------------------------------------------------
#
# STT / LLM が誤変換しやすい固有名詞・商品名・ブランド名・取引先名を、確定的な
# 単純文字列の find -> replace で最後に直す。カスタム語彙 (Whisper initial_prompt
# への注入) が「認識しやすくするヒント」なのに対し、これは「最終出力を確定で
# 固定する」後段の安全網。LLM の気まぐれに左右されない。
#
# 仕様: 正規表現ではなく単純な substring 置換。大文字小文字は区別する。ルールは
# 与えられた順に適用する (前のルールの結果に次のルールがかかる)。空文字 from は
# 無視する (全位置マッチの暴発を防ぐ)。空入力・空ルールは no-op。


def apply_replacements(
    text: str, rules: tuple[tuple[str, str], ...]
) -> str:
    """``rules`` の (from, to) を順に単純置換した文字列を返す。

    - 正規表現ではなく ``str.replace`` による substring 置換 (大小区別)。
    - ルールは順序どおり適用 (前段の結果に次段がかかる)。
    - ``from`` が空文字のルールは無視する。
    - 空入力 / 空ルールはそのまま返す (既存挙動を壊さない)。
    """
    if not text or not rules:
        return text
    for src, dst in rules:
        if src:
            text = text.replace(src, dst)
    return text
