"""Phase F: 録音開始時の画面コンテキストを Accessibility (AX) で読み取る。

「今フォーカスしている入力欄の文字・選択テキスト・ウィンドウタイトル・
アプリ名」を取得し、そこから固有名詞・専門用語を抽出して Whisper の
``initial_prompt`` と整形 LLM の文脈ヒントに渡す。初出の語 (相手の名前・
プロジェクト名など) を「今画面に映っているか」で補強し、書き起こし精度を
上げるのが狙い。

設計上の柱:

- **追加 pip 依存ゼロ**。AX 系シンボルは pyobjc (rumps の推移依存) の
  ``ApplicationServices`` に全て含まれる。import は ``feedback.py`` 同様に
  try/except で optional 化し、失敗しても落ちない。
- **ハング対策**: ``AXUIElementCopyAttributeValue`` は応答しないアプリで
  ブロックしうる。``AXUIElementSetMessagingTimeout`` を一次防衛に据え、
  呼び出し側 (app.py) はさらに別スレッド + join timeout で囲う。
- **プライバシー**: パスワード欄 (``AXSecureTextField``) と denylist アプリは
  本文を読まない。読んだ値は ``redact_secrets`` で機密パターンをマスク。
  生テキストは履歴・ログに残さない (呼び出し側の責務)。全処理ローカル。
- **戻りコード**: AX は例外でなく ``(AXError, value)`` を返す。``kAXErrorSuccess``
  以外は「読めなかった」として空に倒す。値が文字列でない (構造体/AXValue/
  リスト) 場合も無視する。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from voiceinput.text_filter import redact_secrets
from voiceinput.vocabulary import extract_terms

_logger = logging.getLogger("voiceinput.screen_context")

try:
    from AppKit import NSWorkspace
    from ApplicationServices import (
        AXIsProcessTrusted,
        AXUIElementCopyAttributeValue,
        AXUIElementCreateApplication,
        AXUIElementSetMessagingTimeout,
        kAXErrorSuccess,
        kAXFocusedUIElementAttribute,
        kAXFocusedWindowAttribute,
        kAXRoleAttribute,
        kAXSelectedTextAttribute,
        kAXTitleAttribute,
        kAXValueAttribute,
    )

    _AX_OK = True
except Exception:  # pragma: no cover - macOS 以外 / pyobjc 欠如時
    _AX_OK = False

# パスワード欄の AX role。これは設定に関係なく本文を読まない。
_SECURE_ROLE = "AXSecureTextField"

# 既知の機密アプリ (bundle id)。該当時は本文を読まずアプリ名のみ。
DEFAULT_DENYLIST: tuple[str, ...] = (
    "com.apple.keychainaccess",
    "com.agilebits.onepassword7",
    "com.agilebits.onepassword4",
    "com.1password.1password",
    "com.bitwarden.desktop",
    "com.lastpass.lastpassmacdesktop",
)

# AX メッセージングのタイムアウト (秒)。ハングの一次防衛。
_AX_MESSAGING_TIMEOUT = 0.2


@dataclass
class ScreenContext:
    """録音開始時点の画面コンテキスト。生テキストはここから先に持ち出さない。"""

    app_name: str = ""
    bundle_id: str = ""
    window_title: str = ""
    focused_text: str = ""
    selected_text: str = ""
    role: str = ""
    trusted: bool = False


def available() -> bool:
    """AX 系シンボルが import できている (= キャプチャ実行可能) か。"""
    return _AX_OK


# ---------------------------------------------------------------------------
# 純関数 (AX 非依存・テスト可能)
# ---------------------------------------------------------------------------


def context_terms(ctx: ScreenContext) -> list[str]:
    """ScreenContext から Whisper 用の語リストを抽出する。

    優先度: 選択テキスト > 入力欄本文 > ウィンドウタイトル。
    ``extract_terms`` (カタカナ/漢字/英数字識別子) を再利用し、出現順に
    重複排除した「語」だけを返す。文・フレーズはそのまま返さない (Whisper の
    幻覚抑制のため)。
    """
    ordered: list[str] = []
    for source in (ctx.selected_text, ctx.focused_text, ctx.window_title):
        if source:
            ordered.extend(extract_terms(source))
    seen: set[str] = set()
    out: list[str] = []
    for term in ordered:
        if term in seen:
            continue
        seen.add(term)
        out.append(term)
    return out


def should_read_value(role: str, bundle_id: str, denylist: Iterable[str]) -> bool:
    """入力欄本文を読んでよいか (パスワード欄 / denylist を弾く純判定)。"""
    if role == _SECURE_ROLE:
        return False
    if bundle_id and bundle_id in tuple(denylist):
        return False
    return True


def resolve_mode(
    app_name: str,
    bundle_id: str,
    window_title: str,
    rules: Iterable[tuple[str, str, str]],
) -> str | None:
    """app_mode ルールを順に評価し、最初にヒットした整形モードを返す (純判定)。

    各ルールは ``(match_type, pattern, mode)``:

    - ``("bundle", pattern, mode)``: ``bundle_id`` が ``pattern`` と完全一致で
      ヒット。AX 権限不要 (bundle は NSWorkspace で常時取れる) なので、
      ``screen_context`` が無効でも機能する基本判定。
    - ``("title", pattern, mode)``: ``window_title`` に ``pattern`` が部分一致
      (大文字小文字無視) でヒット。``window_title`` は AX 経由でしか取れない
      ため、空のとき (AX 無効/未許可) は title ルールをスキップする。

    どのルールにもヒットしなければ ``None`` (呼び出し側で手動モードに倒す)。
    """
    title_lower = window_title.lower() if window_title else ""
    for match_type, pattern, mode in rules:
        if match_type == "bundle":
            if bundle_id and bundle_id == pattern:
                return mode
        elif match_type == "title":
            if title_lower and pattern and pattern.lower() in title_lower:
                return mode
    return None


# ---------------------------------------------------------------------------
# AX 読み取り (権限依存)
# ---------------------------------------------------------------------------


def _ax_string(element, attr) -> str:
    """AXUIElementCopyAttributeValue を呼び、文字列値だけ受理する。

    - AX は例外でなく ``(err, value)`` を返す。
    - ``kAXErrorSuccess`` 以外は空。
    - 値が文字列でない (AXValue 構造体・リスト等) 場合は空。
    """
    try:
        err, value = AXUIElementCopyAttributeValue(element, attr, None)
    except Exception:
        return ""
    if err != kAXErrorSuccess or value is None:
        return ""
    # 文字列値のみ受理する。pyobjc では CFString/NSString は Python str に
    # ブリッジされる。AXValue 構造体・数値・bool・配列・辞書などは
    # ここで弾く (str() フォールバックを置くと型エラー値が context_terms まで
    # 漏れるため、あえて持たない)。
    if isinstance(value, str):
        return value
    return ""


def _set_timeout(element) -> None:
    try:
        AXUIElementSetMessagingTimeout(element, _AX_MESSAGING_TIMEOUT)
    except Exception:
        pass


def _window_title(app_element) -> str:
    """アプリ要素 → focused window → title を読む (失敗しても空)。"""
    try:
        err, window = AXUIElementCopyAttributeValue(
            app_element, kAXFocusedWindowAttribute, None
        )
    except Exception:
        return ""
    if err != kAXErrorSuccess or window is None:
        return ""
    _set_timeout(window)
    return _ax_string(window, kAXTitleAttribute)


def frontmost_app() -> tuple[str, str]:
    """フォアグラウンドアプリの ``(app_name, bundle_id)`` を返す。

    ``NSWorkspace`` のみ使用 (AX 権限不要・高速・非ブロッキング)。app_mode の
    自動切替判定に使う軽量取得で、``screen_context_enabled`` や AX 信頼状態・
    ``logging_enabled`` には一切依存しない。録音開始時のクリティカルパスから
    呼んでも安全。取得できなければ ``("", "")``。
    """
    if not _AX_OK:
        return ("", "")
    try:
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            return ("", "")
        return (app.localizedName() or "", app.bundleIdentifier() or "")
    except Exception:
        _logger.debug("frontmost_app read failed", exc_info=True)
        return ("", "")


def capture(
    *,
    read_value: bool = True,
    max_value_chars: int = 600,
    denylist: Iterable[str] = DEFAULT_DENYLIST,
) -> ScreenContext:
    """フォアグラウンドアプリの画面コンテキストを読む。

    例外は外に出さず、読めなかった項目は空のまま返す。AX が信頼されていない /
    pyobjc が無い環境では app 名のみ (それも取れなければ全空) を返す。
    """
    ctx = ScreenContext()
    denylist = tuple(denylist)

    # アプリ名/バンドルID は AX 権限なしでも取得可能。
    try:
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is not None:
            ctx.app_name = app.localizedName() or ""
            ctx.bundle_id = app.bundleIdentifier() or ""
            _pid = app.processIdentifier()
        else:
            _pid = None
    except Exception:
        _logger.debug("frontmost app read failed", exc_info=True)
        _pid = None

    if not _AX_OK:
        return ctx

    try:
        ctx.trusted = bool(AXIsProcessTrusted())
    except Exception:
        ctx.trusted = False
    if not ctx.trusted:
        return ctx

    try:
        # アプリ要素 (pid 経由) から focused element と window を辿る。
        if _pid is None:
            return ctx
        app_element = AXUIElementCreateApplication(_pid)
        _set_timeout(app_element)

        # window title (本文より機密性が低いので read_value 判定の外で取得)
        ctx.window_title = _window_title(app_element)

        err, focused = AXUIElementCopyAttributeValue(
            app_element, kAXFocusedUIElementAttribute, None
        )
        if err == kAXErrorSuccess and focused is not None:
            _set_timeout(focused)
            ctx.role = _ax_string(focused, kAXRoleAttribute)
            if read_value and should_read_value(ctx.role, ctx.bundle_id, denylist):
                val = _ax_string(focused, kAXValueAttribute)
                sel = _ax_string(focused, kAXSelectedTextAttribute)
                if val:
                    ctx.focused_text = redact_secrets(val[:max_value_chars])
                if sel:
                    ctx.selected_text = redact_secrets(sel[:max_value_chars])
    except Exception:
        _logger.debug("AX capture failed", exc_info=True)

    return ctx
