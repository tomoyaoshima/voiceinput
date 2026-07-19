from dataclasses import dataclass, replace
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    hotkey: str
    whisper_model: str
    whisper_language: str
    ollama_endpoint: str
    ollama_model: str
    ollama_timeout_sec: float
    ollama_temperature: float
    ollama_num_predict: int
    ollama_num_ctx: int
    ollama_keep_alive: str
    default_format_mode: str
    auto_paste: bool
    notify: bool
    logging_enabled: bool
    log_dir: Path
    prompts_dir: Path
    # Custom vocabulary (Phase E)
    vocabulary_enabled: bool
    vocabulary_manual: tuple[str, ...]
    vocabulary_history_size: int
    vocabulary_top_n: int
    vocabulary_rebuild_every: int
    vocabulary_max_chars: int
    # Whisper initial_prompt の形式 (Phase G)。
    # "list": 「語、語、語。」句読点プライミング + 末尾優先 (推奨)
    # "terms": 従来のスペース区切り (rollback 用)
    vocabulary_prompt_style: str
    # Screen context awareness (Phase F)
    screen_context_enabled: bool
    screen_context_terms_max_chars: int
    screen_context_llm_max_chars: int
    screen_context_read_focused_value: bool
    screen_context_denylist_bundles: tuple[str, ...]
    # 決定的テキスト置換 (STT/LLM が誤変換しやすい固有名詞・商品名を最終固定)
    # (from, to) の順序付きタプル列。空なら無効。
    replacements: tuple[tuple[str, str], ...]
    # アプリ別モード自動切替: フォーカス中アプリ/ウィンドウに応じて整形モードを
    # 自動選択する。(match_type, pattern, mode) の順序付きタプル列。
    # match_type は "bundle" (bundle_id 完全一致) / "title" (window_title 部分一致)。
    app_mode_enabled: bool
    app_mode_rules: tuple[tuple[str, str, str], ...]


def default_config() -> Config:
    return Config(
        hotkey="<cmd_r>",
        whisper_model="mlx-community/whisper-large-v3-turbo",
        whisper_language="ja",
        ollama_endpoint="http://localhost:11434",
        # 実運用比較 (qwen2.5:7b/14b, qwen3:8b, gemma4:e4b/26b) の結論として
        # gemma4:e4b を default に採用:
        #   - 控えめな整形 (発話を勝手に言い換えない、誤訂正をしない)
        #   - 速い (~0.4s) - menu bar UX に体感ピッタリ
        #   - meta-commentary を出さない (post-processing 不要)
        # 別モデルが欲しい場合は menu bar から選択 → state.json に永続化される。
        ollama_model="gemma4:e4b",
        ollama_timeout_sec=60.0,
        ollama_temperature=0.2,
        # 音声入力 1 ターンの実態 (~200-400 文字) には 512 でも足りるが、
        # gemma4:26b など内部 reasoning する系は出力前に 700+ token 消費する。
        # 短い stop で早期終了する fast model (qwen2.5 など) のレイテンシは
        # 上限値に依存しないので、無理せず 1024 を default にしておく。
        ollama_num_predict=1024,
        ollama_num_ctx=1024,
        # メニューバー常駐アプリなので Ollama 側もほぼ常駐させる方が UX 的に正しい
        ollama_keep_alive="24h",
        default_format_mode="clean",
        auto_paste=True,
        notify=True,
        logging_enabled=True,
        log_dir=Path("~/Library/Logs/voiceinput").expanduser(),
        prompts_dir=Path("./prompts").resolve(),
        # Phase E: カスタム語彙 (Whisper initial_prompt)
        vocabulary_enabled=True,
        vocabulary_manual=(),
        vocabulary_history_size=100,
        vocabulary_top_n=30,
        vocabulary_rebuild_every=10,
        vocabulary_max_chars=200,
        vocabulary_prompt_style="list",
        # Phase F: 画面コンテキスト認識 (録音開始時に AX で画面テキストを読む)
        screen_context_enabled=True,
        # Whisper prompt に足す画面語の文字数枠。幻覚抑制のため控えめ。
        screen_context_terms_max_chars=120,
        # LLM 整形に渡す「表記参考語」の文字数枠。語リストのみなので短くて十分
        # (長いと挿入・引用の誘惑が増えるので控えめに)。
        screen_context_llm_max_chars=120,
        # 入力欄本文を読むか。False なら件名/選択テキスト/アプリ名のみ。
        screen_context_read_focused_value=True,
        # 本文を読まない機密アプリ (空 = screen_context.DEFAULT_DENYLIST を使う)。
        screen_context_denylist_bundles=(),
        # 決定的テキスト置換 (既定は無効 = 空タプル)。
        replacements=(),
        # アプリ別モード自動切替 (既定は無効)。
        app_mode_enabled=False,
        app_mode_rules=(),
    )


def _parse_replacements(
    raw: object, default: tuple[tuple[str, str], ...]
) -> tuple[tuple[str, str], ...]:
    """config.yaml の ``replacements`` (list of {from, to}) をタプル列に変換する。

    - ``raw`` が None / 空なら ``default`` を返す。
    - 各要素は ``{"from": ..., "to": ...}`` を期待。``from`` が空のものは捨てる。
    - 型が壊れている要素は黙って読み飛ばす (非エンジニアの手書きに耐える)。
    """
    if not raw or not isinstance(raw, list):
        return default
    out: list[tuple[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        src = item.get("from")
        dst = item.get("to", "")
        if src is None:
            continue
        src = str(src)
        dst = str(dst) if dst is not None else ""
        if src:
            out.append((src, dst))
    return tuple(out)


def _parse_app_mode_rules(
    raw: object, default: tuple[tuple[str, str, str], ...]
) -> tuple[tuple[str, str, str], ...]:
    """config.yaml の ``app_mode.rules`` を (match_type, pattern, mode) 列に変換。

    各要素は ``{"match_bundle": ..., "mode": ...}`` または
    ``{"match_title": ..., "mode": ...}`` を期待。``mode`` 欠落や match パターン
    欠落の要素は読み飛ばす。両方指定された場合は bundle を優先する。
    """
    if not raw or not isinstance(raw, list):
        return default
    out: list[tuple[str, str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        mode = item.get("mode")
        if not mode:
            continue
        mode = str(mode)
        bundle = item.get("match_bundle")
        title = item.get("match_title")
        if bundle:
            out.append(("bundle", str(bundle), mode))
        elif title:
            out.append(("title", str(title), mode))
    return tuple(out)


def load_config(path: Path | None = None) -> Config:
    cfg = default_config()
    if path is None:
        candidates = [
            Path("./config.yaml"),
            Path("~/.config/voiceinput/config.yaml").expanduser(),
        ]
        path = next((p for p in candidates if p.exists()), None)
    if path is None or not path.exists():
        return cfg
    data = yaml.safe_load(path.read_text()) or {}
    whisper = data.get("whisper") or {}
    ollama = data.get("ollama") or {}
    voc = data.get("vocabulary") or {}
    sc = data.get("screen_context") or {}
    am = data.get("app_mode") or {}
    return replace(
        cfg,
        hotkey=data.get("hotkey", cfg.hotkey),
        whisper_model=whisper.get("model", cfg.whisper_model),
        whisper_language=whisper.get("language", cfg.whisper_language),
        ollama_endpoint=ollama.get("endpoint", cfg.ollama_endpoint),
        ollama_model=ollama.get("model", cfg.ollama_model),
        ollama_timeout_sec=ollama.get("timeout_sec", cfg.ollama_timeout_sec),
        ollama_temperature=ollama.get("temperature", cfg.ollama_temperature),
        ollama_num_predict=ollama.get("num_predict", cfg.ollama_num_predict),
        ollama_num_ctx=ollama.get("num_ctx", cfg.ollama_num_ctx),
        ollama_keep_alive=ollama.get("keep_alive", cfg.ollama_keep_alive),
        default_format_mode=data.get("default_format_mode", cfg.default_format_mode),
        auto_paste=data.get("auto_paste", cfg.auto_paste),
        notify=data.get("notify", cfg.notify),
        logging_enabled=data.get("logging_enabled", cfg.logging_enabled),
        log_dir=Path(data["log_dir"]).expanduser() if "log_dir" in data else cfg.log_dir,
        prompts_dir=Path(data["prompts_dir"]).resolve() if "prompts_dir" in data else cfg.prompts_dir,
        vocabulary_enabled=voc.get("enabled", cfg.vocabulary_enabled),
        vocabulary_manual=tuple(voc["manual"]) if "manual" in voc else cfg.vocabulary_manual,
        vocabulary_history_size=voc.get("history_size", cfg.vocabulary_history_size),
        vocabulary_top_n=voc.get("top_n", cfg.vocabulary_top_n),
        vocabulary_rebuild_every=voc.get("rebuild_every", cfg.vocabulary_rebuild_every),
        vocabulary_max_chars=voc.get("max_chars", cfg.vocabulary_max_chars),
        vocabulary_prompt_style=voc.get("prompt_style", cfg.vocabulary_prompt_style),
        screen_context_enabled=sc.get("enabled", cfg.screen_context_enabled),
        screen_context_terms_max_chars=sc.get(
            "terms_max_chars", cfg.screen_context_terms_max_chars
        ),
        screen_context_llm_max_chars=sc.get(
            "llm_max_chars", cfg.screen_context_llm_max_chars
        ),
        screen_context_read_focused_value=sc.get(
            "read_focused_value", cfg.screen_context_read_focused_value
        ),
        screen_context_denylist_bundles=(
            tuple(sc["denylist_bundles"])
            if "denylist_bundles" in sc
            else cfg.screen_context_denylist_bundles
        ),
        replacements=_parse_replacements(
            data.get("replacements"), cfg.replacements
        ),
        app_mode_enabled=am.get("enabled", cfg.app_mode_enabled),
        app_mode_rules=_parse_app_mode_rules(
            am.get("rules"), cfg.app_mode_rules
        ),
    )
