import datetime as _dt
import re
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import rumps

try:
    from Foundation import NSOperationQueue
except Exception:  # pragma: no cover
    NSOperationQueue = None  # type: ignore

from voiceinput import feedback, screen_context, vad
from voiceinput.config import Config, default_config, load_config
from voiceinput.history import History, HistoryEntry, make_entry
from voiceinput.hotkey import HotkeyManager
from voiceinput.llm import FormatPipeline, OllamaClient
from voiceinput.logger import setup_logger
from voiceinput.paste import copy_to_clipboard, notify, paste_active_app
from voiceinput.recorder import Recorder
from voiceinput.state import RecordingStateMachine
from voiceinput.stt import WhisperSTT
from voiceinput.text_filter import apply_replacements
from voiceinput.user_state import UserStateStore
from voiceinput.vocabulary import (
    VocabularyBuilder,
    compose_initial_prompt,
    parse_vocabulary_lines,
    render_prompt_list,
    render_vocabulary_file,
)

TITLE_IDLE = "🎤"
TITLE_HELD = "🔴 REC"
TITLE_LATCHED = "🟠 REC"
TITLE_PROCESSING = "⏳ …"

SOUND_START = "start"
SOUND_STOP = "stop"
SOUND_ERROR = "error"

HISTORY_MENU_LIMIT = 10  # サブメニューに何件表示するか
HISTORY_LABEL_CHARS = 32

# silence が何回連続したら "マイク OFF かも" と警告するか
SILENCE_WARNING_THRESHOLD = 3

# 画面コンテキストのキャプチャスレッドがこの秒数以上生きていたら、
# AX 呼び出しがハングしているとみなして見限る (永続スキップ防止)。
# 通常は AXUIElementSetMessagingTimeout(0.2s) により ~1s 以内に終わる。
HUNG_CAPTURE_SEC = 5.0

# STT Model サブメニューに出す候補 (Phase G)。
# turbo が速度/精度バランスのデフォルト。精度重視なら large-v3 (3-4 倍遅い)、
# 日本語特化の kotoba (個人変換・句読点弱め、LLM 整形が補完する前提)。
STT_MODEL_CANDIDATES = (
    "mlx-community/whisper-large-v3-turbo",
    "mlx-community/whisper-large-v3-mlx",
    "mlx-community/whisper-large-v3-turbo-q4",
    "kaiinui/kotoba-whisper-v2.0-mlx",
)


def _run_on_main_thread(fn: Callable[[], None]) -> None:
    """UI 更新を main thread で実行する。

    rumps の `App.title` などは内部で AppKit / NSStatusItem を更新するため、
    別スレッドから書き換えても画面に反映されないことがある。NSOperationQueue
    の mainQueue に block を投げると次の RunLoop で確実に main で走る。
    """
    if NSOperationQueue is None:
        fn()
        return
    NSOperationQueue.mainQueue().addOperationWithBlock_(fn)


def _resolve_pref(*, user_value: str | None, config_value: str, default_value: str) -> str:
    """state.json と config.yaml と組み込みデフォルトから初期値を選ぶ。

    優先順位: ``config.yaml`` で明示指定されている (= デフォルト値と異なる)
    場合はそれが最優先。明示されていない場合のみ ``state.json`` 由来の値を
    使う。state も無ければ組み込みデフォルト。
    """
    if config_value != default_value:
        return config_value
    if user_value:
        return user_value
    return default_value


def _resolve_bool_pref(*, user_value: bool | None, default_value: bool) -> bool:
    """bool 設定の初期値を state.json 優先で解決する。

    menu トグルで保存された ``user_value`` (True/False) があればそれを使い、
    None (未設定) なら config/組み込みの ``default_value`` に従う。
    """
    if user_value is None:
        return default_value
    return user_value


class VoiceInputApp(rumps.App):
    def __init__(self, config: Config) -> None:
        super().__init__("voiceinput", title=TITLE_IDLE, quit_button="Quit")
        self.config = config
        self.logger = setup_logger(config.log_dir, config.logging_enabled)

        # 永続化された前回のメニュー選択を読み込み、config.yaml で明示指定
        # された値を上書きしないように優先順位を解決する。
        self.user_state_store = UserStateStore()
        defaults = default_config()
        persisted = self.user_state_store.load()
        initial_llm_model = _resolve_pref(
            user_value=persisted.llm_model,
            config_value=config.ollama_model,
            default_value=defaults.ollama_model,
        )
        initial_format_mode = _resolve_pref(
            user_value=persisted.format_mode,
            config_value=config.default_format_mode,
            default_value=defaults.default_format_mode,
        )
        initial_whisper_model = _resolve_pref(
            user_value=persisted.whisper_model,
            config_value=config.whisper_model,
            default_value=defaults.whisper_model,
        )

        self.recorder = Recorder()
        self.stt = WhisperSTT(initial_whisper_model, config.whisper_language)
        self.llm = OllamaClient(
            config.ollama_endpoint,
            initial_llm_model,
            config.ollama_timeout_sec,
            temperature=config.ollama_temperature,
            num_predict=config.ollama_num_predict,
            num_ctx=config.ollama_num_ctx,
            keep_alive=config.ollama_keep_alive,
        )
        self.format_pipeline = FormatPipeline(self.llm, config.prompts_dir)
        self.format_mode = initial_format_mode
        self.history = History()
        self._history_lookup: dict[str, HistoryEntry] = {}
        self.runtime = SimpleNamespace(
            auto_paste=config.auto_paste,
            logging_enabled=config.logging_enabled,
        )
        self._toggle_active = False
        self._silence_streak = 0

        # Phase E: カスタム語彙 (Whisper initial_prompt). enabled=False なら
        # builder は持たず、_initial_prompt は常に空。
        # 手動リストは config.yaml + state.json (GUI で追加した語) を合算する。
        if config.vocabulary_enabled:
            self.vocab_builder: VocabularyBuilder | None = VocabularyBuilder(
                history=self.history,
                manual=self._compose_manual_vocab(persisted.manual_vocabulary),
                history_size=config.vocabulary_history_size,
                top_n=config.vocabulary_top_n,
                max_chars=config.vocabulary_max_chars,
            )
        else:
            self.vocab_builder = None
        self._initial_prompt: str = ""
        # _run_pipeline 完了ごとにインクリメント。rebuild_every に達したら
        # バックグラウンドで vocabulary を再構築する。
        self._pipeline_count = 0

        # Phase F: 画面コンテキスト認識。menu トグル (state.json) を config
        # デフォルトより優先する。生テキストは _screen_ctx の外へ出さない。
        self.screen_context_enabled = _resolve_bool_pref(
            user_value=persisted.screen_context_enabled,
            default_value=config.screen_context_enabled,
        )
        self._screen_ctx: screen_context.ScreenContext | None = None
        self._screen_ctx_lock = threading.Lock()
        self._ctx_generation = 0
        self._capture_thread: threading.Thread | None = None
        self._capture_started_at = 0.0
        # 現在の録音が capture を開始した時の世代 ID。capture を開始しなかった
        # 録音 (screen OFF / logging OFF / AX 無効) では None。stop 時に
        # pipeline へ渡し、None なら screen_ctx を一切消費させない。
        self._active_capture_gen: int | None = None
        # denylist は config 指定があればそれ、無ければモジュール既定。
        self._screen_denylist = (
            config.screen_context_denylist_bundles
            or screen_context.DEFAULT_DENYLIST
        )

        # app_mode: 録音開始時のフォアグラウンドアプリ (app_name, bundle_id)。
        # screen_context とは独立に NSWorkspace で取得し、_run_pipeline で
        # window_title (取れていれば) と合わせて整形モードを自動判定する。
        self._record_app: tuple[str, str] = ("", "")

        self.state_machine = RecordingStateMachine(
            on_start=self._on_recording_start,
            on_stop=self._on_recording_stop,
            on_latch=self._on_recording_latch,
        )
        self.hotkey = HotkeyManager()
        self.hotkey.register_press_release(
            config.hotkey,
            on_press=self._dispatch_press,
            on_release=self._dispatch_release,
        )
        # Esc 単独押しで録音をキャンセル (audio 破棄、文字起こし・ペーストなし)
        self.hotkey.register_cancel(self._on_cancel_recording)

        self._build_menu()

        threading.Thread(target=self._warmup, daemon=True).start()
        self.logger.info(
            "voiceinput started (hotkey=%s mode=%s model=%s solo=%s vocab=%s screen_ctx=%s)",
            config.hotkey,
            self.format_mode,
            self.llm.model,
            self.hotkey.is_solo_modifier,
            "on" if self.vocab_builder else "off",
            "on" if (self.screen_context_enabled and screen_context.available()) else "off",
        )

    # --- UI helpers (main thread guaranteed) ---

    def _set_ui(self, title: str, status: str) -> None:
        def update():
            self.title = title
            self._status_item.title = f"Status: {status}"
        _run_on_main_thread(update)

    # --- hotkey dispatch ---

    def _dispatch_press(self) -> None:
        if self.hotkey.is_solo_modifier:
            self._on_solo_toggle()
        else:
            self.state_machine.on_press()

    def _dispatch_release(self) -> None:
        if self.hotkey.is_solo_modifier:
            return
        self.state_machine.on_release()

    # --- Esc cancel ---

    def _on_cancel_recording(self) -> None:
        """Esc 押下で録音を中断する。

        録音中でなければ no-op (idle / processing 中の Esc は無害化)。
        中断時は audio バッファを破棄し、文字起こし・LLM・ペーストには
        渡さない。SOUND_ERROR + UI を idle に戻す。

        solo modifier path (cmd_r 等) と combo path (state_machine) の
        両方に対応する。listener thread から呼ばれるので _set_ui が
        メインスレッド経由で UI を触る。
        """
        from voiceinput.state import RecordingMode

        canceled = False
        if self.hotkey.is_solo_modifier:
            if self._toggle_active:
                self._toggle_active = False
                canceled = True
        else:
            if self.state_machine.mode != RecordingMode.IDLE:
                self.state_machine.force_idle()
                canceled = True

        if not canceled:
            return

        try:
            self.recorder.stop()  # 戻り値の audio は意図的に捨てる
        except Exception:
            self.logger.exception("recorder.stop failed during cancel")

        # 旧キャプチャ結果を無効化 (世代を進めて in-flight ワーカーの書込を破棄)
        self._invalidate_screen_ctx()
        # app_mode: 録音時アプリの記録もクリア (次録音は開始時に再取得するので
        # 実害はないが、キャンセル済み録音のアプリ判定が残らないよう明示)。
        self._record_app = ("", "")

        feedback.play(SOUND_ERROR)
        self._set_ui(TITLE_IDLE, "idle")
        notify(
            "voiceinput",
            "❌ 録音をキャンセルしました",
            enabled=self.config.notify,
        )
        self.logger.info("recording canceled via Esc")

    # --- app_mode (アプリ別モード自動切替) ---

    def _capture_record_app(self) -> None:
        """録音開始時にフォアグラウンドアプリを記録する (app_mode 用)。

        NSWorkspace のみで軽量・非ブロッキング。app_mode 無効時は空に倒す。
        録音停止後に画面が変わっても、録音時点のアプリで判定できるよう、
        ここで snapshot しておく (screen_context の有効/無効に依存しない)。
        """
        if not self.config.app_mode_enabled:
            self._record_app = ("", "")
            return
        try:
            self._record_app = screen_context.frontmost_app()
        except Exception:
            self.logger.debug("frontmost app capture failed", exc_info=True)
            self._record_app = ("", "")

    def _resolve_effective_mode(
        self, screen_ctx: "screen_context.ScreenContext | None"
    ) -> tuple[str, str | None]:
        """app_mode 有効時、録音時アプリ + window_title から整形モードを自動判定。

        戻り値 ``(mode, auto_source)``。``auto_source`` は自動判定の根拠 (bundle_id
        や app 名) で、手動モードのままなら ``None``。bundle 判定は AX 不要なので
        screen_ctx が無くても効き、title 判定は window_title が取れた時のみ効く。
        未知モードを指すルールは無視して手動モードにフォールバックする。
        """
        if not self.config.app_mode_enabled:
            return self.format_mode, None
        app_name, bundle_id = self._record_app
        window_title = screen_ctx.window_title if screen_ctx is not None else ""
        auto = screen_context.resolve_mode(
            app_name, bundle_id, window_title, self.config.app_mode_rules
        )
        if auto is None:
            return self.format_mode, None
        if auto not in self.format_pipeline.available_modes():
            self.logger.warning(
                "app_mode rule points to unknown mode %r; using manual %s",
                auto,
                self.format_mode,
            )
            return self.format_mode, None
        source = bundle_id or app_name or window_title or "?"
        return auto, source

    # --- 画面コンテキスト (Phase F) ---

    def _start_screen_capture(self) -> None:
        """録音開始時に画面コンテキストを別スレッドで読む。

        録音を 1ms もブロックしないため daemon thread。世代 ID を採番し、
        ワーカーは自分の世代が最新の時だけ ``_screen_ctx`` に書き込む。
        AX がハングしうるアプリ対策として、直前のキャプチャがまだ生存して
        いれば今回はスキップ (スレッドリーク防止)。
        """
        # capture を開始しない録音は active_gen=None にして、stop 時に
        # pipeline へ「この録音に screen_ctx は無い」と明示する。
        if not self.screen_context_enabled:
            self._active_capture_gen = None
            return
        # Logging OFF 連動: 機密配慮で screen context も止める。
        if not self.runtime.logging_enabled:
            self._active_capture_gen = None
            return
        if not screen_context.available():
            self._active_capture_gen = None
            return
        with self._screen_ctx_lock:
            self._ctx_generation += 1
            self._screen_ctx = None
            gen = self._ctx_generation
            # この録音の世代を記録 (worker を起動できなくても、空の ctx を
            # 正しく「この世代のもの」として扱えるようにする)。
            self._active_capture_gen = gen
            prev = self._capture_thread
            if prev is not None and prev.is_alive():
                # 前回の AX 呼び出しがまだ生きている = ハング中の可能性。
                # 通常は messaging timeout (0.2s) で ~1s 以内に終わるので、
                # HUNG_CAPTURE_SEC 以上生きていれば見限って新規キャプチャを
                # 起動する (旧スレッドはリークするが永続ロックを防ぐ)。
                age = time.monotonic() - self._capture_started_at
                if age < HUNG_CAPTURE_SEC:
                    # まだ正常範囲内 → 今回は worker を起動しない。_screen_ctx は
                    # None のままなので、stop 時には空コンテキストになる。
                    return
            t = threading.Thread(
                target=self._capture_worker, args=(gen,), daemon=True
            )
            self._capture_thread = t
            self._capture_started_at = time.monotonic()
        t.start()

    def _capture_worker(self, gen: int) -> None:
        try:
            ctx = screen_context.capture(
                read_value=self.config.screen_context_read_focused_value,
                max_value_chars=600,
                denylist=self._screen_denylist,
            )
        except Exception:
            self.logger.debug("screen capture failed", exc_info=True)
            return
        with self._screen_ctx_lock:
            if gen == self._ctx_generation:
                self._screen_ctx = ctx

    def _invalidate_screen_ctx(self) -> None:
        """世代を進めて現在の screen_ctx を捨てる (キャンセル/消費後用)。"""
        with self._screen_ctx_lock:
            self._ctx_generation += 1
            self._screen_ctx = None

    def _capture_gen_for_stop(self) -> int | None:
        """stop 時に pipeline へ渡す世代 ID を読む。

        現在の録音が capture を開始していれば その世代、していなければ None。
        """
        with self._screen_ctx_lock:
            return self._active_capture_gen

    def _take_screen_ctx(
        self, expected_gen: int | None
    ) -> "screen_context.ScreenContext | None":
        """この録音の世代に紐づく screen_ctx を取り出し、内部をクリアする。

        ``expected_gen`` が None (= この録音は capture していない)、または
        現在の世代と一致しない (= 既に次録音が始まっている) 場合は None を
        返し、取り違え・誤消費を防ぐ。
        """
        if expected_gen is None:
            return None
        with self._screen_ctx_lock:
            if self._ctx_generation != expected_gen:
                return None
            ctx = self._screen_ctx
            self._screen_ctx = None
        return ctx

    # --- solo-modifier toggle ---

    def _on_solo_toggle(self) -> None:
        if not self._toggle_active:
            # 開始パス。recorder.start() は sounddevice が PortAudioError を
            # 投げる可能性がある (マイク占有、AirPods 切替、TCC 拒否など)。
            # その場合 _toggle_active を True のまま放置すると、次の press が
            # stop パスに行って空バッファ → 0 samples ログという厄介な
            # silent failure になるので、必ずロールバックする。
            self._toggle_active = True
            feedback.play(SOUND_START)
            try:
                self.recorder.start()
            except Exception as e:
                self._toggle_active = False
                feedback.play(SOUND_ERROR)
                self._set_ui(TITLE_IDLE, "idle")
                self.logger.exception("recorder.start failed (toggle)")
                notify(
                    "voiceinput",
                    f"録音開始に失敗しました: {e}",
                    enabled=self.config.notify,
                )
                return
            self._capture_record_app()
            self._start_screen_capture()
            self._set_ui(TITLE_LATCHED, "recording")
            notify(
                "voiceinput",
                "🟠 録音中 (もう一度押すと停止)",
                enabled=self.config.notify,
            )
            self.logger.info("toggle recording started")
        else:
            self._toggle_active = False
            feedback.play(SOUND_STOP)
            audio = self.recorder.stop()
            # この録音の capture 世代を pipeline に渡す。pipeline がこの ctx を
            # 消費する前に次録音が始まっても、世代不一致で取り違えない。capture
            # していない録音なら None で screen_ctx を一切消費させない。
            gen = self._capture_gen_for_stop()
            self._set_ui(TITLE_PROCESSING, "processing")
            self.logger.info("toggle recording stopped (%d samples)", len(audio))
            threading.Thread(
                target=self._run_pipeline, args=(audio, gen), daemon=True
            ).start()

    # --- menu construction ---

    def _build_menu(self) -> None:
        self._status_item = rumps.MenuItem("Status: idle")
        self._fmt_items: dict[str, rumps.MenuItem] = {
            mode: rumps.MenuItem(mode, callback=self._on_select_mode)
            for mode in self.format_pipeline.available_modes()
        }
        self._llm_items: dict[str, rumps.MenuItem] = {}
        self._auto_paste_item = rumps.MenuItem(
            "Auto-paste", callback=self._on_toggle_auto_paste
        )
        self._logging_item = rumps.MenuItem(
            "Logging", callback=self._on_toggle_logging
        )
        # Phase F: 画面コンテキスト認識のトグル (AX が使える環境のみ表示)。
        self._screen_context_item = rumps.MenuItem(
            "Screen context", callback=self._on_toggle_screen_context
        )
        menu: list = [
            self._status_item,
            None,
            ("Format", list(self._fmt_items.values())),
            ("LLM Model", self._build_llm_items()),
            ("STT Model", self._build_stt_items()),
            ("History", self._build_history_items()),
        ]
        # Vocabulary 機能が有効なときだけメニューを出す
        if self.vocab_builder is not None:
            menu.append(("Vocabulary", self._build_vocabulary_items()))
        menu.extend([self._auto_paste_item, self._logging_item])
        if screen_context.available():
            menu.append(self._screen_context_item)
        self.menu = menu
        self._auto_paste_item.state = self.runtime.auto_paste
        self._logging_item.state = self.runtime.logging_enabled
        self._screen_context_item.state = self.screen_context_enabled
        self._mark_format_mode(self.format_mode)

    def _mark_format_mode(self, mode: str) -> None:
        for name, item in self._fmt_items.items():
            item.state = name == mode

    # --- LLM model submenu ---

    def _build_llm_items(self) -> list:
        available = self.llm.list_models()
        if self.llm.model not in available:
            available = [self.llm.model] + available
        self._llm_items = {}
        items: list = []
        for name in available:
            item = rumps.MenuItem(name, callback=self._on_select_llm)
            item.state = name == self.llm.model
            self._llm_items[name] = item
            items.append(item)
        items.append(None)
        items.append(
            rumps.MenuItem("Refresh model list", callback=self._on_refresh_llm)
        )
        return items

    def _refresh_llm_menu(self) -> None:
        llm_menu = self.menu.get("LLM Model")
        if llm_menu is None:
            return

        def update():
            llm_menu.clear()
            for item in self._build_llm_items():
                if item is None:
                    llm_menu.add(rumps.separator)
                else:
                    llm_menu.add(item)
        _run_on_main_thread(update)

    def _on_select_llm(self, sender) -> None:
        if sender.title == self.llm.model:
            return
        self.llm.set_model(sender.title)
        for name, item in self._llm_items.items():
            item.state = name == sender.title
        # 次回起動時にこのモデル選択を復元できるよう state.json に保存
        try:
            self.user_state_store.update(llm_model=sender.title)
        except OSError:
            self.logger.exception("failed to persist llm model selection")
        notify(
            "voiceinput", f"LLM モデル: {sender.title}", enabled=self.config.notify
        )
        self.logger.info("llm model -> %s", sender.title)
        # 新モデルを背景で warmup (初回呼び出しのコールドスタート抑制)
        threading.Thread(target=self.llm.warmup, daemon=True).start()

    def _on_refresh_llm(self, _) -> None:
        self._refresh_llm_menu()
        notify("voiceinput", "モデル一覧を更新しました", enabled=self.config.notify)

    # --- STT model submenu (Phase G) ---

    def _build_stt_items(self) -> list:
        candidates = list(STT_MODEL_CANDIDATES)
        if self.stt.model_name not in candidates:
            candidates = [self.stt.model_name] + candidates
        self._stt_items: dict[str, rumps.MenuItem] = {}
        items: list = []
        for name in candidates:
            item = rumps.MenuItem(name, callback=self._on_select_stt)
            item.state = name == self.stt.model_name
            self._stt_items[name] = item
            items.append(item)
        return items

    def _mark_stt_model(self) -> None:
        """チェックマークを実際の self.stt.model_name に合わせる。"""
        current = self.stt.model_name

        def update():
            for n, item in self._stt_items.items():
                item.state = n == current
        _run_on_main_thread(update)

    def _on_select_stt(self, sender) -> None:
        name = sender.title
        if name == self.stt.model_name:
            return
        notify(
            "voiceinput",
            f"STT モデル準備中: {name} (初回はダウンロードあり)",
            enabled=self.config.notify,
        )
        self.logger.info("stt model switching -> %s", name)

        def switch_in_background() -> None:
            # switch_model はロード成功後にのみ model_name を書き換える。
            # 失敗時は例外 = model_name は元のまま (revert 不要)。ロードは
            # WhisperSTT の RLock 内で行われ、録音パイプラインと直列化される。
            try:
                self.stt.switch_model(name)
            except Exception:
                self.logger.exception("stt model switch failed")
                self._mark_stt_model()  # チェックを実モデルに戻す
                notify(
                    "voiceinput",
                    f"⚠️ {name} をロードできませんでした。"
                    f"{self.stt.model_name} のまま使います。",
                    enabled=self.config.notify,
                )
                return
            # 成功: state.json へ永続化 + チェック更新
            try:
                self.user_state_store.update(whisper_model=name)
            except OSError:
                self.logger.exception("failed to persist whisper model selection")
            self._mark_stt_model()
            self.logger.info("stt model -> %s", name)
            notify(
                "voiceinput",
                f"STT モデル準備完了: {name}",
                enabled=self.config.notify,
            )

        threading.Thread(target=switch_in_background, daemon=True).start()

    # --- History submenu ---

    def _format_history_label(self, entry: HistoryEntry) -> str:
        dt = _dt.datetime.fromtimestamp(entry.timestamp)
        preview = (entry.text or "").replace("\n", " ").strip()
        if len(preview) > HISTORY_LABEL_CHARS:
            preview = preview[:HISTORY_LABEL_CHARS] + "…"
        return f"{dt.strftime('%m/%d %H:%M')}  {preview or '(空)'}"

    def _build_history_items(self) -> list:
        entries = self.history.list(limit=HISTORY_MENU_LIMIT)
        self._history_lookup = {}
        items: list = []
        if not entries:
            placeholder = rumps.MenuItem("(履歴はまだありません)")
            placeholder.set_callback(None)
            items.append(placeholder)
        else:
            for entry in reversed(entries):  # 新しいものから順に
                label = self._format_history_label(entry)
                # 同一秒に複数履歴があった場合のラベル衝突対策
                base = label
                suffix = 1
                while label in self._history_lookup:
                    suffix += 1
                    label = f"{base} ({suffix})"
                item = rumps.MenuItem(label, callback=self._on_history_item_click)
                items.append(item)
                self._history_lookup[label] = entry
        items.append(None)
        items.append(
            rumps.MenuItem(
                "Open history file", callback=self._on_open_history_file
            )
        )
        items.append(rumps.MenuItem("Clear history", callback=self._on_clear_history))
        return items

    def _refresh_history_menu(self) -> None:
        history_menu = self.menu.get("History")
        if history_menu is None:
            return

        def update():
            history_menu.clear()
            for item in self._build_history_items():
                if item is None:
                    history_menu.add(rumps.separator)
                else:
                    history_menu.add(item)
        _run_on_main_thread(update)

    def _on_history_item_click(self, sender) -> None:
        entry = self._history_lookup.get(sender.title)
        if entry is None or not entry.text:
            return
        copy_to_clipboard(entry.text)
        notify(
            "voiceinput",
            f"履歴をコピー: {entry.text[:60]}",
            enabled=self.config.notify,
        )
        self.logger.info("history copied (%d chars)", len(entry.text))

    def _on_open_history_file(self, _) -> None:
        path = self.history.path
        if not path.exists():
            notify("voiceinput", "履歴ファイルはまだありません", enabled=self.config.notify)
            return
        subprocess.Popen(["open", str(path)])

    def _on_clear_history(self, _) -> None:
        self.history.clear()
        self._refresh_history_menu()
        notify("voiceinput", "履歴をクリアしました", enabled=self.config.notify)
        self.logger.info("history cleared")

    # --- Vocabulary submenu (Phase E) ---

    def _compose_manual_vocab(self, gui_words: list[str]) -> list[str]:
        """config.yaml の manual + GUI 追加語を順序保持で重複排除して結合。

        config.yaml が先 (= 設定ファイル経由の意図を優先順位上位に)。
        ``app.__init__`` と ``_on_add_vocabulary_word`` / ``_on_remove`` の
        三箇所で同じロジックが必要なのでヘルパー化した。
        """
        seen: set[str] = set()
        out: list[str] = []
        for w in list(self.config.vocabulary_manual) + list(gui_words):
            w = (w or "").strip()
            if not w or w in seen:
                continue
            seen.add(w)
            out.append(w)
        return out

    def _gui_vocabulary(self) -> list[str]:
        """state.json に保存されている GUI 追加語を読み出す。"""
        try:
            return list(self.user_state_store.load().manual_vocabulary)
        except OSError:
            self.logger.exception("failed to load manual_vocabulary")
            return []

    def _build_vocabulary_items(self) -> list:
        """`Vocabulary` サブメニューの中身。

        - 現在の initial_prompt のプレビュー (placeholder)
        - "+ 単語を追加..." (rumps.Window で複数語入力)
        - "登録済み (N)" → 各語をクリックで削除
        - "Refresh now" でキャッシュ無視の即時再構築
        """
        if self.vocab_builder is None:
            return []
        items: list = []
        preview = self._initial_prompt or "(まだビルドされていません)"
        if len(preview) > 60:
            preview = preview[:60] + "…"
        info = rumps.MenuItem(f"現在: {preview}")
        info.set_callback(None)
        items.append(info)
        items.append(None)

        items.append(
            rumps.MenuItem(
                "+ 単語を追加…", callback=self._on_add_vocabulary_word
            )
        )
        items.append(
            rumps.MenuItem(
                "✎ 一覧をまとめて編集…", callback=self._on_edit_vocabulary_bulk
            )
        )
        gui_words = self._gui_vocabulary()
        if gui_words:
            sub_items = [
                rumps.MenuItem(
                    f"× {w}", callback=self._on_remove_vocabulary_word
                )
                for w in gui_words
            ]
            items.append((f"登録済み ({len(gui_words)})", sub_items))
        else:
            disabled = rumps.MenuItem("登録済み: (なし)")
            disabled.set_callback(None)
            items.append(disabled)

        items.append(None)
        items.append(
            rumps.MenuItem("Refresh now", callback=self._on_refresh_vocabulary)
        )
        return items

    def _refresh_vocabulary_menu(self) -> None:
        if self.vocab_builder is None:
            return
        voc_menu = self.menu.get("Vocabulary")
        if voc_menu is None:
            return

        def update():
            voc_menu.clear()
            for item in self._build_vocabulary_items():
                if item is None:
                    voc_menu.add(rumps.separator)
                else:
                    voc_menu.add(item)
        _run_on_main_thread(update)

    def _on_refresh_vocabulary(self, _) -> None:
        if self.vocab_builder is None:
            return
        self._rebuild_vocabulary_async(force=True, notify_user=True)

    def _prompt_text_via_osascript(
        self, *, title: str, message: str, default: str = ""
    ) -> str | None:
        """osascript の display dialog で 1 行テキストを受け取る。

        以前は rumps.Window を使っていたが、voiceinput は menu bar
        accessory アプリ (LSUIElement 相当) のため、Window がフォーカス
        を取れず "フリーズしたように見える" 事故が起きた。osascript の
        dialog は別プロセスで動くので確実に最前面・キーボード入力可能。

        キャンセル / エラー時は ``None`` を返す。空文字入力で OK を
        押した場合は ``""`` を返す (呼び出し側で扱う)。
        """
        def esc(s: str) -> str:
            return (
                s.replace("\\", "\\\\")
                .replace('"', '\\"')
                # 改行は AppleScript 文字列リテラル内に直接入れられない。
                # message 内では \n を半角スペースに畳み込む (情報は短めに)。
                .replace("\n", " ")
            )

        script = (
            f'display dialog "{esc(message)}" '
            f'default answer "{esc(default)}" '
            f'with title "{esc(title)}" '
            f'with icon note '
            f'buttons {{"キャンセル", "追加"}} '
            f'default button "追加" '
            f'cancel button "キャンセル"'
        )
        try:
            proc = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=600,  # 10 分でタイムアウト (放置防止)
            )
        except subprocess.SubprocessError:
            self.logger.exception("osascript dialog failed")
            return None
        if proc.returncode != 0:
            # キャンセル / dialog 拒否
            return None
        out = (proc.stdout or "").strip()
        # 形式例: "button returned:追加, text returned:voiceinput, Codex"
        if "text returned:" not in out:
            return None
        return out.split("text returned:", 1)[1].strip()

    def _confirm_via_osascript(
        self, *, title: str, message: str, ok_label: str
    ) -> bool:
        """osascript の display dialog でボタンのみの確認を取る。

        ``ok_label`` ボタンが押されたら True、キャンセル/エラーは False。
        テキスト入力は取らない (``_prompt_text_via_osascript`` との違い)。
        別プロセスのダイアログなので、待っている間もユーザーは TextEdit 等で
        編集を続けられる。
        """

        def esc(s: str) -> str:
            return (
                s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
            )

        script = (
            f'display dialog "{esc(message)}" '
            f'with title "{esc(title)}" '
            f"with icon note "
            f'buttons {{"キャンセル", "{esc(ok_label)}"}} '
            f'default button "{esc(ok_label)}" '
            f'cancel button "キャンセル"'
        )
        try:
            proc = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=600,  # 10 分でタイムアウト (_prompt_text_via_osascript と揃える)
            )
        except subprocess.SubprocessError:
            self.logger.exception("osascript confirm dialog failed")
            return False
        return proc.returncode == 0

    def _vocabulary_edit_path(self) -> Path:
        """「まとめて編集」で TextEdit に開く一時ファイルのパス。

        state.json と同じ ``~/Library/Application Support/voiceinput/`` 配下に
        置く。毎回現在の語彙で上書きしてから開くので、残置しても害はない。
        """
        return self.user_state_store.path.parent / "vocabulary_edit.txt"

    def _on_add_vocabulary_word(self, _) -> None:
        """osascript dialog で語を入力させて state.json に追加する。

        カンマ / 読点 / スペース / セミコロンで区切ると複数語まとめて
        登録できる。既存の語と重複したものはスキップ。state を更新したら
        VocabularyBuilder の manual を差し替えて即時 rebuild。
        """
        if self.vocab_builder is None:
            return
        text = self._prompt_text_via_osascript(
            title="カスタム語彙を追加",
            message=(
                "登録したい単語を入力してください。"
                "カンマ / 読点(、) / スペース / セミコロンで区切ると複数登録できます。"
            ),
            default="",
        )
        if text is None:
            return  # キャンセル
        raw = text.strip()
        if not raw:
            return
        new_words = [w.strip() for w in re.split(r"[,、;\s]+", raw) if w.strip()]
        existing = self._gui_vocabulary()
        added: list[str] = []
        for w in new_words:
            if w in existing:
                continue
            existing.append(w)
            added.append(w)
        if not added:
            notify(
                "voiceinput",
                "(すでに登録済みでした)",
                enabled=self.config.notify,
            )
            return
        try:
            self.user_state_store.update(manual_vocabulary=existing)
        except OSError:
            self.logger.exception("failed to persist manual_vocabulary")
            return
        # VocabularyBuilder の manual を差し替えて即時 rebuild
        self.vocab_builder.manual = self._compose_manual_vocab(existing)
        self._rebuild_vocabulary_async(force=True, notify_user=False)
        self._refresh_vocabulary_menu()
        notify(
            "voiceinput",
            f"語彙を追加: {', '.join(added)}",
            enabled=self.config.notify,
        )
        self.logger.info(
            "vocabulary added: %s (now %d gui words)", added, len(existing)
        )

    def _on_edit_vocabulary_bulk(self, _) -> None:
        """現在の語彙を TextEdit に 1 行 1 語で開き、保存後に全置換する。

        osascript の確認ダイアログ待ち + TextEdit 編集はユーザー操作待ちで長時間
        かかりうる。rumps コールバック (メインスレッド) でブロックすると録音・
        モード切替・メニュー操作がすべて固まるため、フロー全体をデーモンスレッド
        に逃がす。UI 更新は notify / ``_refresh_vocabulary_menu`` (内部で main
        thread に投げる) 経由なのでスレッド外から呼んでも安全。
        """
        if self.vocab_builder is None:
            return
        threading.Thread(
            target=self._edit_vocabulary_bulk_worker, daemon=True
        ).start()

    def _edit_vocabulary_bulk_worker(self) -> None:
        """`_on_edit_vocabulary_bulk` の本体 (デーモンスレッドで実行)。

        フロー:
        1. GUI 語彙を ``render_vocabulary_file`` でテキスト化して編集ファイルに
           書き出し、``open -e`` (TextEdit) で開く。
        2. 確認ダイアログを出す。ユーザーは TextEdit で編集・保存してから「反映」。
        3. 「反映」ならファイルを読み直し、``parse_vocabulary_lines`` で語リストに
           変換して state を全置換 → builder 差し替え → 再構築。
        キャンセル / 失敗時は語彙を変更しない。
        """
        if self.vocab_builder is None:
            return
        path = self._vocabulary_edit_path()
        try:
            path.write_text(
                render_vocabulary_file(self._gui_vocabulary()), encoding="utf-8"
            )
        except OSError:
            self.logger.exception("failed to write vocabulary edit file")
            notify(
                "voiceinput",
                "編集ファイルを作成できませんでした",
                enabled=self.config.notify,
            )
            return
        # TextEdit で開く (別プロセス)。
        try:
            subprocess.Popen(["open", "-e", str(path)])
        except OSError:
            self.logger.exception("failed to open vocabulary edit file")
            notify(
                "voiceinput",
                "エディタを開けませんでした",
                enabled=self.config.notify,
            )
            return
        # TextEdit が前面に立ち上がってから確認ダイアログを出す (初回起動時の
        # フォーカス競合で TextEdit がダイアログ背面に隠れるのを緩和)。
        time.sleep(0.5)
        confirmed = self._confirm_via_osascript(
            title="カスタム語彙をまとめて編集",
            message=(
                "TextEdit で 1 行に 1 語ずつ編集し、保存 (Cmd+S) してから"
                "「反映」を押してください。"
            ),
            ok_label="反映",
        )
        if not confirmed:
            notify(
                "voiceinput",
                "語彙の編集をキャンセルしました",
                enabled=self.config.notify,
            )
            return
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            self.logger.exception("failed to read vocabulary edit file")
            notify(
                "voiceinput",
                "編集ファイルを読み込めませんでした",
                enabled=self.config.notify,
            )
            return
        # 「まとめて編集」は full replace。ダイアログ待ちの間に + 単語を追加 /
        # 個別削除を行うと、その変更はこの全置換で上書きされる (編集中は他の
        # 語彙操作を避ける想定)。
        new_words = parse_vocabulary_lines(text)
        try:
            self.user_state_store.update(manual_vocabulary=new_words)
        except OSError:
            self.logger.exception("failed to persist manual_vocabulary")
            notify(
                "voiceinput",
                "語彙の保存に失敗しました",
                enabled=self.config.notify,
            )
            return
        # VocabularyBuilder の manual を差し替えて即時 rebuild
        self.vocab_builder.manual = self._compose_manual_vocab(new_words)
        self._rebuild_vocabulary_async(force=True, notify_user=False)
        self._refresh_vocabulary_menu()
        notify(
            "voiceinput",
            f"語彙を更新しました ({len(new_words)} 語)",
            enabled=self.config.notify,
        )
        self.logger.info(
            "vocabulary bulk-edited (now %d gui words)", len(new_words)
        )

    def _on_remove_vocabulary_word(self, sender) -> None:
        """登録済みサブメニューからクリックされた語を削除する。

        sender.title は "× <word>" 形式。先頭の "× " を取り除いて
        実体の語を取得し state から削除する。
        """
        if self.vocab_builder is None:
            return
        label = sender.title or ""
        # ラベルは "× <word>" 形式。lstrip だと "×" や空白を文字集合として
        # 剥がし、語頭が × の語を壊すので removeprefix で固定接頭辞だけ除く。
        word = label.removeprefix("× ").strip()
        if not word:
            return
        existing = self._gui_vocabulary()
        if word not in existing:
            return
        existing = [w for w in existing if w != word]
        try:
            self.user_state_store.update(manual_vocabulary=existing)
        except OSError:
            self.logger.exception("failed to persist manual_vocabulary")
            return
        self.vocab_builder.manual = self._compose_manual_vocab(existing)
        self._rebuild_vocabulary_async(force=True, notify_user=False)
        self._refresh_vocabulary_menu()
        notify(
            "voiceinput", f"語彙を削除: {word}", enabled=self.config.notify
        )
        self.logger.info(
            "vocabulary removed: %s (now %d gui words)", word, len(existing)
        )

    def _rebuild_vocabulary_async(
        self, *, force: bool = False, notify_user: bool = False
    ) -> None:
        """履歴 mtime チェック + 構築をバックグラウンドで行う。

        ``force=True`` のときは menu の "Refresh now" 経由なのでキャッシュ無視。
        ``notify_user=True`` のときはユーザーが意図して呼んだので、結果を
        通知センターに出してフィードバックする。
        """
        if self.vocab_builder is None:
            return

        def task() -> None:
            try:
                prev = self._initial_prompt
                value = self.vocab_builder.build(force=force)
                self._initial_prompt = value
                self._refresh_vocabulary_menu()
                if notify_user:
                    msg = (
                        f"語彙: {len(value)} 文字を反映しました"
                        if value
                        else "語彙: 履歴が少なく抽出できませんでした"
                    )
                    notify("voiceinput", msg, enabled=self.config.notify)
                if prev != value:
                    self.logger.info(
                        "vocabulary rebuilt (%d chars)", len(value)
                    )
            except Exception:
                self.logger.exception("vocabulary build failed")

        threading.Thread(target=task, daemon=True).start()

    def _maybe_refresh_vocabulary(self) -> None:
        """``_run_pipeline`` の最後から呼ばれる。N 件ごとに自動再構築。"""
        if self.vocab_builder is None:
            return
        self._pipeline_count += 1
        every = max(self.config.vocabulary_rebuild_every, 1)
        if self._pipeline_count % every == 0:
            self._rebuild_vocabulary_async(force=False)

    # --- menu callbacks ---

    def _on_select_mode(self, sender) -> None:
        self.format_mode = sender.title
        self._mark_format_mode(sender.title)
        try:
            self.user_state_store.update(format_mode=sender.title)
        except OSError:
            self.logger.exception("failed to persist format mode selection")
        self.logger.info("format mode -> %s", sender.title)

    def _on_toggle_auto_paste(self, sender) -> None:
        self.runtime.auto_paste = not self.runtime.auto_paste
        sender.state = self.runtime.auto_paste
        self.logger.info("auto_paste -> %s", self.runtime.auto_paste)

    def _on_toggle_logging(self, sender) -> None:
        self.runtime.logging_enabled = not self.runtime.logging_enabled
        sender.state = self.runtime.logging_enabled
        if not self.runtime.logging_enabled:
            # Logging OFF (機密配慮) に切り替えたら、録音中に取得済みの
            # 画面コンテキストも破棄して pipeline に渡らないようにする。
            self._invalidate_screen_ctx()
        self.logger = setup_logger(self.config.log_dir, self.runtime.logging_enabled)
        self.logger.info("logging -> %s", self.runtime.logging_enabled)

    def _on_toggle_screen_context(self, sender) -> None:
        """画面コンテキスト認識の ON/OFF。state.json に永続化する。"""
        self.screen_context_enabled = not self.screen_context_enabled
        sender.state = self.screen_context_enabled
        if not self.screen_context_enabled:
            # OFF にしたら in-flight のキャプチャ結果も破棄する。
            self._invalidate_screen_ctx()
        try:
            self.user_state_store.update(
                screen_context_enabled=self.screen_context_enabled
            )
        except OSError:
            self.logger.exception("failed to persist screen_context_enabled")
        notify(
            "voiceinput",
            "🖥️ 画面コンテキスト: " + ("ON" if self.screen_context_enabled else "OFF"),
            enabled=self.config.notify,
        )
        self.logger.info("screen_context -> %s", self.screen_context_enabled)

    # --- push-to-talk callbacks (combo hotkey only) ---

    def _on_recording_start(self) -> None:
        feedback.play(SOUND_START)
        try:
            self.recorder.start()
        except Exception as e:
            # state_machine は既に HELD に遷移済み。on_start が失敗したことを
            # 知らないので、ここで明示的に IDLE に戻さないと次の release で
            # on_stop が呼ばれて空 buffer の 0 samples ログが出てしまう。
            self.state_machine.force_idle()
            feedback.play(SOUND_ERROR)
            self._set_ui(TITLE_IDLE, "idle")
            self.logger.exception("recorder.start failed (push-to-talk)")
            notify(
                "voiceinput",
                f"録音開始に失敗しました: {e}",
                enabled=self.config.notify,
            )
            return
        self._capture_record_app()
        self._start_screen_capture()
        self._set_ui(TITLE_HELD, "recording (hold)")
        notify(
            "voiceinput",
            "🔴 録音中 (離すと送信 / 短押しで継続録音)",
            enabled=self.config.notify,
        )
        self.logger.info("recording started")

    def _on_recording_latch(self) -> None:
        feedback.play(SOUND_START)
        self._set_ui(TITLE_LATCHED, "recording (latched)")
        notify(
            "voiceinput",
            "🟠 継続録音中 (もう一度押すと停止)",
            enabled=self.config.notify,
        )
        self.logger.info("recording latched")

    def _on_recording_stop(self) -> None:
        feedback.play(SOUND_STOP)
        audio = self.recorder.stop()
        gen = self._capture_gen_for_stop()
        self._set_ui(TITLE_PROCESSING, "processing")
        self.logger.info("recording stopped (%d samples)", len(audio))
        threading.Thread(
            target=self._run_pipeline, args=(audio, gen), daemon=True
        ).start()

    # --- pipeline ---

    def _warmup(self) -> None:
        try:
            vad.warmup()
            self.logger.info("vad warmup done")
        except Exception:
            self.logger.exception("vad warmup failed")
        try:
            self.stt.warmup()
            self.logger.info("stt warmup done")
        except Exception:
            self.logger.exception("stt warmup failed")
        try:
            self.llm.warmup()
            self.logger.info("llm warmup done")
        except Exception:
            self.logger.exception("llm warmup failed")
        # 起動時に 1 回だけ vocabulary をビルドしておく。これ以降は
        # _maybe_refresh_vocabulary が N 件ごとに更新する。
        if self.vocab_builder is not None:
            try:
                self._initial_prompt = self.vocab_builder.build()
                self._refresh_vocabulary_menu()
                self.logger.info(
                    "vocabulary built (%d chars)", len(self._initial_prompt)
                )
            except Exception:
                self.logger.exception("vocabulary initial build failed")

    def _run_pipeline(self, audio: np.ndarray, ctx_gen: int | None = None) -> None:
        sample_rate = max(self.recorder.sample_rate, 1)
        audio_sec = len(audio) / sample_rate
        try:
            # 録音バッファのピーク振幅。0.0 ならマイク自体から音が来ていない
            # (デバイス OFF / ミュート / 別アプリが占有中 など)
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            t_vad0 = time.perf_counter()
            # Phase G: VAD 境界で ≤28s のチャンクに分割する。initial_prompt は
            # 最初の 30 秒窓にしか効かないため、チャンクごとに prompt を
            # 与えることで長文の後半にも語彙ヒントを効かせる。
            # 30 秒以内の発話 (大多数) はチャンク 1 個 = 従来と同一。
            chunks = vad.split_speech_chunks(audio, sample_rate=sample_rate)
            t_vad1 = time.perf_counter()
            trimmed_samples = sum(c.size for c in chunks)
            trimmed_sec = trimmed_samples / sample_rate
            if trimmed_samples == 0:
                # 全部無音 → STT/LLM スキップ
                self._silence_streak += 1
                self.logger.info(
                    "vad=%.2fs audio=%.2fs peak=%.4f trimmed=0.00s -> skipped (silence, streak=%d)",
                    t_vad1 - t_vad0,
                    audio_sec,
                    peak,
                    self._silence_streak,
                )
                if peak < 1e-4:
                    # 完全無音 → マイクが反応していない
                    notify(
                        "voiceinput",
                        "🎙️ マイクが無音です。デバイスやミュートを確認してください。",
                        enabled=self.config.notify,
                    )
                elif self._silence_streak >= SILENCE_WARNING_THRESHOLD:
                    notify(
                        "voiceinput",
                        f"⚠️ {self._silence_streak} 回連続で発話を検出できません。マイク設定や音量を確認してください。",
                        enabled=self.config.notify,
                    )
                else:
                    notify(
                        "voiceinput",
                        "(発話を検出できませんでした)",
                        enabled=self.config.notify,
                    )
                return
            # 発話を検出できたので silence streak をリセット
            self._silence_streak = 0

            # Phase F: この録音の世代に紐づく画面コンテキストだけを取り出す。
            # (取り出すと内部はクリアされ、次録音に持ち越さない。世代不一致 =
            # 既に次録音が始まっている場合は None を返し、取り違えを防ぐ。)
            screen_ctx = self._take_screen_ctx(ctx_gen)
            # app_mode: この録音のアプリ + (取れていれば) window_title から整形
            # モードを自動判定する。手動選択 (self.format_mode) は変えず、この
            # 録音だけ effective_mode を適用する。app_mode 無効時は手動モード。
            effective_mode, auto_mode_source = self._resolve_effective_mode(
                screen_ctx
            )
            screen_terms: list[str] = []
            llm_context = ""
            if screen_ctx is not None:
                screen_terms = screen_context.context_terms(screen_ctx)
                # LLM には「正しい表記の参考語」だけを渡す。周辺の文そのものは
                # 渡さない (モデルが他文を引用・挿入するのを防ぐ)。Whisper と
                # 同じ語リストを LLM 用の枠で切り詰めるだけ。
                llm_context = compose_initial_prompt(
                    "",
                    screen_terms,
                    max_chars=self.config.screen_context_llm_max_chars,
                )

            # E.3: STT 直前に initial_prompt を組み立てる。enabled=False なら ""。
            #
            # トークン枠の事実 (mlx_whisper 検証済み): initial_prompt は
            # **末尾 223 token が保持され、超過時は先頭が切られる**。また
            # 条件付けは末尾 (直近) ほど強い。→ 重要語 (画面語) は「末尾」に
            # 置く。旧実装は「先頭優先」で逆だった。
            capped_screen = compose_initial_prompt(
                "", screen_terms, max_chars=self.config.screen_context_terms_max_chars
            )
            if self.config.vocabulary_prompt_style == "terms":
                # rollback 用: 従来のスペース区切り (画面語が先頭)
                initial_prompt = compose_initial_prompt(
                    self._initial_prompt,
                    capped_screen.split(),
                    max_chars=self.config.vocabulary_max_chars,
                )
            else:
                # "list" (推奨): 「語、語、語。」句読点プライミング形式。
                # render_prompt_list は「重要度が低い順」を受け取り末尾優先で
                # 切り詰めるので、static (低頻度→高頻度→手動) を反転して前に、
                # 画面語を最後に置く。
                ordered_terms = list(reversed(self._initial_prompt.split()))
                ordered_terms.extend(capped_screen.split())
                initial_prompt = render_prompt_list(ordered_terms)

            t0 = time.perf_counter()
            text = self.stt.transcribe_chunks(
                chunks, initial_prompt=initial_prompt or None
            )
            t1 = time.perf_counter()
            # D.1: LLM 呼び出し前に cold/warm を snapshot。
            # FormatPipeline がスキップする入力 (raw / 空 / 短文) では LLM を
            # 触らないので "skip" として記録する。
            will_call_llm = self.format_pipeline.will_call_llm(text, effective_mode)
            cold_label = (
                ("cold" if self.llm.is_cold() else "warm") if will_call_llm else "skip"
            )
            formatted = self.format_pipeline.format(
                text, effective_mode, context=llm_context
            )
            # 決定的置換 (最終段)。固有名詞・商品名の誤変換を確定で直す。raw を
            # 含む全モードの出力に対し paste 直前で 1 回適用。空ルールなら no-op。
            formatted = apply_replacements(formatted, self.config.replacements)
            t2 = time.perf_counter()
            # Phase F: 画面コンテキストの利用状況をログ (生テキストは出さない)。
            if screen_ctx is not None:
                self.logger.info(
                    "screen_ctx app=%s trusted=%s terms=%d",
                    screen_ctx.app_name or "?",
                    screen_ctx.trusted,
                    len(screen_terms),
                )
            t_paste0 = time.perf_counter()
            copy_to_clipboard(formatted)
            if self.runtime.auto_paste and formatted:
                paste_active_app()
            t_paste1 = time.perf_counter()
            # 自動切替されたときは通知本文にモード名を付けてユーザーが検証できる
            # ようにする (例: "[mail_en] Hi John, ...")。
            notify_body = formatted[:80] or "(空)"
            if auto_mode_source:
                notify_body = f"[{effective_mode}] {notify_body}"
            notify("voiceinput", notify_body, enabled=self.config.notify)
            # ログの mode は実適用モード。自動切替時は根拠も併記する。
            mode_label = effective_mode
            if auto_mode_source:
                mode_label = f"{effective_mode}(auto:{auto_mode_source})"
            self.logger.info(
                "vad=%.2fs audio=%.2fs->%.2fs (%.0f%%) chunks=%d stt=%.2fs "
                "llm=%.2fs(%s) paste=%.2fs mode=%s len=%d vocab=%d",
                t_vad1 - t_vad0,
                audio_sec,
                trimmed_sec,
                100 * trimmed_sec / audio_sec if audio_sec > 0 else 0,
                len(chunks),
                t1 - t0,
                t2 - t1,
                cold_label,
                t_paste1 - t_paste0,
                mode_label,
                len(formatted),
                len(initial_prompt),
            )
            if self.runtime.logging_enabled and formatted:
                self.history.append(
                    make_entry(
                        mode=effective_mode,
                        text=formatted,
                        raw_text=text,
                        audio_sec=audio_sec,
                        stt_sec=t1 - t0,
                        llm_sec=t2 - t1,
                    )
                )
                self._refresh_history_menu()
                # E.3: 履歴が増えたので、N 回ごとに vocabulary を再構築する
                self._maybe_refresh_vocabulary()
        except Exception as e:
            feedback.play(SOUND_ERROR)
            self.logger.exception("pipeline failed")
            notify("voiceinput", f"エラー: {e}", enabled=self.config.notify)
        finally:
            self._set_ui(TITLE_IDLE, "idle")


def main() -> None:
    cfg = load_config()
    app = VoiceInputApp(cfg)
    app.hotkey.start()
    app.run()
