import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx
import yaml


# 5 文字以下は LLM をかけても整形効果が薄く、逆に「補完しすぎ」事故が起きやすい。
# raw として直接出力する閾値。
SHORT_TEXT_SKIP_THRESHOLD = 5

# mode ごとの num_predict 上書き。指定がない mode は OllamaClient のデフォルトを使う。
# mail / mail_en mode は本文補強・翻訳で長くなりがちなので大きめに固定。
NUM_PREDICT_BY_MODE: dict[str, int] = {
    "mail": 1024,
    "mail_en": 1024,
}

# 出力が複数段落になりうる mode。これらは strip_llm_meta_commentary の
# 「複数段落 → 最後の 1 段落だけ採用」ロジックを無効化し、本文の段落構造を
# 保持する (メール本文が結びの 1 段落に削られる事故を防ぐ)。clean / raw 系は
# 1 ターンの短い整形なので従来どおり最後の段落を採用する。
MULTI_PARAGRAPH_MODES: frozenset[str] = frozenset({"mail", "mail_en"})


def _parse_keep_alive(spec: str | int | float) -> float:
    """Ollama の keep_alive 表記を秒数に変換する。

    "30m" -> 1800, "1h" -> 3600, "24h" -> 86400, "30s" -> 30,
    "-1" や負値は無期限 (= float('inf'))。
    数値だけなら秒として解釈。
    """
    if isinstance(spec, (int, float)):
        return float("inf") if spec < 0 else float(spec)
    s = str(spec).strip().lower()
    if not s:
        return 0.0
    if s in ("-1", "infinite", "inf"):
        return float("inf")
    try:
        if s.endswith("ms"):
            return float(s[:-2]) / 1000.0
        if s.endswith("s"):
            return float(s[:-1])
        if s.endswith("m"):
            return float(s[:-1]) * 60.0
        if s.endswith("h"):
            return float(s[:-1]) * 3600.0
        return float(s)
    except ValueError:
        return 0.0


class GenerateClient(Protocol):
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        num_predict_override: int | None = None,
    ) -> str: ...


class OllamaClient:
    def __init__(
        self,
        endpoint: str,
        model: str,
        timeout: float,
        temperature: float = 0.2,
        num_predict: int = 512,
        num_ctx: int = 1024,
        keep_alive: str = "24h",
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.num_predict = num_predict
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self._keep_alive_seconds = _parse_keep_alive(keep_alive)
        # 直近 generate() が成功した時刻 (monotonic)。Ollama 側でモデルが
        # アンロード済みかを推測するのに使う。
        self._last_call_at: float | None = None

    def is_cold(self) -> bool:
        """次の generate() がコールドスタートになる見込みかを返す。

        - まだ一度も呼ばれていない、または
        - 前回呼び出しから keep_alive を超えて経過している
        ならコールド扱い。warmup() も generate() を呼ぶので _last_call_at を
        更新する → ウォーム扱いになる。
        """
        if self._last_call_at is None:
            return True
        if self._keep_alive_seconds == float("inf"):
            return False
        return (time.monotonic() - self._last_call_at) > self._keep_alive_seconds

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        num_predict_override: int | None = None,
    ) -> str:
        num_predict = (
            num_predict_override
            if num_predict_override is not None
            else self.num_predict
        )
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            # qwen3 / qwq など "thinking model" は default で chain-of-thought
            # を回し、voiceinput の単純な整形タスクでも 30+ 秒かかる事故が
            # 起きる (測定済み: qwen3:8b で 0.36s → 30.82s と 85 倍差)。
            # 整形は reasoning 不要なので think を明示的に off にする。
            # 非対応モデル (qwen2.5 / gemma4 / phi4 など) は this field を
            # 単純に無視するため副作用なし。
            "think": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": num_predict,
                "num_ctx": self.num_ctx,
            },
        }
        if system:
            payload["system"] = system
        r = httpx.post(
            f"{self.endpoint}/api/generate", json=payload, timeout=self.timeout
        )
        r.raise_for_status()
        result = r.json().get("response", "").strip()
        # 成功時のみ更新。失敗 (HTTPStatusError 等) は cold のまま放置することで
        # 次回もコールド扱いにし、モニタリングしやすくする。
        self._last_call_at = time.monotonic()
        return result

    def warmup(self) -> None:
        try:
            self.generate("ok", system="一文字だけ返してください")
        except Exception:
            pass

    def set_model(self, name: str) -> None:
        self.model = name
        # 別モデルへ切り替えた直後は Ollama 側でロードし直しになるので
        # コールド扱いに戻す
        self._last_call_at = None

    def list_models(self) -> list[str]:
        """Ollama に現在 pull されているモデル名を返す。失敗時は空リスト。"""
        try:
            r = httpx.get(f"{self.endpoint}/api/tags", timeout=5.0)
            r.raise_for_status()
            data = r.json()
            return sorted({m["name"] for m in data.get("models", []) if m.get("name")})
        except Exception:
            return []


@dataclass
class FormatPrompt:
    name: str
    system: str
    user_template: str


class FormatPipeline:
    def __init__(self, client: GenerateClient, prompts_dir: Path) -> None:
        self.client = client
        self.prompts: dict[str, FormatPrompt] = {}
        if prompts_dir.exists():
            for p in sorted(prompts_dir.glob("*.yaml")):
                data = yaml.safe_load(p.read_text())
                self.prompts[data["name"]] = FormatPrompt(**data)

    def available_modes(self) -> list[str]:
        return ["raw", *self.prompts.keys()]

    def will_call_llm(self, text: str, mode: str) -> bool:
        """この入力で実際に LLM を呼ぶ予定か。

        raw mode、空文字、5 文字以下の極短入力は LLM を呼ばずに raw 返却。
        この述語は app.py が「cold/llm を測るべきか」を判断するのに使う。
        """
        if mode == "raw":
            return False
        if not text or len(text.strip()) <= SHORT_TEXT_SKIP_THRESHOLD:
            return False
        return mode in self.prompts

    def format(self, text: str, mode: str, *, context: str = "") -> str:
        if not self.will_call_llm(text, mode):
            # raw / 空 / 短すぎ は LLM をスキップして raw を返す
            if mode != "raw" and mode not in self.prompts and text:
                # 未知の mode は明示エラー (呼び出し側のバグ)
                raise ValueError(f"unknown format mode: {mode}")
            return text
        p = self.prompts[mode]
        num_predict = NUM_PREDICT_BY_MODE.get(mode)
        user_prompt = p.user_template.format(text=text)
        # Phase F: 画面コンテキストの「表記」参考語。非空時のみ user プロンプト
        # 先頭に付与。空なら従来と完全に同一 (既存挙動・テストを壊さない)。
        # 渡すのは語のリストだけ (周辺の文は渡さない)。発話に同じ語が出た時の
        # 漢字・カタカナ・英字の表記を整えるためだけに使い、語の挿入・引用は
        # 禁止する、と強く縛る。システム側でも二重に縛る。
        if context:
            user_prompt = (
                "[参考表記: 以下は画面にあった語の正しい表記。発話文に同じ語が"
                "出てきたときの漢字・カタカナ・英字の表記を整えるためだけに使う。"
                "これらの語を新しく挿入・引用してはならない。発話に無い語は使わない]\n"
                f"{context}\n"
                "---\n"
                f"{user_prompt}"
            )
        raw_out = self.client.generate(
            user_prompt,
            system=p.system,
            num_predict_override=num_predict,
        )
        # LLM の meta-commentary 防衛層 (clean / mail / mail_en 共通)。
        # プロンプトでも禁じているが 100% は守られない (qwen2.5:14b で
        # 観測済み)。ここで括弧の判断理由・代替案を strip する。
        # mail / mail_en は本文が複数段落になりうるので段落の畳み込みは無効化し、
        # メタ括弧の除去だけを効かせる。
        from voiceinput.text_filter import strip_llm_meta_commentary

        collapse = mode not in MULTI_PARAGRAPH_MODES
        return strip_llm_meta_commentary(raw_out, collapse_paragraphs=collapse)
