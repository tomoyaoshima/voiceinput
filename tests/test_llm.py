import time
from pathlib import Path
from typing import Any

import httpx

from voiceinput.llm import (
    FormatPipeline,
    OllamaClient,
    SHORT_TEXT_SKIP_THRESHOLD,
    _parse_keep_alive,
)


class StubClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, int | None]] = []

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        num_predict_override: int | None = None,
    ) -> str:
        self.calls.append((prompt, system, num_predict_override))
        return "整形結果"


class StubReturning:
    """戻り値を指定できる stub。段落保持/畳み込みの検証用。"""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str | None, int | None]] = []

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        num_predict_override: int | None = None,
    ) -> str:
        self.calls.append((prompt, system, num_predict_override))
        return self.response


def test_raw_mode_returns_text_unchanged(tmp_path: Path):
    pipe = FormatPipeline(StubClient(), tmp_path)
    assert pipe.format("元のテキスト", "raw") == "元のテキスト"


def test_clean_mode_calls_llm_with_template(tmp_path: Path):
    (tmp_path / "clean.yaml").write_text(
        'name: clean\nsystem: "S"\nuser_template: "U {text}"\n'
    )
    client = StubClient()
    pipe = FormatPipeline(client, tmp_path)
    out = pipe.format("こんにちは皆さん", "clean")
    assert out == "整形結果"
    # clean mode は num_predict_override=None (OllamaClient のデフォルトを使う)
    assert client.calls == [("U こんにちは皆さん", "S", None)]


def test_clean_mode_without_context_is_unchanged(tmp_path: Path):
    """context 省略時は従来と完全に同一の user プロンプト (回帰防止)。"""
    (tmp_path / "clean.yaml").write_text(
        'name: clean\nsystem: "S"\nuser_template: "U {text}"\n'
    )
    client = StubClient()
    pipe = FormatPipeline(client, tmp_path)
    pipe.format("こんにちは皆さん", "clean")
    assert client.calls == [("U こんにちは皆さん", "S", None)]


def test_context_is_prepended_to_user_prompt(tmp_path: Path):
    """context 非空時、参考表記ブロックが本文の前に付き、本文も保持される。"""
    (tmp_path / "clean.yaml").write_text(
        'name: clean\nsystem: "S"\nuser_template: "U {text}"\n'
    )
    client = StubClient()
    pipe = FormatPipeline(client, tmp_path)
    pipe.format("こんにちは皆さん", "clean", context="山田 案件X")
    prompt = client.calls[0][0]
    assert "参考表記" in prompt
    assert "山田" in prompt
    assert "---" in prompt
    assert "U こんにちは皆さん" in prompt
    # 本文(---より後)が末尾に来る
    assert prompt.strip().endswith("U こんにちは皆さん")
    # 「挿入・引用してはならない」旨の縛りが入る
    assert "挿入" in prompt or "引用" in prompt


def test_empty_context_does_not_add_block(tmp_path: Path):
    (tmp_path / "clean.yaml").write_text(
        'name: clean\nsystem: "S"\nuser_template: "U {text}"\n'
    )
    client = StubClient()
    pipe = FormatPipeline(client, tmp_path)
    pipe.format("こんにちは皆さん", "clean", context="")
    assert "参考表記" not in client.calls[0][0]


def test_mail_mode_overrides_num_predict(tmp_path: Path):
    """mail mode は本文補強で長くなるため num_predict を 1024 に拡大する。"""
    (tmp_path / "mail.yaml").write_text(
        'name: mail\nsystem: "S"\nuser_template: "U {text}"\n'
    )
    client = StubClient()
    pipe = FormatPipeline(client, tmp_path)
    out = pipe.format("お世話になっております", "mail")
    assert out == "整形結果"
    # mail mode のときだけ override される
    assert client.calls == [("U お世話になっております", "S", 1024)]


def test_mail_en_overrides_num_predict(tmp_path: Path):
    """mail_en mode も翻訳本文で長くなるため num_predict を 1024 に拡大する。"""
    (tmp_path / "mail_en.yaml").write_text(
        'name: mail_en\nsystem: "S"\nuser_template: "U {text}"\n'
    )
    client = StubClient()
    pipe = FormatPipeline(client, tmp_path)
    out = pipe.format("お世話になっております", "mail_en")
    assert out == "整形結果"
    assert client.calls == [("U お世話になっております", "S", 1024)]


def test_mail_en_preserves_multiple_paragraphs(tmp_path: Path):
    """mail_en は複数段落の英文メール本文を保持する (最後の段落に畳まない)。"""
    (tmp_path / "mail_en.yaml").write_text(
        'name: mail_en\nsystem: "S"\nuser_template: "{text}"\n'
    )
    client = StubReturning("Hi John,\n\nYour order has shipped.\n\nBest regards,")
    pipe = FormatPipeline(client, tmp_path)
    out = pipe.format("十分な長さの入力テキストです", "mail_en")
    assert out == "Hi John,\n\nYour order has shipped.\n\nBest regards,"


def test_mail_preserves_multiple_paragraphs(tmp_path: Path):
    """mail (日本語) も複数段落本文を保持する (バグ修正の回帰防止)。"""
    (tmp_path / "mail.yaml").write_text(
        'name: mail\nsystem: "S"\nuser_template: "{text}"\n'
    )
    client = StubReturning("お世話になっております。\n\n本文です。\n\nよろしくお願いいたします。")
    pipe = FormatPipeline(client, tmp_path)
    out = pipe.format("十分な長さの入力テキストです", "mail")
    assert out == "お世話になっております。\n\n本文です。\n\nよろしくお願いいたします。"


def test_clean_collapses_multiple_paragraphs(tmp_path: Path):
    """clean は従来どおり複数段落を最後の段落に畳む (回帰防止)。"""
    (tmp_path / "clean.yaml").write_text(
        'name: clean\nsystem: "S"\nuser_template: "{text}"\n'
    )
    client = StubReturning("誤った文。\n\n（ただし、修正します：）\n\n正しい文。")
    pipe = FormatPipeline(client, tmp_path)
    out = pipe.format("十分な長さの入力テキストです", "clean")
    assert out == "正しい文。"


def test_available_modes_includes_raw_and_loaded_prompts(tmp_path: Path):
    (tmp_path / "clean.yaml").write_text(
        'name: clean\nsystem: ""\nuser_template: "{text}"\n'
    )
    (tmp_path / "mail.yaml").write_text(
        'name: mail\nsystem: ""\nuser_template: "{text}"\n'
    )
    pipe = FormatPipeline(StubClient(), tmp_path)
    assert pipe.available_modes() == ["raw", "clean", "mail"]


def test_unknown_mode_raises(tmp_path: Path):
    pipe = FormatPipeline(StubClient(), tmp_path)
    try:
        pipe.format("十分な長さの入力テキストです", "nonexistent")
    except ValueError as e:
        assert "nonexistent" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_empty_text_short_circuits_to_empty_without_calling_llm(tmp_path: Path):
    (tmp_path / "clean.yaml").write_text(
        'name: clean\nsystem: ""\nuser_template: "{text}"\n'
    )
    client = StubClient()
    pipe = FormatPipeline(client, tmp_path)
    assert pipe.format("", "clean") == ""
    assert client.calls == []


def test_short_text_under_threshold_skips_llm(tmp_path: Path):
    """5 文字以下の極短入力は LLM を呼ばずに raw を返す (D.5)。"""
    (tmp_path / "clean.yaml").write_text(
        'name: clean\nsystem: ""\nuser_template: "{text}"\n'
    )
    client = StubClient()
    pipe = FormatPipeline(client, tmp_path)
    # 閾値ピッタリは skip 対象 (≤ 5)
    short = "あ" * SHORT_TEXT_SKIP_THRESHOLD
    assert pipe.format(short, "clean") == short
    assert client.calls == []
    # 1 文字超えると LLM が呼ばれる
    long = "あ" * (SHORT_TEXT_SKIP_THRESHOLD + 1)
    pipe.format(long, "clean")
    assert len(client.calls) == 1


def test_will_call_llm_predicate(tmp_path: Path):
    """app.py が cold/warm を測るかの判断に使う述語。"""
    (tmp_path / "clean.yaml").write_text(
        'name: clean\nsystem: ""\nuser_template: "{text}"\n'
    )
    pipe = FormatPipeline(StubClient(), tmp_path)
    assert pipe.will_call_llm("これは十分長い文章です", "clean") is True
    assert pipe.will_call_llm("", "clean") is False
    assert pipe.will_call_llm("短い", "clean") is False
    assert pipe.will_call_llm("十分な長さの入力テキスト", "raw") is False


def test_ollama_client_set_model_changes_active_model():
    client = OllamaClient(
        "http://localhost:11434", "qwen2.5:14b", timeout=30.0, temperature=0.2
    )
    assert client.model == "qwen2.5:14b"
    client.set_model("qwen2.5:7b")
    assert client.model == "qwen2.5:7b"


def test_ollama_client_list_models_returns_sorted_names(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "models": [
                    {"name": "qwen2.5:14b"},
                    {"name": "phi4:latest"},
                    {"name": "qwen2.5:7b"},
                    {"name": ""},  # 空名前は除外
                    {},  # name 欠落も除外
                ]
            }

    monkeypatch.setattr(httpx, "get", lambda url, timeout=None: FakeResponse())
    client = OllamaClient(
        "http://localhost:11434", "qwen2.5:14b", timeout=30.0, temperature=0.2
    )
    models = client.list_models()
    assert models == ["phi4:latest", "qwen2.5:14b", "qwen2.5:7b"]


def test_ollama_client_list_models_swallows_errors(monkeypatch):
    def raise_(url, timeout=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(httpx, "get", raise_)
    client = OllamaClient(
        "http://localhost:11434", "qwen2.5:14b", timeout=30.0, temperature=0.2
    )
    assert client.list_models() == []


def test_ollama_client_sends_temperature_option(monkeypatch):
    """temperature / num_predict / num_ctx が options に、keep_alive がトップレベルに乗る"""
    captured: dict[str, Any] = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "ok"}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)
    client = OllamaClient(
        "http://localhost:11434",
        "qwen2.5:14b",
        timeout=30.0,
        temperature=0.15,
        num_predict=512,
        num_ctx=1024,
        keep_alive="1h",
    )
    out = client.generate("hello", system="be brief")
    assert out == "ok"
    assert captured["json"]["options"] == {
        "temperature": 0.15,
        "num_predict": 512,
        "num_ctx": 1024,
    }
    assert captured["json"]["keep_alive"] == "1h"
    assert captured["json"]["system"] == "be brief"
    assert captured["json"]["model"] == "qwen2.5:14b"
    assert captured["url"].endswith("/api/generate")
    # voiceinput は整形タスクで reasoning 不要。qwen3 / qwq 系の思考時間
    # (測定: qwen3:8b で 30.82s) を回避するために think=false を明示送信する。
    assert captured["json"]["think"] is False


def test_ollama_client_num_predict_override(monkeypatch):
    """generate(num_predict_override=...) は options.num_predict を一時的に置き換える"""
    captured: dict[str, Any] = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "ok"}

    def fake_post(url, json=None, timeout=None):
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)
    client = OllamaClient(
        "http://localhost:11434",
        "qwen2.5:14b",
        timeout=30.0,
        num_predict=512,
        num_ctx=1024,
    )
    client.generate("hi", num_predict_override=2048)
    assert captured["json"]["options"]["num_predict"] == 2048


def test_parse_keep_alive_units():
    """30s, 30m, 1h, 24h, -1, 数値 ベタ書きのいずれも秒に変換できる"""
    assert _parse_keep_alive("30s") == 30
    assert _parse_keep_alive("30m") == 1800
    assert _parse_keep_alive("1h") == 3600
    assert _parse_keep_alive("24h") == 86400
    assert _parse_keep_alive("-1") == float("inf")
    assert _parse_keep_alive(-1) == float("inf")
    assert _parse_keep_alive("60") == 60
    # 解析できないものは 0 (= 即 cold)
    assert _parse_keep_alive("garbage") == 0


def test_ollama_client_is_cold_initially_then_warm():
    client = OllamaClient(
        "http://localhost:11434",
        "qwen2.5:14b",
        timeout=30.0,
        keep_alive="1h",
    )
    # まだ一度も呼んでいない → cold
    assert client.is_cold() is True
    # _last_call_at を直接立ててウォーム挙動を確認 (HTTP は叩かない)
    client._last_call_at = time.monotonic()
    assert client.is_cold() is False


def test_ollama_client_set_model_resets_to_cold():
    client = OllamaClient(
        "http://localhost:11434",
        "qwen2.5:7b",
        timeout=30.0,
        keep_alive="1h",
    )
    client._last_call_at = time.monotonic()  # warm 状態にする
    assert client.is_cold() is False
    client.set_model("qwen2.5:14b")
    # モデル切替で cold に戻る
    assert client.is_cold() is True
