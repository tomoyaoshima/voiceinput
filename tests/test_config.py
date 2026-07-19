from pathlib import Path

import yaml

from voiceinput.config import Config, default_config, load_config


def test_default_config_has_required_fields():
    cfg = default_config()
    assert isinstance(cfg, Config)
    assert cfg.hotkey == "<cmd_r>"
    # 体感速度を優先して 7b デフォルト。14b は menu bar から都度切替で永続化される
    # gemma4:12b は実運用比較 (qwen2.5:7b/14b, qwen3:8b, gemma4:e4b/26b) の
    # 中で速度と精度のバランスが最良だったため default に採用。
    # より軽く速くしたい場合は gemma4:e4b。
    assert cfg.ollama_model == "gemma4:12b"
    assert cfg.default_format_mode == "clean"
    assert cfg.whisper_language == "ja"
    assert cfg.auto_paste is True
    assert cfg.notify is True
    assert cfg.logging_enabled is True
    assert cfg.ollama_temperature == 0.2
    # 元 2048 → 短縮していたが、gemma4:26b など内部 reasoning する系は
    # 本文出力前に 700+ token 消費するため 1024 まで引き戻した。
    # qwen2.5 のような fast model は stop で早期終了するので、上限を
    # 1024 にしてもレイテンシには影響しない。
    assert cfg.ollama_num_predict == 1024
    assert cfg.ollama_num_ctx == 1024
    # menu bar 常駐アプリと整合性を取る
    assert cfg.ollama_keep_alive == "24h"
    # Phase E: vocabulary defaults
    assert cfg.vocabulary_enabled is True
    assert cfg.vocabulary_manual == ()
    assert cfg.vocabulary_history_size == 100
    assert cfg.vocabulary_top_n == 30
    assert cfg.vocabulary_rebuild_every == 10
    assert cfg.vocabulary_max_chars == 200
    # Phase G: prompt 形式は句読点プライミングの "list" がデフォルト
    assert cfg.vocabulary_prompt_style == "list"
    # Phase F: screen context defaults
    assert cfg.screen_context_enabled is True
    assert cfg.screen_context_terms_max_chars == 120
    assert cfg.screen_context_llm_max_chars == 120
    assert cfg.screen_context_read_focused_value is True
    assert cfg.screen_context_denylist_bundles == ()
    # 決定的置換 (既定は無効)
    assert cfg.replacements == ()
    # アプリ別モード自動切替 (既定は無効)
    assert cfg.app_mode_enabled is False
    assert cfg.app_mode_rules == ()


def test_load_config_reads_screen_context_section(tmp_path: Path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "screen_context": {
                    "enabled": False,
                    "terms_max_chars": 80,
                    "llm_max_chars": 200,
                    "read_focused_value": False,
                    "denylist_bundles": ["com.foo.bar", "com.baz.qux"],
                }
            }
        )
    )
    cfg = load_config(yaml_path)
    assert cfg.screen_context_enabled is False
    assert cfg.screen_context_terms_max_chars == 80
    assert cfg.screen_context_llm_max_chars == 200
    assert cfg.screen_context_read_focused_value is False
    assert cfg.screen_context_denylist_bundles == ("com.foo.bar", "com.baz.qux")


def test_load_config_reads_vocabulary_section(tmp_path: Path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "vocabulary": {
                    "enabled": False,
                    "manual": ["voiceinput", "Codex"],
                    "history_size": 50,
                    "top_n": 10,
                    "rebuild_every": 5,
                    "max_chars": 120,
                    "prompt_style": "terms",
                }
            }
        )
    )
    cfg = load_config(yaml_path)
    assert cfg.vocabulary_enabled is False
    assert cfg.vocabulary_manual == ("voiceinput", "Codex")
    assert cfg.vocabulary_history_size == 50
    assert cfg.vocabulary_top_n == 10
    assert cfg.vocabulary_rebuild_every == 5
    assert cfg.vocabulary_max_chars == 120
    assert cfg.vocabulary_prompt_style == "terms"


def test_load_config_reads_replacements(tmp_path: Path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "replacements": [
                    {"from": "ペイパル", "to": "PayPal"},
                    {"from": "さんぷる", "to": "サンプル商会"},
                ]
            },
            allow_unicode=True,
        )
    )
    cfg = load_config(yaml_path)
    assert cfg.replacements == (("ペイパル", "PayPal"), ("さんぷる", "サンプル商会"))


def test_load_config_replacements_skips_malformed(tmp_path: Path):
    """from 欠落・非 dict 要素は読み飛ばし、to 欠落は空文字 (削除) 扱い。"""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "replacements": [
                    {"from": "良い", "to": "OK"},
                    {"to": "from無し"},       # from 欠落 → 捨てる
                    "壊れた要素",              # 非 dict → 捨てる
                    {"from": "あー"},          # to 欠落 → 空文字 (削除)
                ]
            },
            allow_unicode=True,
        )
    )
    cfg = load_config(yaml_path)
    assert cfg.replacements == (("良い", "OK"), ("あー", ""))


def test_load_config_reads_app_mode(tmp_path: Path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "app_mode": {
                    "enabled": True,
                    "rules": [
                        {"match_bundle": "com.apple.mail", "mode": "mail"},
                        {"match_title": "Gmail", "mode": "mail_en"},
                    ],
                }
            },
            allow_unicode=True,
        )
    )
    cfg = load_config(yaml_path)
    assert cfg.app_mode_enabled is True
    assert cfg.app_mode_rules == (
        ("bundle", "com.apple.mail", "mail"),
        ("title", "Gmail", "mail_en"),
    )


def test_load_config_app_mode_skips_malformed(tmp_path: Path):
    """mode 欠落 / match パターン欠落の要素は読み飛ばす。"""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "app_mode": {
                    "rules": [
                        {"match_bundle": "com.apple.mail", "mode": "mail"},
                        {"match_bundle": "com.x.y"},          # mode 欠落 → 捨てる
                        {"mode": "clean"},                     # match 欠落 → 捨てる
                        "壊れた要素",                           # 非 dict → 捨てる
                    ]
                }
            },
            allow_unicode=True,
        )
    )
    cfg = load_config(yaml_path)
    # enabled 未指定なので default False のまま
    assert cfg.app_mode_enabled is False
    assert cfg.app_mode_rules == (("bundle", "com.apple.mail", "mail"),)


def test_load_config_overrides_defaults(tmp_path: Path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "ollama": {"model": "phi4:latest", "temperature": 0.05},
                "auto_paste": False,
            }
        )
    )
    cfg = load_config(yaml_path)
    assert cfg.ollama_model == "phi4:latest"
    assert cfg.ollama_temperature == 0.05
    assert cfg.auto_paste is False
    # 上書きされてないフィールドは default のまま
    assert cfg.hotkey == "<cmd_r>"
    assert cfg.whisper_language == "ja"


def test_load_config_returns_defaults_when_path_missing(tmp_path: Path):
    cfg = load_config(tmp_path / "does_not_exist.yaml")
    assert cfg == default_config()


def test_load_config_expands_user_in_log_dir(tmp_path: Path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(yaml.safe_dump({"log_dir": "~/Library/Logs/voiceinput-test"}))
    cfg = load_config(yaml_path)
    assert "~" not in str(cfg.log_dir)
    assert str(cfg.log_dir).endswith("voiceinput-test")
