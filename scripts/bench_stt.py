#!/usr/bin/env python
"""STT モデル比較ベンチ: CER (文字誤り率) と latency を計測する。

使い方:
    # TTS (say/Kyoko) 合成音声でモデル比較
    uv run python scripts/bench_stt.py

    # モデルを絞る / prompt 条件を絞る
    uv run python scripts/bench_stt.py --models turbo,large-v3 --prompts none,list

    # 実発話セット (推奨: TTS と両方で優位なモデルだけ採用する)
    #   ~/.cache/voiceinput/bench/real/ に NNN.wav (16kHz mono) と
    #   NNN.txt (正解文) のペアを置いて:
    uv run python scripts/bench_stt.py --audio-dir ~/.cache/voiceinput/bench/real

注意:
- TTS 音声は人間の発話と音響特性が違うため、絶対値ではなく相対比較用。
  さらに TTS は固有名詞を読み間違えることがあるので、fixture 文は
  「Kyoko が正しく読める語」だけで構成している (新しい文を足す時は
  一度 `say -v Kyoko "文"` を耳で確認してから)。
- 初回はモデルのダウンロード (数百 MB〜数 GB) が走る。
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voiceinput import vad  # noqa: E402
from voiceinput.metrics import cer  # noqa: E402
from voiceinput.stt import WhisperSTT  # noqa: E402
from voiceinput.vocabulary import render_prompt_list  # noqa: E402

CACHE_DIR = Path("~/.cache/voiceinput/bench").expanduser()

# モデル短縮名 → HF repo
MODELS = {
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "turbo-q4": "mlx-community/whisper-large-v3-turbo-q4",
    "kotoba": "kaiinui/kotoba-whisper-v2.0-mlx",
}

# ベンチ用語彙 (fixture 文に含まれる語 + アプリの典型語彙)。
# prompt 条件 "list"/"terms" でこれを渡す。
BENCH_TERMS = [
    "ボイスインプット",
    "クロードコード",
    "オラマ",
    "ジェンマ",
    "アクセシビリティ",
    "スクリーンショット",
    "リファクタリング",
    "デプロイ",
    "マージ",
    "レビュー",
]

# fixture: (id, 話速, テキスト)。proper=True は固有名詞・カタカナ語サブセット。
# Kyoko の読みが安定する語のみ (漢字の固有名詞は読み違いリスクがあるので
# カタカナ語中心。新規追加時は say -v Kyoko で耳確認すること)。
FIXTURES = [
    ("f01", 180, "明日の会議は10時から始まりますので、資料の準備をお願いします。", False),
    ("f02", 180, "先ほどのメールに返信しましたので、内容をご確認ください。", False),
    ("f03", 220, "この件については、来週の月曜日までに回答をいただけると助かります。", False),
    ("f04", 180, "音声入力の精度を上げるために、いくつかの改善を試しています。", False),
    ("f05", 220, "録音した内容は自動的に文字起こしされて、クリップボードに入ります。", False),
    ("f06", 180, "ボイスインプットのリファクタリングをクロードコードに依頼しました。", True),
    ("f07", 180, "オラマでジェンマのモデルをダウンロードして切り替えてください。", True),
    ("f08", 220, "アクセシビリティの設定からスクリーンショットの権限を確認します。", True),
    ("f09", 180, "デプロイの前にレビューとマージをお願いしたいです。", True),
    ("f10", 220, "カタカナのブランドネームはディクテーションで間違えやすいです。", True),
]


def synthesize_fixtures() -> list[tuple[str, str, bool, Path]]:
    """say (Kyoko) で fixture 音声を合成してキャッシュし、一覧を返す。

    キャッシュキーに text のハッシュを含める (fixture 文を書き換えた時に
    古い音声で CER を測ってしまう事故を防ぐ)。
    """
    import hashlib

    tts_dir = CACHE_DIR / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for fid, rate, text, proper in FIXTURES:
        digest = hashlib.sha1(text.encode()).hexdigest()[:8]
        wav_path = tts_dir / f"{fid}_r{rate}_{digest}.wav"
        if not wav_path.exists():
            subprocess.run(
                [
                    "say",
                    "-v",
                    "Kyoko",
                    "-r",
                    str(rate),
                    "-o",
                    str(wav_path),
                    "--data-format=LEI16@16000",
                    text,
                ],
                check=True,
            )
        out.append((fid, text, proper, wav_path))
    return out


def load_real_set(audio_dir: Path) -> list[tuple[str, str, bool, Path]]:
    """実発話セット (NNN.wav + NNN.txt) を読み込む。"""
    out = []
    for wav_path in sorted(audio_dir.glob("*.wav")):
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            print(f"  ! {wav_path.name}: 正解 {txt_path.name} が無いのでスキップ")
            continue
        text = txt_path.read_text(encoding="utf-8").strip()
        out.append((wav_path.stem, text, False, wav_path))
    return out


def read_wav(path: Path) -> np.ndarray:
    """16kHz mono int16 の WAV を読む。形式不一致は ValueError。

    (assert にすると -O 実行で消えて不正データを黙って計測してしまう)
    """
    with wave.open(str(path)) as w:
        if (
            w.getframerate() != 16000
            or w.getnchannels() != 1
            or w.getsampwidth() != 2
        ):
            raise ValueError(
                f"{path.name}: 16kHz/mono/int16 ではありません "
                f"(rate={w.getframerate()}, ch={w.getnchannels()}, "
                f"width={w.getsampwidth()})。変換例: "
                f"afconvert -f WAVE -d LEI16@16000 -c 1 {path.name} out.wav"
            )
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def build_prompt(style: str) -> str | None:
    if style == "none":
        return None
    if style == "terms":
        return " ".join(BENCH_TERMS)
    if style == "list":
        return render_prompt_list(BENCH_TERMS)
    raise ValueError(style)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--models",
        default="turbo,large-v3",
        help=f"比較するモデル (カンマ区切り、候補: {','.join(MODELS)})",
    )
    ap.add_argument(
        "--prompts",
        default="none,list",
        help="prompt 条件 (カンマ区切り: none, terms, list)",
    )
    ap.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="実発話セットのディレクトリ (NNN.wav + NNN.txt)",
    )
    ap.add_argument(
        "--no-vad", action="store_true", help="VAD を通さず生音声で計測"
    )
    args = ap.parse_args()

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    prompt_styles = [p.strip() for p in args.prompts.split(",") if p.strip()]
    for m in model_keys:
        if m not in MODELS:
            print(f"unknown model: {m} (候補: {', '.join(MODELS)})")
            return 1

    if args.audio_dir:
        samples = load_real_set(args.audio_dir.expanduser())
        source = f"実発話 ({args.audio_dir})"
    else:
        samples = synthesize_fixtures()
        source = "TTS (say/Kyoko)"
    if not samples:
        print("サンプルがありません")
        return 1

    print(f"# STT ベンチ — {source}, {len(samples)} サンプル")
    print(f"# models={model_keys} prompts={prompt_styles} vad={not args.no_vad}")
    print()
    header = f"{'model':<10} {'prompt':<7} {'CER':>7} {'CER(固有)':>9} {'lat_med':>8} {'lat_p95':>8}"
    print(header)
    print("-" * len(header))

    # モデル単位で直列 (mlx_whisper ModelHolder は 1 モデルキャッシュ)
    for mkey in model_keys:
        stt = WhisperSTT(MODELS[mkey])
        try:
            stt.warmup()
        except Exception as e:
            print(f"{mkey:<10} ロード失敗: {e}")
            continue
        for pstyle in prompt_styles:
            prompt = build_prompt(pstyle)
            cers: list[float] = []
            cers_proper: list[float] = []
            lats: list[float] = []
            for _fid, ref_text, proper, wav_path in samples:
                try:
                    audio = read_wav(wav_path)
                except ValueError as e:
                    print(f"  ! skip: {e}")
                    continue
                t0 = time.perf_counter()
                if args.no_vad:
                    hyp = stt.transcribe(audio, initial_prompt=prompt)
                else:
                    chunks = vad.split_speech_chunks(audio, sample_rate=16000)
                    hyp = stt.transcribe_chunks(chunks, initial_prompt=prompt)
                lat = time.perf_counter() - t0
                c = cer(ref_text, hyp)
                cers.append(c)
                if proper:
                    cers_proper.append(c)
                lats.append(lat)
            lat_sorted = sorted(lats)
            p95 = lat_sorted[min(len(lat_sorted) - 1, int(len(lat_sorted) * 0.95))]
            proper_str = (
                f"{statistics.mean(cers_proper):.3f}" if cers_proper else "-"
            )
            print(
                f"{mkey:<10} {pstyle:<7} {statistics.mean(cers):>7.3f} "
                f"{proper_str:>9} {statistics.median(lats):>7.2f}s {p95:>7.2f}s"
            )
    print()
    print("# 判定基準: 全体 1.2x 予算 → stt 単体で turbo 比 +0.4s 程度まで許容。")
    print("# TTS と実発話の両方で CER が改善するモデルのみ採用を推奨。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
