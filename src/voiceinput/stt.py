import logging
import threading

import mlx_whisper
import numpy as np

from voiceinput.text_filter import strip_whisper_hallucinations

_logger = logging.getLogger("voiceinput.stt")

# temperature fallback の上限。mlx_whisper のデフォルトは (0.0, 0.2, ..., 1.0)
# だが、高温 (0.6-1.0) のサンプリングは幻覚単語の温床なので 0.4 で打ち切る。
# fallback 打ち切りにより「品質条件を満たさない最終結果」がそのまま採用される
# 挙動になる (mlx_whisper transcribe.py の decode_with_fallback を検証済み) が、
# その中で最も危険な「反復文」は下の _drop_repetition_segments で破棄する。
_TEMPERATURES = (0.0, 0.2, 0.4)

# 反復ガードの判定値。最終温度まで fallback した (= 品質条件を満たせなかった)
# うえで compression_ratio が閾値を超えているセグメントは、
# 「ありがとうございますありがとうございます…」型の反復である可能性が
# 高いので本文から落とす。
_GUARD_TEMPERATURE = _TEMPERATURES[-1]
_GUARD_COMPRESSION_RATIO = 2.0


def _drop_repetition_segments(result: dict) -> tuple[str, float]:
    """反復セグメントを除いた本文と、観測された最大 temperature を返す。

    mlx_whisper の result["segments"] には各セグメントの
    temperature / compression_ratio / avg_logprob が入っている。
    fallback が最終温度まで達し (temperature >= 0.4) かつ
    compression_ratio > 2.0 のセグメントは反復幻覚とみなして破棄する。

    segments が無い / 形が想定外の場合は result["text"] をそのまま返す
    (defensive: mlx_whisper のバージョン差異で落とさない)。
    """
    segments = result.get("segments")
    if not isinstance(segments, list) or not segments:
        return (result.get("text") or "").strip(), 0.0

    kept: list[str] = []
    dropped = 0
    max_temp = 0.0
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        temp = float(seg.get("temperature") or 0.0)
        cr = float(seg.get("compression_ratio") or 0.0)
        max_temp = max(max_temp, temp)
        if temp >= _GUARD_TEMPERATURE and cr > _GUARD_COMPRESSION_RATIO:
            dropped += 1
            continue
        kept.append((seg.get("text") or ""))
    if dropped:
        _logger.warning(
            "dropped %d repetition-suspect segment(s) (temp>=%.1f, cr>%.1f)",
            dropped,
            _GUARD_TEMPERATURE,
            _GUARD_COMPRESSION_RATIO,
        )
    return "".join(kept).strip(), max_temp


class WhisperSTT:
    def __init__(self, model_name: str, language: str = "ja") -> None:
        self.model_name = model_name
        self.language = language
        # mlx_whisper の ModelHolder はロックなしの単一スロットキャッシュ
        # (transcribe.py で検証済み)。モデル切替の warmup と録音パイプラインの
        # transcribe が同時に走ると、二重ロードのメモリスパイクやモデル
        # 取り違えが起きるため、この RLock で直列化する。
        # (RLock なのは switch_model がロック保持のまま transcribe を呼ぶため)
        self._lock = threading.RLock()

    def transcribe(
        self,
        audio: np.ndarray,
        initial_prompt: str | None = None,
        *,
        _model_name: str | None = None,
    ) -> str:
        if audio.size == 0:
            return ""
        kwargs: dict = dict(
            path_or_hf_repo=_model_name or self.model_name,
            language=self.language,
            # 前のセグメントテキストに引っ張られて幻覚を増幅させない
            condition_on_previous_text=False,
            # mlx_whisper デフォルトと同値 (0.6)。明示しておく
            no_speech_threshold=0.6,
            # 出力の繰り返しっぽい区間 (幻覚の典型) を破棄しやすくする
            # (mlx デフォルト 2.4 より厳しめ)
            compression_ratio_threshold=2.0,
            # fallback 判定に使われる (mlx デフォルトと同値だが明示)
            logprob_threshold=-1.0,
            # 高温リトライを 0.4 で打ち切る (幻覚抑止)
            temperature=_TEMPERATURES,
            # fallback (t>0) 時のみ 3 系列サンプリングして最良を選ぶ。
            # t=0 の greedy パスでは mlx_whisper が自動で無視するのでコストゼロ。
            # 注意: beam_size は mlx_whisper 未実装 (渡すと即例外) — 使わない。
            best_of=3,
        )
        # カスタム語彙ヒント。空文字なら渡さない (Whisper のデフォルト挙動を維持)。
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt
        with self._lock:
            result = mlx_whisper.transcribe(audio.astype(np.float32), **kwargs)
        text, max_temp = _drop_repetition_segments(result)
        if max_temp > 0.0:
            # fallback 発動率の可視化 (best_of コストが予算内かの実測用)
            _logger.info("stt fallback engaged (max_temperature=%.1f)", max_temp)
        return strip_whisper_hallucinations(text)

    def transcribe_chunks(
        self, chunks: list[np.ndarray], initial_prompt: str | None = None
    ) -> str:
        """複数チャンクを順に transcribe して連結する。

        Whisper の initial_prompt は最初の 30 秒窓にしか効かない
        (condition_on_previous_text=False では窓ごとに prompt がリセット
        される) ため、長い録音は VAD 境界で ≤28s のチャンクに分割し、
        **チャンクごとに同じ prompt を与える**ことで後半にも語彙ヒントを
        効かせる。チャンク 1 個なら transcribe と完全に同一パス。

        モデル名はループ開始時にスナップショットする — 処理中に menu で
        モデル切替が起きても、1 回の録音は必ず同一モデルで全チャンクを
        処理する (前半 turbo / 後半 large-v3 のような取り違えを防ぐ)。
        """
        if not chunks:
            return ""
        model = self.model_name
        if len(chunks) == 1:
            return self.transcribe(
                chunks[0], initial_prompt=initial_prompt, _model_name=model
            )
        texts: list[str] = []
        for chunk in chunks:
            t = self.transcribe(
                chunk, initial_prompt=initial_prompt, _model_name=model
            )
            if t:
                texts.append(t)
        # 日本語は語間スペースなしで連結。他言語は単語結合を防ぐため空白区切り。
        sep = "" if self.language == "ja" else " "
        return sep.join(texts)

    def switch_model(self, name: str) -> None:
        """新モデルをロード・warmup してから切り替える (成功時のみ反映)。

        - ロード (数百 MB〜数 GB のダウンロードを含みうる) はロック内で行い、
          ModelHolder (単一スロットキャッシュ) の競合を防ぐ。
        - ``model_name`` の書き換えは **warmup 成功後**。失敗時は例外を
          投げ、model_name は変更されない (呼び出し側の revert 不要)。
        - ロック保持中に録音パイプラインが transcribe を呼ぶとロード完了まで
          待たされるが、単一スロットキャッシュ上ロード開始後に旧モデルで
          転写してもスラッシングするだけなので、待つのが正しい挙動。
        """
        with self._lock:
            # 成功するまで self.model_name は触らない
            self.transcribe(
                np.zeros(16000, dtype=np.float32), _model_name=name
            )
            self.model_name = name

    def warmup(self) -> None:
        self.transcribe(np.zeros(16000, dtype=np.float32))
