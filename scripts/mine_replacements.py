#!/usr/bin/env python
"""履歴から確定置換 (replacements) の候補を提示する。

使い方:
    uv run python scripts/mine_replacements.py [--min-count 2] [--limit 200]

履歴 (~/Library/Application Support/voiceinput/history.jsonl) の
raw_text (STT 生出力) と text (LLM 整形後) の差分から、繰り返し出現する
「誤変換 → 修正」ペアを頻度順に提示する。良さそうな候補を config.yaml の
`replacements:` にコピペすれば、以後は LLM の気まぐれに頼らず確定で直る。

出力はそのまま config.yaml に貼れる YAML 形式。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voiceinput.config import load_config  # noqa: E402
from voiceinput.history import History  # noqa: E402
from voiceinput.mining import mine_replacement_candidates  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="この回数以上繰り返された差分だけを候補にする (default: 2)",
    )
    ap.add_argument(
        "--limit", type=int, default=200, help="遡る履歴件数 (default: 200)"
    )
    args = ap.parse_args()

    history = History()
    entries = [
        (e.raw_text or "", e.text or "")
        for e in history.list(limit=args.limit)
        if e.raw_text and e.text
    ]
    if not entries:
        print("履歴がありません (raw_text 付きのエントリが 0 件)。")
        return 1

    candidates = mine_replacement_candidates(entries, min_count=args.min_count)
    print(f"# 履歴 {len(entries)} 件から抽出した置換候補 (出現回数順)")
    print(f"# 採用するものを config.yaml の `replacements:` にコピペしてください。")
    print(f"# ⚠️ from が一般語 (例: 回収) の場合、その語を本当に使う発話まで")
    print(f"#    置換されます。固有名詞・カタカナ誤変換など「常に直したい」もの")
    print(f"#    だけを採用してください。")
    print()
    if not candidates:
        print("# (候補なし — 繰り返される誤変換は見つかりませんでした)")
        return 0
    # 既に config で採用済みのルールにはマークを付ける (履歴の text は
    # replacements 適用後なので、採用済みペアも候補に出続けるため)
    adopted = {src for src, _to in load_config().replacements}
    for src, dst, cnt in candidates:
        mark = " (採用済み)" if src in adopted else ""
        print(f"# {cnt} 回{mark}")
        print(f'  - from: "{src}"')
        print(f'    to: "{dst}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
