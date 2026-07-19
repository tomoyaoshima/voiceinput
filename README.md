# voiceinput

**Apple Silicon Mac 上で完全ローカルに動く AI 音声入力ツール。**
右 Command キーを押して喋る → 文字起こし → AI が整形 → 今開いているアプリにそのままペースト。

AquaVoice や Typeless のような音声入力を、**クラウドに一切送らず・月額 0 円**で実現します。
音声もテキストも Mac の中だけで処理されるので、仕事の内容や機密情報を安心して喋れます。

```
🎤 右Cmd → 録音 → mlx-whisper (文字起こし) → Ollama (整形) → アクティブアプリへペースト
                        すべてローカル / 通信ゼロ
```

**主な特徴**

- 🔒 **完全ローカル** — 音声・テキストが外部に出ない。オフラインでも動く
- ⚡ **速い** — 発話終了から 1〜3 秒でペースト (Apple Silicon の MLX を利用)
- 🧹 **AI 整形** — 「えーと」などのフィラー除去・句読点補完・同音異義の修正
- 📚 **自分の語彙を学習** — 履歴から固有名詞を自動抽出して認識精度を上げる
- 🖥️ **画面コンテキスト認識** — 今開いている入力欄の文字を読んで表記を合わせる
- 📮 **モード切替** — そのまま / 整形 / ビジネスメール / 英訳メール
- 🎛️ **メニューバー常駐** — モデル切替・語彙編集・履歴が全部そこから

> Swift ネイティブ版へ移植する前提の Python プロトタイプですが、日常利用に耐える完成度です。

---

## 必要なもの

- **Apple Silicon の Mac** (M1 以降) / macOS 14+
  ※ Whisper の高速化に Apple の MLX を使うため Intel Mac は対象外です
- メモリ 16GB 以上を推奨 (LLM 整形モデルを載せるため)
- [uv](https://docs.astral.sh/uv/) — Python 環境を自動で用意してくれるツール
  (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- [Ollama](https://ollama.com/) — ローカル LLM 実行環境。インストールして起動しておく
- マイク

## インストール

```bash
git clone https://github.com/tomoyaoshima/voiceinput.git
cd voiceinput

# 仮想環境と依存をまとめて構築 (uv 管理の arm64 Python を使う)
uv sync --python-preference only-managed

# Ollama に整形用モデルを pull (default は gemma4:e4b ≈ 9.6GB)
# 実運用比較で「控えめ整形・速い・meta なし」が一番フィット
ollama pull gemma4:e4b
# 任意: 整形を強めにしたい時用のサブ候補
# ollama pull qwen2.5:14b      # 同音異義の修正に積極的 (meta は内蔵 strip)
# ollama pull gemma4:26b       # 高精度・内部 reasoning 系

# 初回は Whisper モデル (~3GB) を先に落としておくと起動が速い
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/whisper-large-v3-mlx')"
```

## 必須の権限 (System Settings → Privacy & Security)

`uv run` で起動した場合、許可する対象は **`.venv/bin/python3.11` または起動した Terminal アプリ** です。
権限はバイナリ単位なので、バイナリを別物に変えると再許可が必要になります。

| 権限 | 何に使うか | 設定場所 |
|------|----------|---------|
| マイク (Microphone) | sounddevice の録音 | System Settings → Privacy & Security → Microphone |
| 入力監視 (Input Monitoring) | グローバルホットキー (Option+Space) の検知 | Privacy & Security → Input Monitoring |
| アクセシビリティ (Accessibility) | 整形結果を Cmd+V で自動ペースト | Privacy & Security → Accessibility |

初回起動時に上記 3 つのダイアログが順番に出ます。すべて許可してアプリを再起動してください。
通知センターの権限は pync が初回 notify 時に自動で要求します。

## 起動

```bash
uv run python -m voiceinput
```

メニューバーに 🎤 が出ます。

### ログイン時に自動起動 (任意)

PC を再起動しても voiceinput が menu bar に勝手に立ち上がるようにしたい場合は、LaunchAgent としてインストール:

```bash
bash scripts/install_autostart.sh
```

`~/Library/LaunchAgents/com.voiceinput.agent.plist` が作られて、ログインのたびに自動起動するようになる。即時にも 1 回起動するので、実行直後から menu bar に 🎤 が出る。

ログ:
```bash
tail -f ~/Library/Logs/voiceinput/launchd-stderr.log
```

自動起動を解除:
```bash
bash scripts/uninstall_autostart.sh
```

### Mac アプリとして起動・再起動する

LaunchAgent をインストール済みなら、`Voice Input.app` をビルドすることで
ターミナルなしで voiceinput を起動・再起動できる:

```bash
bash scripts/build_control_app.sh
```

`~/Applications/Voice Input.app` が生成される。
- **Spotlight** (Cmd+Space) で「Voice Input」と打って Enter するだけで起動
- Dock に追加したい場合は Finder → `~/Applications` からドラッグ
- voiceinput が **録音中・処理中のときに叩いても何もしない**（強制終了なし）
- すでに起動済みの場合は「すでに起動しています」通知のみ
- 停止は menu bar 🎤 → **Quit**

> ℹ️ `Voice Input.app` は launchctl を呼ぶだけで TCC 権限を必要としません。
> TCC（マイク・入力監視・アクセシビリティ）は引き続き `uv` に付与されます。

> ⚠️ 自動起動した voiceinput は **`uv` から起動**されるため、TCC 権限は **`uv` バイナリ**に対して付与する必要があります(マイク・入力監視・アクセシビリティ)。初回起動時に macOS のダイアログが出るので、すべて許可してください。

### ホットキー操作

デフォルトは **右 Command キー単独押し** (`<cmd_r>`) で**シンプルなトグル動作**:

- **右 Cmd を 1 回押す**: 録音開始 (🎤 → 🟠)
- **もう一度押す**: 録音停止 → 文字起こし → ペースト
- **録音中に Esc**: キャンセル (音声を破棄、文字起こし・ペーストなし)
- 右 Cmd を他のキーと組み合わせた場合 (Cmd+C など) はトグルしないので、通常のキー操作は邪魔しない

`config.yaml` の `hotkey` を `<alt>+<space>` のような **修飾キー + メインキー** の組み合わせに変えると、AquaVoice 風の **押し続け録音 + 短押し2回でラッチ** モードに切り替わる:

- 押し続ける: 押している間だけ録音 (push-to-talk)、離した瞬間に文字起こし開始
- 短くタップ (350ms 未満): 継続録音モード (🟠)、もう一度タップすると停止
- ⚠️ Space をメインキーにすると押下中にメモなどへ Space が漏れる場合あり。`<ctrl>+<alt>+v` のような Space を使わない組み合わせを推奨

| menu bar 表示 | 状態 | 何をしているか |
|---|---|---|
| `🎤` | idle | 待機中 |
| `🔴 REC` | recording (hold) | 押下中の録音 (combo モードのみ) |
| `🟠 REC` | recording | 継続録音中 (もう一度押すと停止) |
| `⏳ …` | processing | 文字起こし + 整形中 |

### フィードバック音

「開始」「停止」を直感的に区別できるよう、numpy で対数周波数スイープを生成して使う:

- **録音開始**: 上昇 sweep (440Hz → 1320Hz、"ひゅっ" と立ち上がる)
- **録音停止**: 下降 sweep (1320Hz → 480Hz、落ち着いて閉じる)
- **エラー時**: 急下降 sweep (660Hz → 110Hz、深く落ちる)

WAV は OS の一時ディレクトリに 1 度だけ生成・キャッシュ、以降は NSSound で即時再生。
音量を消したければシステムの「サウンドエフェクト」音量で。

## 設定ファイル

`config.yaml.example` を `~/.config/voiceinput/config.yaml` か `./config.yaml` にコピーして編集します。

```yaml
hotkey: "<cmd_r>"
whisper:
  model: "mlx-community/whisper-large-v3-turbo"  # 高速・実用精度 (推奨)
  language: "ja"
ollama:
  endpoint: "http://localhost:11434"
  model: "gemma4:e4b"      # 推奨デフォルト。menu bar から別モデルへ切替可 (state.json に永続化)
  num_predict: 1024         # gemma4:26b 等の reasoning 系も含めた安全値
  num_ctx: 1024
  keep_alive: "24h"         # menu bar 常駐に合わせて Ollama 側もほぼ常駐
default_format_mode: "clean"     # raw | clean | mail
auto_paste: true                  # menu bar からも切替可
notify: true
logging_enabled: true             # menu bar からも切替可
log_dir: "~/Library/Logs/voiceinput"
prompts_dir: "./prompts"
vocabulary:
  enabled: true
  manual: ["voiceinput", "Codex"]   # 必ず認識させたい固有名詞
  history_size: 100
  top_n: 30
  rebuild_every: 10
  max_chars: 200
```

設定の読み込み順: `./config.yaml` → `~/.config/voiceinput/config.yaml` → 内蔵デフォルト。

## 整形モード

- **raw**: STT 結果をそのまま出力 (LLM を呼ばない)
- **clean**: フィラー除去・句読点補正・誤認識の自然修正
- **mail**: です・ます調・敬語・段落整形

`prompts/clean.yaml` `prompts/mail.yaml` を編集すればプロンプトをカスタムできます。

## メニューバーの使い方

メニューバーの 🎤 をクリック:

**アプリの操作はすべてメニューバーから完結します** (設定ファイルを触らなくても
モード切替・モデル切替・語彙登録ができます)。

```
🎤
├── Status: idle                    現在の状態
├── ──
├── Format          ▶  raw / clean / mail / mail_en
├── LLM Model       ▶  整形用モデルの一覧 + Refresh model list
├── STT Model       ▶  文字起こしモデルの切替
├── History         ▶  直近 10 件 + Open history file / Clear history
├── Vocabulary      ▶  + 単語を追加… / ✎ 一覧をまとめて編集… / 登録済み (N) / Refresh now
├── Auto-paste          ✓ トグル
├── Logging             ✓ トグル
├── Screen context      ✓ トグル
└── Quit
```

- **Status**: 現在の状態 (idle / recording / processing)
- **Format**: 整形モード切替。`raw` (整形なし) / `clean` (フィラー除去・句読点) /
  `mail` (です・ます調のビジネスメール) / `mail_en` (英語ビジネスメールに翻訳)。
  `prompts/` に yaml を足せば独自モードも追加できる
- **LLM Model**: Ollama に pull 済みのモデル一覧から選択。クリックで即時切替 +
  背景 warmup。`Refresh model list` で `ollama pull` 直後のモデルも反映
- **STT Model**: 文字起こしモデルの切替 (turbo / large-v3 / turbo-q4 / kotoba)。
  ロード成功後に切り替わり、失敗時は元のモデルのまま (初回選択時はダウンロードあり)
- **History**: 直近 10 件の音声入力結果。クリックでクリップボードに再コピー、
  `Open history file` で全履歴を開く、`Clear history` で全消去
- **Vocabulary**: カスタム語彙の管理。`+ 単語を追加…` でダイアログから登録
  (カンマ・読点・スペース区切りで複数まとめて可)、`✎ 一覧をまとめて編集…` で
  テキストエディタ風にまとめて編集、`登録済み (N)` から各語をクリックで削除、
  `Refresh now` で履歴から即時再構築
- **Auto-paste**: クリップボード貼り付けのみ ↔ 自動ペーストの切替
- **Logging**: ログ収集 + 履歴記録の ON/OFF (機密の話を扱う時用)
- **Screen context**: 画面コンテキスト認識の ON/OFF (AX が使える環境のみ表示)
- **Quit**: 終了

メニューでの選択 (`format_mode` / `llm_model` / `whisper_model` /
`screen_context` / 追加した語彙) は
`~/Library/Application Support/voiceinput/state.json` に保存されるので、
アプリを再起動しても前回の設定で立ち上がる。`config.yaml` で明示指定された
値があればそちらが state より優先される。

## 画面コンテキスト認識 (Phase F)

録音開始時に「今フォーカスしている入力欄・選択テキスト・ウィンドウタイトル・
アプリ名」を macOS の Accessibility API で読み取り、そこから固有名詞を抽出して
**その録音だけ** Whisper の `initial_prompt` と整形 LLM に渡す。LLM には「正しい
表記の参考語リスト」だけを渡し (周辺の文そのものは渡さない)、発話に同じ語が出た時の
**漢字・カタカナ・英字の表記補正のみ**に使う (語の挿入・引用は禁止)。画面に映っている
初出の語 (相手の名前・プロジェクト名・専門用語) の表記精度が上がる。

> プログラミング補足: Accessibility API = macOS が「画面上の UI 要素やその文字」を
> プログラムから読めるようにする仕組み。スクリーンショットではなくテキストとして
> 読むので速く、画像サイズ上限の問題もない。

- **追加権限なし**: ペースト (Cmd+V) と同じ Accessibility 権限を使う。
- **プライバシー**: 全処理ローカル完結 (外部送信ゼロ)。パスワード欄 (`AXSecureTextField`)
  と機密アプリ (Keychain / 1Password 等) は本文を読まない。メール / トークン /
  カード番号などの機密パターンは自動マスク (███)。取得した生テキストは履歴・ログに
  残さない (ログは `app名 / trusted / 抽出語数` だけ)。menu の **Screen context** で
  即 OFF 可。**Logging を OFF にすると連動して無効**になる。
- **設定**: `config.yaml` の `screen_context:` セクション参照。入力欄本文まで読むのが
  不安なら `read_focused_value: false` にすると件名/選択テキスト/アプリ名のみ使う。

## 履歴

整形済みテキストは `~/Library/Application Support/voiceinput/history.jsonl` に
JSONL で append される (直近 200 件)。
1 行 1 入力で `timestamp / mode / text / raw_text / audio_sec / stt_sec / llm_sec`
を保存。Logging を OFF にしている間は履歴も書き込まれない。

## ログ

`~/Library/Logs/voiceinput/voiceinput.log` (5MB × 5 世代でローテーション)。
記録内容: VAD 所要、録音時間 → トリム後の発話時間、STT 所要、LLM 所要 (cold/warm/skip)、ペースト所要、整形モード、本文長、Whisper に渡した語彙の文字数。本文そのものはログ無効化中は出力されません。

例:
```
vad=0.05s audio=18.50s->4.20s (23%) stt=0.65s llm=0.92s(warm) paste=0.04s mode=clean len=42 vocab=120
```

- `llm` の括弧内は **cold/warm/skip**: cold は Ollama がモデルを再ロードしている (前回呼び出しから keep_alive を超えた)、warm はメモリ常駐中、skip は短文 (≤5 文字) や raw mode で LLM を呼ばなかったケース。
- `vocab` は Whisper の `initial_prompt` に渡した語彙文字列の長さ。0 ならカスタム語彙が空 (Phase E 無効、または履歴 0 件 + manual 空)。

## VAD (発話区間抽出)

録音バッファに無音区間が多いと Whisper の処理が無駄に長引くため、silero-vad で
発話区間だけを抜き出してから STT に渡す。例えば 30 秒録音(発話 5 秒)なら
5 秒ぶんに圧縮されて STT が 5-6 倍速くなる。録音が完全に無音だった場合は
STT/LLM をスキップして「発話を検出できませんでした」通知だけ出す。

Phase G の精度改善:
- 発話区間の**直結をやめ、区間の間に 400ms の無音を挿入** (不自然な密着による
  単語誤認識を防ぐ)。短いポーズ (700ms 未満) は元から保持
- 長い録音 (VAD 圧縮後 28 秒超) は **VAD 境界でチャンク分割**し、チャンクごとに
  語彙ヒントを与えて転写する (Whisper の initial_prompt は最初の 30 秒窓に
  しか効かないため。長文の後半でも固有名詞が当たるようになる)
- temperature fallback を 0.4 で打ち切り + 反復セグメントの後段破棄 (幻覚抑止)

## カスタム語彙ヒント (Whisper initial_prompt)

固有名詞 (人名・サービス名・社内用語) は Whisper のゼロショットだと誤変換
されやすい。voiceinput はこれを自動で補正するために、整形済み履歴から
**頻出のカタカナ・漢字・英数字識別子** を抽出し、`config.yaml` の
`vocabulary.manual` リストと合成して、Whisper の `initial_prompt` に渡す。

- Phase G から形式は**「語、語、語。」の句読点プライミング** (句読点出力が
  安定し、重要語 = 画面語を末尾に置いて条件付けを最大化)。従来のスペース
  区切りに戻すには `vocabulary.prompt_style: "terms"`
- 自動再構築は `vocabulary.rebuild_every` (デフォルト 10) 回ごとにバックグラウンドで実行
- 即時反映したい時は menu bar → **Vocabulary → Refresh now**
- raw_text には誤認識が混じり得るので、抽出ソースは LLM 整形後の `text` のみ
- OFF にしたい場合は `config.yaml` で `vocabulary.enabled: false`

## STT モデルの切替とベンチ (Phase G)

menu bar → **STT Model** で Whisper モデルを切り替えられる (選択は state.json に
永続化)。候補: `whisper-large-v3-turbo` (デフォルト・速度と精度のバランス) /
`whisper-large-v3-mlx` (精度最良・3-4 倍遅い) / `turbo-q4` (高速) /
`kotoba-whisper-v2.0-mlx` (日本語特化・個人変換)。初回選択時はダウンロードが走る。

モデル比較は同梱のベンチで:

```bash
# TTS 合成音声での相対比較 (turbo は計測済み: CER 0.007 / 0.45s)
uv run python scripts/bench_stt.py --models turbo,large-v3

# 実発話セット (推奨)。~/.cache/voiceinput/bench/real/ に
# 001.wav (16kHz mono) + 001.txt (正解文) のペアを 20-30 個置いて:
uv run python scripts/bench_stt.py --audio-dir ~/.cache/voiceinput/bench/real
```

TTS 音声は綺麗すぎてモデル差が出にくいので、**採用判定は実発話セットで**。

## 確定置換の候補マイニング (Phase G)

繰り返し出る固有名詞の誤変換は、語彙ヒントより `replacements` (確定置換) が
最も確実。履歴から候補を自動抽出できる:

```bash
uv run python scripts/mine_replacements.py
```

「誤変換 → 修正」ペアが頻度順に出るので、良いものを `config.yaml` の
`replacements:` にコピペする。from が一般語のもの (文脈依存) は採用しないこと。

## トラブルシュート

**ホットキーが効かない**
- 入力監視権限が未付与 → System Settings → Privacy & Security → Input Monitoring で `python3.11` (または Terminal) を ON
- 別アプリが Option+Space を奪っている → IME 切替などの設定を確認、`config.yaml` の `hotkey` を `<ctrl>+<alt>+<space>` などに変更

**Cmd+V がメモ帳に入らない**
- アクセシビリティ権限が未付与 → System Settings → Privacy & Security → Accessibility に `python3.11` を追加して ON

**Ollama がタイムアウトする**
- `ollama list` で `gemma4:e4b` (default) が入っているか確認
- 大きめのモデル (qwen2.5:14b / gemma4:26b 等) の初回呼び出しはコールドスタートで 5-10 秒かかる。アプリ起動直後に warmup 済みなのでしばらく待つ
- それでも遅ければ menu bar の **LLM Model** から軽量モデル (`gemma4:e4b`, `qwen2.5:7b`, `phi4:latest`) に切替

**さらに高速化したい (品質を少し落としても速度優先)**
- Whisper を q4 量子化版に切替: `config.yaml` で
  `whisper.model: "mlx-community/whisper-large-v3-turbo-q4"` を指定。
  STT が概ね 1.5-2 倍速くなるが、固有名詞の精度がやや落ちる。
  Phase E のカスタム語彙 (Whisper `initial_prompt`) で
  ある程度はカバーできる
- LLM は default の `gemma4:e4b` のまま運用 (軽量 9.6GB / ~0.4s)。さらに速くしたければ `qwen2.5:7b` や `phi4:latest`
- `config.yaml` で `ollama.keep_alive: "-1"` にすると Ollama がモデルを永続的に
  メモリ保持するので、長時間放置後のコールドスタートを完全に消せる
  (Mac mini のメモリが厳しいときは戻す)

**初回起動が遅い / Whisper モデルがダウンロードされない**
- 上の `snapshot_download` を手動で先に走らせる
- ネットワーク接続確認、`~/.cache/huggingface/hub/` を覗く

**録音が無音になる**
- マイク権限を確認。`uv run python -c "import sounddevice as sd; print(sd.query_devices())"` で入力デバイス一覧を表示

## テスト

```bash
uv run pytest tests/ -v
```

config / logger / FormatPipeline のユニットテストが走ります。
録音・STT・menu bar まわりは macOS 実機で手動確認してください。

## ディレクトリ構成

```
voiceinput/
├── pyproject.toml
├── README.md
├── config.yaml.example
├── prompts/
│   ├── clean.yaml
│   └── mail.yaml
├── src/voiceinput/
│   ├── __main__.py     # python -m voiceinput エントリ
│   ├── app.py          # rumps menu bar アプリ + パイプライン
│   ├── hotkey.py       # pynput GlobalHotKeys
│   ├── recorder.py     # sounddevice 録音
│   ├── stt.py          # mlx-whisper ラッパー
│   ├── llm.py          # Ollama HTTP + FormatPipeline
│   ├── paste.py        # pyperclip + Cmd+V + pync
│   ├── config.py       # YAML 設定
│   └── logger.py       # RotatingFileHandler
└── tests/
```

## ライセンス

MIT License — 商用・改変・再配布いずれも自由です。詳細は [LICENSE](LICENSE) を参照。

## 免責

個人が自分用に作ったツールを公開しているものです。動作保証・サポートの約束はありません。
不具合や改善案は Issue / Pull Request でどうぞ (対応をお約束はできませんが歓迎します)。

## 謝辞

- [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) — Apple Silicon 向け Whisper
- [Ollama](https://ollama.com/) — ローカル LLM 実行環境
- [silero-vad](https://github.com/snakers4/silero-vad) — 発話区間検出
- [rumps](https://github.com/jaredks/rumps) — macOS メニューバーアプリ
