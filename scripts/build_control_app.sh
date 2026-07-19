#!/usr/bin/env bash
# Build a tiny AppleScript .app that safely starts (or reports already-running)
# the voiceinput LaunchAgent — without force-killing a running instance.
#
# Codex adversarial-review findings addressed:
#  1. [high]  No -k flag: never force-kills a running instance.
#             If voiceinput is already up → no-op + "already running" notification.
#  2. [high]  Real health check: polls pgrep for up to 5 s before declaring
#             success; surfaces an actionable error (log path) if it stays down.
#  3. [medium] Plist guard: if the LaunchAgent was never installed (or was
#             removed), the app shows a guided error instead of a raw shell failure.
#
# Usage:   bash scripts/build_control_app.sh
#
# Optional env vars:
#   INSTALL_DIR   where to put Voice Input.app (default: ~/Applications)

set -euo pipefail

APP_NAME="Voice Input.app"
INSTALL_DIR="${INSTALL_DIR:-$HOME/Applications}"
APP_PATH="$INSTALL_DIR/$APP_NAME"
LABEL="com.voiceinput.agent"

mkdir -p "$INSTALL_DIR"

TMP_SCRIPT="$(mktemp /tmp/voiceinput-control.XXXXXX).applescript"
trap 'rm -f "$TMP_SCRIPT"' EXIT

# NOTE: heredoc uses 'APPLESCRIPT' (quoted) so the shell does NOT expand
# variables inside — the AppleScript itself builds paths at runtime.
cat > "$TMP_SCRIPT" <<'APPLESCRIPT'
on run
    set agentLabel to "com.voiceinput.agent"
    set uid to do shell script "id -u"
    set homeDir to POSIX path of (path to home folder)
    set plistPath to homeDir & "Library/LaunchAgents/" & agentLabel & ".plist"
    set logDir to homeDir & "Library/Logs/voiceinput"

    -- 1. LaunchAgent plist が存在するか確認
    --    未インストールなら案内メッセージを出して終了
    set plistExists to do shell script "[ -f " & quoted form of plistPath & " ] && echo yes || echo no"
    if plistExists is "no" then
        display notification "先に install_autostart.sh を実行してください。" ¬
            with title "Voice Input" subtitle "⚠️ セットアップ未完了"
        return
    end if

    -- 2. すでに動いているなら強制終了しない（録音・処理中を守る）
    set isRunning to do shell script ¬
        "pgrep -f -- '-m voiceinput' > /dev/null 2>&1 && echo yes || echo no"
    if isRunning is "yes" then
        display notification "すでに起動しています。menu bar の 🎤 を確認してください。" ¬
            with title "Voice Input"
        return
    end if

    -- 3. 停止中なら kickstart（-k なし = 強制終了なし）
    --    失敗した場合は load してから再試行（unloaded 状態に対応）
    set kickErr to ""
    try
        do shell script "launchctl kickstart gui/" & uid & "/" & agentLabel
    on error
        try
            do shell script "launchctl load -w " & quoted form of plistPath
            do shell script "launchctl kickstart gui/" & uid & "/" & agentLabel
        on error errMsg
            set kickErr to errMsg
        end try
    end try

    if kickErr is not "" then
        display notification "起動できませんでした: " & kickErr ¬
            with title "Voice Input" subtitle "⚠️ エラー"
        return
    end if

    -- 4. ヘルスチェック: 最大 5 秒ポーリング（0.5 s × 10 回）
    --    launchd が accept しただけでなく、プロセスが実際に上がったことを確認する
    set started to false
    repeat 10 times
        delay 0.5
        set alive to do shell script ¬
            "pgrep -f -- '-m voiceinput' > /dev/null 2>&1 && echo yes || echo no"
        if alive is "yes" then
            set started to true
            exit repeat
        end if
    end repeat

    if started then
        display notification "voiceinput を起動しました。menu bar の 🎤 を確認してください。" ¬
            with title "Voice Input"
    else
        display notification "起動できませんでした。ログを確認してください。" ¬
            with title "Voice Input" subtitle "⚠️ エラー"
        -- ログディレクトリを Finder で開いて調査しやすくする
        do shell script "open " & quoted form of logDir
    end if
end run
APPLESCRIPT

rm -rf "$APP_PATH"
osacompile -o "$APP_PATH" "$TMP_SCRIPT"

echo "✅ Built: $APP_PATH"
echo
echo "  • Spotlight (Cmd+Space) で 'Voice Input' → Enter"
echo "  • Finder → ~/Applications のアイコンを Dock にドラッグでも OK"
echo "  • 起動中に叩いても録音・処理を邪魔しません（no-op になります）"
echo "  • 停止は menu bar 🎤 → Quit"
