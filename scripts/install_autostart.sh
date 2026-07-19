#!/usr/bin/env bash
# Install voiceinput as a per-user LaunchAgent so it starts on login.
#
# Run once after the project is set up (`uv sync`). The agent is also
# loaded into the running session, so voiceinput should pop into the
# menu bar within a few seconds.
#
# TCC permissions are still attributed to whichever binary calls into
# the macOS APIs — typically `uv` for our setup. If they have not been
# granted yet, the first launch will surface the standard system
# dialogs as "uv". Approve them once and they stick.
#
# Usage:   bash scripts/install_autostart.sh
# Remove:  bash scripts/uninstall_autostart.sh
#
# Optional env vars:
#   UV       path to uv binary (default: ~/.local/bin/uv or PATH)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
UV_PATH="${UV:-$HOME/.local/bin/uv}"
LABEL="com.voiceinput.agent"
PLIST_DEST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG_DIR="$HOME/Library/Logs/voiceinput"

if [ ! -x "$UV_PATH" ]; then
    UV_PATH="$(command -v uv 2>/dev/null || true)"
fi
if [ -z "$UV_PATH" ] || [ ! -x "$UV_PATH" ]; then
    echo "uv not found. Set UV=/path/to/uv or install uv first." >&2
    exit 1
fi

if [ ! -e "$PROJECT_DIR/.venv" ]; then
    echo ".venv not initialized. Run 'uv sync' first." >&2
    exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR"

# Stop any voiceinput that is already running, no matter how it was
# started (terminal `uv run`, an old LaunchAgent, a leftover from a
# previous session). Without this the menu bar ends up with two 🎤
# icons after install.
pkill -f '-m voiceinput' 2>/dev/null || true
sleep 0.3

# If the agent is already loaded, unload it before rewriting the plist.
launchctl unload "$PLIST_DEST" 2>/dev/null || true

cat > "$PLIST_DEST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$UV_PATH</string>
        <string>run</string>
        <string>--directory</string>
        <string>$PROJECT_DIR</string>
        <string>python</string>
        <string>-m</string>
        <string>voiceinput</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
    <key>ProcessType</key>
    <string>Interactive</string>
    <key>StandardOutPath</key>
    <string>$LOG_DIR/launchd-stdout.log</string>
    <key>StandardErrorPath</key>
    <string>$LOG_DIR/launchd-stderr.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin</string>
    </dict>
</dict>
</plist>
EOF

launchctl load -w "$PLIST_DEST"

echo "✅ Installed launch agent: $PLIST_DEST"
echo
echo "  • voiceinput will start automatically every time you log in."
echo "  • It is also running right now (check the menu bar for 🎤)."
echo "  • Logs:  tail -f $LOG_DIR/launchd-stderr.log"
echo "  • Stop:  bash scripts/uninstall_autostart.sh"
