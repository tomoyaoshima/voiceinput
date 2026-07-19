#!/usr/bin/env bash
# Stop the voiceinput LaunchAgent and remove its plist so it no longer
# starts on login.
#
# This does NOT touch the project itself or the user's TCC permissions.
# To run voiceinput manually after uninstalling, use:
#   uv run python -m voiceinput

set -euo pipefail

LABEL="com.voiceinput.agent"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

if [ -f "$PLIST" ]; then
    launchctl unload "$PLIST" 2>/dev/null || true
    rm "$PLIST"
    echo "✅ Removed: $PLIST"
else
    echo "Not installed (no plist at $PLIST)."
fi

# Also kill any running voiceinput so the menu bar disappears immediately.
pkill -f 'python -m voiceinput' 2>/dev/null || true
