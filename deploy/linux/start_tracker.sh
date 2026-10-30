#!/bin/bash
# Started by melia-tracker.service (systemd --user).  The preview window needs the logged-in
# desktop (:0); before anyone logs in the tracker runs without it (WILDLIFE_PREVIEW=0).
cd "$(dirname "$0")/../.." || exit 1
export DISPLAY="${DISPLAY:-:0}" XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
if ! xdpyinfo >/dev/null 2>&1; then
    export WILDLIFE_PREVIEW=0
    echo "no desktop on $DISPLAY - running without the preview window"
fi
exec venv/bin/python main.py
