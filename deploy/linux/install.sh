#!/bin/bash
# Install the tracker, its watchdog and the GitHub auto-sync as systemd --user units
# (the Linux replacement of deploy/windows-tasks).  Run once:  bash deploy/linux/install.sh
set -e
here="$(cd "$(dirname "$0")" && pwd)"
units="$HOME/.config/systemd/user"
mkdir -p "$units"
chmod +x "$here/start_tracker.sh" "$here/watchdog.sh"
for u in melia-tracker.service melia-tracker-watchdog.service melia-tracker-watchdog.timer \
         melia-autosync.service melia-autosync.timer; do
    ln -sf "$here/$u" "$units/$u"
done
systemctl --user daemon-reload
systemctl --user enable --now melia-tracker.service melia-tracker-watchdog.timer melia-autosync.timer
# start at boot, before (and without) a desktop login
loginctl enable-linger "$USER"
systemctl --user --no-pager status melia-tracker.service | head -5
