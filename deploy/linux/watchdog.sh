#!/bin/bash
# melia-tracker-watchdog.timer, every 5 min.  systemd restarts a tracker that EXITS; this
# restarts one that is frozen (23/10 on Windows: alive, but no control-loop line for an hour)
# and one that was left stopped by a failed start.
cd "$(dirname "$0")/../.." || exit 1
LOG=logs/watchdog.log
HEARTBEAT=logs/ptz_trace.csv          # one line per second from the control loop
FROZEN_SECONDS=300
stamp() { date '+%Y-%m-%d %H:%M:%S'; }

if ! systemctl --user is-active --quiet melia-tracker.service; then
    echo "$(stamp)  tracker not running - starting it" >> "$LOG"
    systemctl --user start melia-tracker.service
    exit 0
fi
up_since=$(date -d "$(systemctl --user show -p ActiveEnterTimestamp --value melia-tracker.service)" +%s 2>/dev/null || echo 0)
now=$(date +%s)
beat=$(stat -c %Y "$HEARTBEAT" 2>/dev/null || echo 0)
if (( now - up_since > FROZEN_SECONDS && now - beat > FROZEN_SECONDS )); then
    echo "$(stamp)  tracker frozen (no $HEARTBEAT line for $(( now - beat )) s) - restarting it" >> "$LOG"
    systemctl --user restart melia-tracker.service
fi
