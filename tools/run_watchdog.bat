@echo off
REM Started daily at 06:00 by task MeliaWildlifeTrackerWatchdog through tools\run_hidden.vbs (no window).
cd /d "%~dp0.."
".venv313\Scripts\python.exe" tools\tracker_watchdog.py >> logs\watchdog_console.log 2>&1
