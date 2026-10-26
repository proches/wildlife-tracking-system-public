@echo off
REM Started every 30 min by task MeliaGitAutoSync through tools\run_hidden.vbs (no window).
cd /d "%~dp0.."
".venv313\Scripts\python.exe" tools\git_autosync.py --reason scheduled >> logs\autosync_console.log 2>&1
