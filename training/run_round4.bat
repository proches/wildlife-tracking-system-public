@echo off
REM Started by task MeliaWildlifeTrainingRound4 through tools\run_hidden.vbs (no window).
cd /d "C:\Users\Melia Serengeti User\Desktop\wildlife_tracker"
echo %DATE% %TIME% run_round4.bat started >> logs\round4_console.log
"C:\Users\Melia Serengeti User\Desktop\wildlife_tracker\.venv313\Scripts\python.exe" training\round4.py --deadline 05:30 >> logs\round4_console.log 2>&1
echo %DATE% %TIME% run_round4.bat finished, exit %ERRORLEVEL% >> logs\round4_console.log
