@echo off
REM Started by task MeliaWildlifeTrainingRound3 through tools\run_hidden.vbs (no window).
REM --reuse: the LILA download, harvest and teacher labels of 22-23/10 are done.
REM --deadline: training stops in time for the tracker to run before 05:30.
cd /d "C:\Users\Melia Serengeti User\Desktop\wildlife_tracker"
echo %DATE% %TIME% run_round3.bat started >> logs\round3_console.log
"C:\Users\Melia Serengeti User\Desktop\wildlife_tracker\.venv313\Scripts\python.exe" training\round3.py --reuse --deadline 05:30 >> logs\round3_console.log 2>&1
echo %DATE% %TIME% run_round3.bat finished, exit %ERRORLEVEL% >> logs\round3_console.log
