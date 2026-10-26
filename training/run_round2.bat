@echo off
cd /d "C:\Users\Melia Serengeti User\Desktop\wildlife_tracker"
echo %DATE% %TIME% run_round2.bat started >> logs\round2_console.log
"C:\Users\Melia Serengeti User\Desktop\wildlife_tracker\.venv313\Scripts\python.exe" training\round2.py >> logs\round2_console.log 2>&1
echo %DATE% %TIME% run_round2.bat finished, exit %ERRORLEVEL% >> logs\round2_console.log

