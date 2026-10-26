@echo off
REM Melia Serengeti Wildlife AI - one-time setup on a new or cleaned PC (read RESTORE.md first).
REM Needs: Python 3.13 64-bit (python.org, with the "py" launcher) and internet.
cd /d "%~dp0"

py -3.13 --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.13 was not found. Install Python 3.13 64-bit from https://www.python.org/downloads/
    echo SpeciesNet does not support Python 3.14 yet.
    pause
    exit /b 1
)

if not exist .venv313\Scripts\python.exe (
    echo Creating .venv313 ...
    py -3.13 -m venv .venv313 || goto :fail
)
.venv313\Scripts\python.exe -m pip install --upgrade pip || goto :fail
echo Installing the exact package versions of the working system (a few GB, takes a while) ...
.venv313\Scripts\python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu || goto :fail

if not exist .env (
    copy .env.example .env >nul
    echo.
    echo  .env created from .env.example - open it and put in the camera password
    echo  ^(WILDLIFE_CAMERA_PASSWORD^) before starting the tracker.
)

echo Checking the installation with the unit tests ...
.venv313\Scripts\python.exe -m unittest tests.test_training_tools tests.test_privacy tests.test_camera_boundaries || goto :fail
echo.
echo Setup done. Start the tracker with run_tracker.bat
pause
exit /b 0

:fail
echo.
echo Setup FAILED at the step above - see RESTORE.md.
pause
exit /b 1
