@echo off
cd /d "%~dp0"

REM Free port 8080 if an older instance is still running (prevents "stale config").
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8080 ^| findstr LISTENING') do (
  taskkill /F /PID %%a >nul 2>nul
)

REM Find a working Python (try the Windows launcher `py`, then `python`).
py -3 -c "import sys" >nul 2>nul && goto runpy
python -c "import sys" >nul 2>nul && goto runpython

echo.
echo Python was not found on this computer.
echo Install it from https://www.python.org/downloads/
echo  - During install, tick the box "Add python.exe to PATH"
echo Then run this file again.
echo.
pause
exit /b 1

:runpy
py -3 launcher.py
exit /b 0

:runpython
python launcher.py
exit /b 0
