@echo off
REM Build a single-file Windows .exe launcher.
REM Requires: pip install pyinstaller

pip install pyinstaller
pyinstaller --onefile --windowed --name ModelAdvisor ^
  --add-data "index.html;." ^
  --add-data "taxonomy.json;." ^
  --add-data "benchmark.json;." ^
  --add-data "keywords.json;." ^
  --add-data "config.json;." ^
  launcher.py

echo.
echo Done. The .exe is in the "dist" folder.
pause
