@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing
".venv\Scripts\python.exe" -c "import torch, diffusers, PIL" >nul 2>&1
if errorlevel 1 goto missing
".venv\Scripts\python.exe" app.py
if errorlevel 1 pause
exit /b
:missing
echo Run setup.bat first, then open this launcher again.
pause
