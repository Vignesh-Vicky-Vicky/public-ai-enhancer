@echo off
setlocal
cd /d "%~dp0"
echo Installing Detail Lab. First setup needs internet and several GB of disk space.
if not exist ".venv\Scripts\python.exe" py -3.12 -m venv .venv
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail
".venv\Scripts\python.exe" download_model.py
if errorlevel 1 goto fail
echo Setup complete. Double-click Start Detail Lab.bat
pause
exit /b 0
:fail
echo Setup failed. Read the error above. Run setup.bat again to resume.
pause
exit /b 1
