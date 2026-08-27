@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Create the project environment first: python -m venv .venv
  exit /b 1
)
set RAWPY_ARGS=
".venv\Scripts\python.exe" -c "import rawpy" >nul 2>nul
if not errorlevel 1 set RAWPY_ARGS=--collect-all rawpy
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --windowed --name NegativeLab --paths src %RAWPY_ARGS% --collect-all tifffile main.py
if errorlevel 1 exit /b 1
echo Portable build created in dist\NegativeLab
