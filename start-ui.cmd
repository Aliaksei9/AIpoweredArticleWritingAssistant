@echo off
setlocal
pushd "%~dp0"
"%~dp0.venv312\Scripts\python.exe" -m src.ui %*
set "ui_result=%errorlevel%"
popd
exit /b %ui_result%
