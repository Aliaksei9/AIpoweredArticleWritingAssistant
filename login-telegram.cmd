@echo off
setlocal
pushd "%~dp0"
"%~dp0.venv312\Scripts\python.exe" -m src.database login --source TG %*
set "login_result=%errorlevel%"
popd
exit /b %login_result%
