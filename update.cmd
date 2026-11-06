@echo off
setlocal
pushd "%~dp0"
"%~dp0.venv312\Scripts\python.exe" -m src.database update %*
set "update_result=%errorlevel%"
popd
exit /b %update_result%
