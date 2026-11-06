@echo off
setlocal
pushd "%~dp0.."
"%~dp0..\.venv312\Scripts\python.exe" -m model_tests.compare_text_models %*
set "comparison_result=%errorlevel%"
popd
exit /b %comparison_result%
