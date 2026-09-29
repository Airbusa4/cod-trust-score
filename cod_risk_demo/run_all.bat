@echo off
rem Runs the whole COD Risk Score demo end to end (synthetic data, for illustration only).
setlocal
cd /d "%~dp0"
set PY=python
if exist .venv\Scripts\python.exe set PY=.venv\Scripts\python.exe
set PYTHONIOENCODING=utf-8

%PY% -m src.generate_data || exit /b 1
%PY% -m src.build_features || exit /b 1
%PY% -m src.validate_data > nul || exit /b 1
%PY% -m src.train || exit /b 1
%PY% -m src.evaluate || exit /b 1
%PY% -m src.decide || exit /b 1
%PY% -m src.explain || exit /b 1
%PY% -m src.build_demo || exit /b 1
%PY% -m src.export_app || exit /b 1
%PY% -m src.summary || exit /b 1
