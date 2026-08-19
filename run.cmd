@echo off
REM Run a Talos script in the project venv, from anywhere, in any shell.
REM
REM   run scripts\test_mic.py
REM   run scripts\test_stt.py --cpu
REM
REM Calls the venv interpreter directly rather than activating it: activation
REM syntax differs between cmd and PowerShell, and getting it wrong is the
REM first thing that goes wrong for anyone setting this up.

setlocal
set "VENV_PYTHON=%USERPROFILE%\.venvs\talos\Scripts\python.exe"

REM The console here is cp1254, so anything printing an em dash or an
REM emoji dies with UnicodeEncodeError rather than printing it badly.
REM Torch's ONNX exporter prints a tick on success, which is a strange
REM way to lose a trained model.
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

if not exist "%VENV_PYTHON%" (
    echo No virtual environment at %VENV_PYTHON%
    echo.
    echo Create it with:
    echo     py -3.12 -m venv "%%USERPROFILE%%\.venvs\talos"
    echo     "%%USERPROFILE%%\.venvs\talos\Scripts\python.exe" -m pip install -r requirements.txt
    exit /b 1
)

if "%~1"=="" (
    echo Usage: run ^<script^> [arguments]
    echo.
    echo     run scripts\check_env.py
    echo     run scripts\test_mic.py
    echo     run scripts\test_stt.py
    echo     run scripts\test_tts.py
    echo     run scripts\test_llm.py
    exit /b 1
)

REM %~dp0 is this file's directory, so the working directory is the project
REM root regardless of where the caller happened to be.
pushd "%~dp0"
"%VENV_PYTHON%" %*
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
