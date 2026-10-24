@echo off
REM Start Talos listening for its name. This is what the desktop shortcut runs;
REM make the shortcut with:  run scripts\make_shortcut.py
REM
REM On a crash the window stays open, because a window that closes on its own
REM error message tells nobody anything.

title Talos
call "%~dp0run.cmd" main.py --wake %*
if errorlevel 1 (
    echo.
    echo Talos stopped with an error. The message above says why.
    pause
)
