@echo off
cd /d "%~dp0"
py -3 scripts\status.py %* 2>nul || python scripts\status.py %*
pause
