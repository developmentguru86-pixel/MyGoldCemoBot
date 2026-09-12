@echo off
cd /d "%~dp0"
py -3 scripts\run_telegram.py 2>nul || python scripts\run_telegram.py
pause
