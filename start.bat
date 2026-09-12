@echo off
cd /d "%~dp0"
echo goldbot: MetaTrader 5 muss offen und im Swissquote-DEMO-Konto eingeloggt sein.
py -3 scripts\bootstrap.py %*
if errorlevel 1 python scripts\bootstrap.py %*
pause
