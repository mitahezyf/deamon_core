@echo off
:: Skrypt startowy serwera DAEMON
:: Uruchom dwuklikiem lub przez: start.bat

cd /d "%~dp0"
echo Uruchamianie serwera DAEMON...
echo Dostepny pod: http://localhost:8000
echo Dokumentacja API: http://localhost:8000/docs
echo.
echo Zatrzymaj przez: Ctrl+C
echo.

.venv\Scripts\python -m app.api.main
pause
