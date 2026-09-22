@echo off
echo =========================================
echo DAEMON Klient - Start (Windows 11)
echo =========================================

echo Sprawdzanie procesu Ollamy...
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if "%ERRORLEVEL%"=="1" (
    echo [OSTRZEZENIE] Proces ollama.exe nie dziala. Uruchamiam Ollame...
    start "" "ollama" app
    timeout /t 5 /nobreak
) else (
    echo [OK] Ollama dziala w tle.
)

echo Aktywowanie venv_win...
if not exist "venv_win\Scripts\activate.bat" (
    echo [BLAD] Nie znaleziono srodowiska venv_win!
    pause
    exit /b 1
)

call venv_win\Scripts\activate.bat

echo Uruchamianie klienta DAEMON...
set PYTHONPATH=.
python client_node_win\client_main.py

pause
