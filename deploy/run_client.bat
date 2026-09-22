@echo off
echo =========================================
echo DAEMON Klient - Start (Windows 11)
echo =========================================

:: Wymuszenie adresu nasluchu sieciowego dla Ollamy na 0.0.0.0:11434 (dostep z wezla LXC i LAN)
set "OLLAMA_HOST=0.0.0.0:11434"

echo Restartowanie procesu Ollamy z nowym adresem nasluchu 0.0.0.0:11434...
taskkill /F /IM ollama.exe /T 2>NUL
taskkill /F /IM ollama_app.exe /T 2>NUL

echo Uruchamianie ollama serve...
start "" ollama serve
timeout /t 3 /nobreak >NUL

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
