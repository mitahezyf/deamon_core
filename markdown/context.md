# 🧠 PROJECT ANTIGRAVITY | DAEMON
**Opis:** Lokalny, multimodalny asystent AI (Agent) operujący na architekturze klient-serwer.
**Cel główny:** Stworzenie w 100% prywatnego bytu AI sterowanego głosem, z dostępem do sieci, wizją, pamięcią długotrwałą i interfejsem webowym dostępnym w sieci lokalnej (LAN).

## 🛠️ STOS TECHNOLOGICZNY (Tech Stack)
* **Hardware:** NVIDIA RTX 3090 (24GB VRAM) - wszystkie modele muszą być zarządzane tak, aby nie przekroczyć limitu VRAM.
* **Środowisko:** Python 3.11, CUDA 12.1, wirtualne środowisko (`.venv`).
* **Mózg (LLM & Vision):** Qwen (wersja Abliterated z funkcją Vision) uruchomiony lokalnie (np. via Ollama / vLLM / llama.cpp).
* **Uszy (STT & Nasłuchiwanie):** * *Wake Word:* `openWakeWord` (Nasłuchiwanie na słowo kluczowe "Daemon" - ultra lekkie, działa na CPU).
    * *Transkrypcja:* `Faster-Whisper` (aktywowany tylko po usłyszeniu Wake Wordu, by oszczędzać zasoby).
* **Usta (TTS):** Coqui XTTS v2. (Zoptymalizowany: używamy pliku wektorowego `.pth` wyciągniętego z czystych próbek audio, by wyeliminować opóźnienie "zimnego startu" i naśladować naturalną intonację).
* **Pamięć Długotrwała (Memory & RAG):** * Wektorowa baza danych (np. `ChromaDB` / `Qdrant`) do pamięci semantycznej (system połączeń i relacji z poprzednich rozmów).
    * Baza SQLite do płaskich logów i konfiguracji.
* **Oczy na świat (Tools & Web):** * `duckduckgo-search` / `BeautifulSoup` zaimplementowane jako system *Function Calling*. Daemon sam decyduje, kiedy musi przeszukać internet.
* **Interfejs i Komunikacja (Backend/GUI):** * *Backend:* `FastAPI` obsługujący WebSockety (niezbędne do streamowania audio w czasie rzeczywistym).
    * *Frontend (GUI):* Lekki interfejs webowy (HTML/JS/React lub Gradio/Streamlit), uruchamiany na porcie `0.0.0.0`, dostępny dla innych urządzeń w sieci lokalnej (LAN).

## ⚙️ ARCHITEKTURA PRZEPŁYWU (Flow)
1.  **Standby:** Skrypt GUI/Backend działa w tle. Przeglądarka (mikrofon) streamuje małe paczki audio do serwera.
2.  **Trigger:** `openWakeWord` wychwytuje słowo "Daemon".
3.  **STT:** `Faster-Whisper` tłumaczy zapytanie użytkownika na tekst.
4.  **Retrieval:** Zapytanie leci do bazy wektorowej (ChromaDB), by wyciągnąć kontekst z poprzednich rozmów.
5.  **Processing:** LLM (Qwen) otrzymuje Prompt + Kontekst z pamięci. LLM decyduje, czy musi użyć narzędzia (np. szukaj w sieci, odczytaj obraz), czy od razu generuje odpowiedź.
6.  **Streaming TTS:** Odpowiedź tekstowa jest cięta na zdania i wysyłana do XTTS. 
7.  **Playback:** Dźwięk jest natychmiast odtwarzany w GUI (WebSockets), bez czekania na wygenerowanie całości. W tym samym czasie tekst pojawia się na ekranie.

## 🚨 OBECNE WYZWANIA / STATUS
* [x] Środowisko Python 3.11 i CUDA skonfigurowane.
* [x] XTTS v2 działa i generuje polski głos z próbki referencyjnej.
* [ ] Przejście XTTS z ładowania pliku `.wav` na ładowanie wyekstrahowanego "odcisku głosu" `.pth` (Speaker Embedding) dla przyspieszenia.
* [ ] Złożenie szkieletu FastAPI pod Web-GUI.
* [ ] Implementacja `openWakeWord` i `Faster-Whisper`.
* [ ] Podpięcie LLM (Qwen) z systemem RAG (Pamięć).
