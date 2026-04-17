# DAEMON — Dokumentacja Struktury Projektu

> Lokalny, prywatny asystent AI sterowany głosem. Architektura klient-serwer.
> Ostatnia aktualizacja: Faza 1 + 2

---

## Mapa katalogów

```
DAEMON_PROJECT/
│
├── app/                        # cały kod źródłowy aplikacji
│   ├── api/                    # serwer FastAPI (backend HTTP + WebSocket)
│   │   ├── __init__.py
│   │   ├── main.py             # punkt wejścia serwera, ładowanie modeli
│   │   ├── schemas.py          # typy danych requestów i odpowiedzi (Pydantic)
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── http.py         # endpointy REST: /health, /synthesize
│   │       └── ws.py           # endpoint WebSocket: /ws/synthesize (streaming audio)
│   │
│   └── core/                   # silniki systemu (TTS, STT, LLM, pamięć)
│       ├── __init__.py
│       ├── config.py           # centralna konfiguracja (plik .env)
│       ├── logger.py           # centralny logger z poziomami DEBUG/INFO/WARNING/ERROR
│       └── vox.py              # silnik TTS — klasa DaemonVox (XTTS v2)
│
├── scripts/
│   └── build_voice_cache.py    # przebuduj cache embeddingów głosu
│
├── models/
│   └── wake_word/              # tu wrzucić .ppn z Picovoice (lub .onnx z openWakeWord)
│
├── voice_samples/              # próbki głosu .wav do klonowania
├── bin/                        # ffmpeg i biblioteki audio (DLLs)
├── tests/                      # wygenerowane pliki .wav (tymczasowe)
├── data/                       # dane runtime — ChromaDB i SQLite (gitignored)
│
├── .env                        # twoja konfiguracja (gitignored — nie commituj!)
├── .env.example                # szablon konfiguracji (commitowany)
├── pyproject.toml              # zależności i konfiguracja projektu
├── .pre-commit-config.yaml     # linting, formatowanie, bezpieczeństwo
├── .gitignore
└── daemon_voice_cache.pth      # cache embeddingów głosu (gitignored)
```

---

## Opis każdego pliku

### `app/core/config.py` — Centralna konfiguracja

**Co robi:** Jeden plik, który trzyma *wszystkie* ustawienia systemu. Czyta je z pliku `.env` i udostępnia reszcie kodu jako obiekt `settings`.

**Jak używać:**
```python
from app.core.config import settings

print(settings.llm_model)   # nazwa modelu Ollama
print(settings.api_port)    # port serwera
print(settings.debug_mode)  # czy DEBUG włączony
```

**Kluczowe pola:**

| Zmienna `.env` | Pole | Domyślna wartość | Opis |
|---|---|---|---|
| `DAEMON_LLM_MODEL` | `llm_model` | `huihui_ai/qwen3.5-abliterated:9b` | Model Ollama |
| `DAEMON_OLLAMA_URL` | `ollama_url` | `http://localhost:11434` | Adres serwera Ollama |
| `DAEMON_LANGUAGE` | `language` | `pl` | Język syntezy i transkrypcji |
| `DAEMON_WHISPER_MODEL` | `whisper_model` | `large-v3` | Rozmiar modelu Whisper |
| `DAEMON_API_HOST` | `api_host` | `0.0.0.0` | Interfejs serwera (LAN) |
| `DAEMON_API_PORT` | `api_port` | `8000` | Port serwera |
| `DAEMON_DEBUG_MODE` | `debug_mode` | `false` | Włącza logi DEBUG |
| `DAEMON_PORCUPINE_ACCESS_KEY` | `porcupine_access_key` | _(puste)_ | Klucz z console.picovoice.ai |

> **Ważne:** Ścieżki (`samples_dir`, `cache_path`, itp.) są ustawiane automatycznie na podstawie lokalizacji pliku `config.py`. Nie trzeba ich podawać w `.env` — chyba że chcesz nadpisać.

---

### `app/core/logger.py` — Centralny Logger

**Co robi:** Jeden punkt konfiguracji logowania dla całego projektu. Każdy moduł tworzy swój logger przez `get_logger("nazwa")`, który jest potomkiem głównego loggera `daemon`.

**Jak używać:**
```python
from app.core.logger import get_logger

log = get_logger("moj_modul")

log.debug("Szczegóły debugowania — widoczne tylko gdy DAEMON_DEBUG_MODE=true")
log.info("Normalna informacja — widoczna zawsze")
log.warning("Coś nienormalnego, ale nie błąd")
log.error("Błąd — coś poszło nie tak", exc_info=True)
```

**Format logu:**
```
2026-04-17 01:30:00 [INFO    ] daemon.vox               — Silnik głosu gotowy.
2026-04-17 01:30:01 [DEBUG   ] daemon.api.ws            — Chunk #3 wygenerowany (2048 próbek)
2026-04-17 01:30:02 [WARNING ] daemon.vox               — FP16 niedostępne: ...
2026-04-17 01:30:05 [ERROR   ] daemon.api.ws            — Błąd WebSocket: ...
```

**Wyciszanie zewnętrznych bibliotek:** Automatycznie wycisza hałaśliwe logi z `uvicorn.access`, `TTS`, `numba`, `urllib3`, `httpx` (pokazują tylko WARNING i wyżej).

---

### `app/core/vox.py` — Silnik TTS (DaemonVox)

**Co robi:** Klasa `DaemonVox` enkapsuluje cały silnik głosu XTTS v2. Ładuje model raz przy starcie i pozwala generować mowę na żądanie — bez ponownego ładowania.

**Cykl życia:**
```python
vox = DaemonVox()
vox.load()    # ładuje XTTS v2 + cache embeddingów (kilka sekund)
vox.warmup()  # rozgrzewa CUDA (eliminuje opóźnienie JIT przy pierwszym żądaniu)

# generowanie — instant po warmupie
for chunk in vox.stream_chunks("Witaj, Daemon gotowy."):
    # każdy chunk to numpy float32 PCM — wysyłasz go od razu
    pass

# lub zapisz do pliku:
vox.synthesize_to_file("Tekst", Path("output.wav"))

# po dodaniu nowych próbek do voice_samples/:
vox.rebuild_cache()
```

**Metody publiczne:**

| Metoda | Opis |
|---|---|
| `load()` | Ładuje model XTTS v2 i cache embeddingów głosu |
| `warmup()` | Rozgrzewa CUDA — eliminuje opóźnienie JIT |
| `stream_chunks(text)` | Generator chunków PCM — używaj do WebSocket |
| `synthesize_to_file(text, path)` | Zapisuje WAV, zwraca statystyki latencji |
| `rebuild_cache()` | Przebudowuje `.pth` z próbek w `voice_samples/` |

---

### `app/api/main.py` — Serwer FastAPI

**Co robi:** Punkt wejścia serwera. Tworzy aplikację FastAPI, rejestruje routery i zarządza cyklem życia modeli przez `lifespan`.

**Uruchomienie:**
```bash
python -m app.api.main
# lub przez uvicorn bezpośrednio:
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

**Kluczowa logika — `lifespan`:** Ładuje `DaemonVox` *raz* przy starcie i przechowuje w `app.state.vox`. Dzięki temu każdy request ma dostęp do gotowego modelu bez ponownego ładowania.

**CORS:** Skonfigurowany na `allow_origins=["*"]` — pozwala na połączenia z GUI w sieci LAN.

---

### `app/api/schemas.py` — Schematy danych

**Co robi:** Definiuje typy danych requestów i odpowiedzi API (Pydantic). FastAPI używa ich do automatycznej walidacji i generowania dokumentacji.

| Klasa | Typ | Opis |
|---|---|---|
| `SynthesizeRequest` | Request | `text` (string), `output` (nazwa pliku WAV) |
| `SynthesizeResponse` | Response | `latency_first_chunk`, `total_time`, `audio_duration`, `output` |
| `HealthResponse` | Response | `status`, `vox_loaded`, `device`, `llm_model`, `api_port` |

---

### `app/api/routes/http.py` — Endpointy REST

**Co robi:** Dwa klasyczne endpointy HTTP.

| Endpoint | Metoda | Opis |
|---|---|---|
| `/health` | GET | Stan serwera — czy modele załadowane, jaki device, jaki model LLM |
| `/synthesize` | POST | Synteza mowy do pliku WAV, zwraca metadane latencji |

**Jak testować:**
```bash
# sprawdź stan serwera
curl http://localhost:8000/health

# wygeneruj mowę
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Witaj, mówi Daemon.", "output": "test.wav"}'
```

---

### `app/api/routes/ws.py` — WebSocket Streaming Audio

**Co robi:** Endpoint WebSocket do streamowania audio w czasie rzeczywistym. Klient wysyła tekst, serwer odsyła chunki PCM natychmiast po wygenerowaniu każdego — bez czekania na całość.

**Protokół:**
```
Klient → serwer: tekst (string UTF-8)
Serwer → klient: N × [4 bajty: długość] + [dane PCM float32]
Serwer → klient: [4 bajty: 0x00000000]  ← sygnał końca strumienia
```

**Adres:** `ws://localhost:8000/ws/synthesize`

---

### `scripts/build_voice_cache.py` — Przebudowanie Cache Głosu

**Co robi:** Zamiennik starego `tuning.py`. Używa klasy `DaemonVox` do przebudowania pliku `daemon_voice_cache.pth` z próbek w `voice_samples/`.

**Kiedy używać:** Po dodaniu nowych plików `.wav` do katalogu `voice_samples/`.

```bash
python scripts/build_voice_cache.py
```

---

### `voice_samples/` — Próbki Głosu

Pliki `.wav` używane do klonowania głosu przez XTTS v2. Im więcej i lepszej jakości, tym lepszy efekt. Wszystkie próbki są łączone w jeden embedding (`.pth`).

> **Wymagania:** próbki mono, 22050 Hz lub 24000 Hz, bez szumów w tle.

---

### `models/wake_word/` — Model Wake Word

Tu wrzucić plik modelu wake word po wytrenowaniu:
- `daemon_windows.ppn` — z Picovoice Porcupine (console.picovoice.ai)

---

### `.env` — Konfiguracja Lokalna

Twój prywatny plik konfiguracyjny. **Nigdy nie commituj go do git** (jest w `.gitignore`). Wzór w `.env.example`.

---

### `pyproject.toml` — Zależności Projektu

Wszystkie zależności Python projektu. Instalacja:
```bash
pip install -e .
# lub tylko zależności bez instalacji pakietu:
pip install -r requirements.txt  # (jeśli wygenerujesz z pyproject)
```

---

## Przepływ danych (obecny stan — Faza 2)

```
                    ┌─────────────────────────────┐
                    │        app/api/main.py      │
                    │  (FastAPI + lifespan)        │
                    │                              │
                    │  app.state.vox = DaemonVox   │
                    └──────────┬──────────────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                                   │
    ┌────────▼────────┐               ┌──────────▼──────────┐
    │ /health         │               │ /ws/synthesize      │
    │ /synthesize     │               │ WebSocket           │
    │ (http.py)       │               │ (ws.py)             │
    └────────┬────────┘               └──────────┬──────────┘
             │                                   │
             └─────────────┬─────────────────────┘
                           │
                  ┌────────▼────────┐
                  │ app/core/vox.py │
                  │  DaemonVox      │
                  │  XTTS v2 GPU    │
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │ app/core/config.py│
                  │  .env           │
                  └─────────────────┘
```

---

## Następne kroki (Faza 3 i dalej)

| Faza | Co zostanie dodane |
|---|---|
| **Faza 3** | `app/core/ears.py` — Porcupine wake word + Faster-Whisper STT |
| **Faza 4** | `app/core/brain.py` — klient Ollama / `app/core/memory.py` — ChromaDB + SQLite / `app/core/tools.py` — DuckDuckGo |
| **Faza 5** | `gui/` — interfejs React (VoiceOrb, historia rozmowy, status) |
| **Faza 6** | Integracja end-to-end, testy, weryfikacja LAN |
