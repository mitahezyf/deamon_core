# DAEMON_PROJECT

Lokalny asystent glosowy oparty o pipeline:

- STT (faster-whisper)
- LLM (Ollama)
- TTS (XTTS)
- Wake word (openwakeword)

## Uruchomienie

```powershell
Set-Location "K:\DAEMON_PROJECT"
.\start.bat
```

albo:

```powershell
Set-Location "K:\DAEMON_PROJECT"
.\.venv\Scripts\python -m app.api.main
```

## Konfiguracja

Podstawowe ustawienia sa w `.env`.

Przyklad kluczowych zmiennych:

```dotenv
DAEMON_LLM_MODEL=qwen3.5:4b
DAEMON_OLLAMA_URL=http://localhost:11434
DAEMON_STT_ENABLED=true
DAEMON_WHISPER_MODEL=large-v3
DAEMON_MODEL_STORE_DIR=.model_store
```

## Modele i storage

- Modele Ollama pozostaja pod kontrola Ollama (np. `K:\OllamaModels`).
- Cache STT/TTS jest trzymany lokalnie w `.model_store` i ignorowany przez git.
- Repo zawiera tylko `.model_store/.gitkeep`.

Struktura lokalna:

- `.model_store/huggingface`
- `.model_store/tts`

## Migracja modeli z C: do repo

Skrypt migracyjny:

```powershell
Set-Location "K:\DAEMON_PROJECT"
.\scripts\move_model_cache_to_repo.ps1
```

Tryb podgladu:

```powershell
Set-Location "K:\DAEMON_PROJECT"
.\scripts\move_model_cache_to_repo.ps1 -DryRun
```

Domyslnie skrypt wykrywa katalog repo wzgledem lokalizacji skryptu.

## Jakosc i testy

```powershell
Set-Location "K:\DAEMON_PROJECT"
python -m pytest tests -q --no-cov
```

```powershell
Set-Location "K:\DAEMON_PROJECT"
pre-commit run --all-files
```

## Uwagi operacyjne

- Glowny wspierany runtime to `app.api.main`.
- Skrypty `daemon_vox.py` i `tuning.py` sa legacy (zgodnosc wsteczna).
- Nie commituj zawartosci `.model_store`.
