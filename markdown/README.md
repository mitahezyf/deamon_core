# DAEMON Project

Wstepne GUI webowe i backend FastAPI dla lokalnego asystenta glosowego.

## Co jest teraz

- FastAPI backend z endpointami:
  - `GET /health`
  - `GET /status`
  - `GET /config/public`
  - `POST /synthesize`
  - `WS /ws/synthesize`
- GUI webowe serwowane z backendu:
  - `GET /`
  - statyczne pliki pod `GET /static/*`
- Sekcje integracyjne pod przyszle moduly:
  - Ears (wake word + STT)
  - Brain (LLM)
  - Memory (RAG)
  - Tools (web/function-calling)

## Szybki start

```powershell
python -m app.api.main
```

Nastepnie otworz:

- `http://localhost:8000/` - GUI
- `http://localhost:8000/docs` - Swagger

## Testy

```powershell
pytest -q
```

## Jak podpinac kolejne moduly

1. Dodaj modul w `app/core/` (np. `ears.py`, `brain.py`).
2. Dodaj statusy i konfiguracje do endpointow `GET /status` i `GET /config/public`.
3. Dodaj eventy i akcje po stronie GUI (`app/web/static/js/app.js`).
4. Dodaj testy API w `tests/api/routes/`.
