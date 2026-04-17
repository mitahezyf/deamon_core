from app.core.config import DaemonSettings


def test_settings_defaults():
    # Test czy domyslne sciezki i wartosci sa ustawiane poprawnie
    settings = DaemonSettings(_env_file=None)
    assert settings.api_port == 8000
    assert settings.api_host == "0.0.0.0"
    assert settings.language == "pl"

    # Sprawdzenie auto-wykrywania sciezek
    assert settings.samples_dir is not None
    assert settings.samples_dir.name == "voice_samples"

    assert settings.cache_path is not None
    assert settings.cache_path.name == "daemon_voice_cache.pth"


def test_settings_overrides(monkeypatch):
    # Test czy zmienne srodowiskowe nadpisuja domyslne wartosci
    monkeypatch.setenv("DAEMON_API_PORT", "9000")
    monkeypatch.setenv("DAEMON_DEBUG_MODE", "true")
    monkeypatch.setenv("DAEMON_LANGUAGE", "en")

    settings = DaemonSettings(_env_file=None)
    assert settings.api_port == 9000
    assert settings.debug_mode is True
    assert settings.language == "en"
