import sys

try:
    from config.server_config import ServerSettings
    server_settings = ServerSettings()
except ImportError:
    server_settings = None

try:
    from config.client_config import ClientSettings
    client_settings = ClientSettings()
except ImportError:
    client_settings = None
