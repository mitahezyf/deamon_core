import asyncio
from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import ServiceUnavailable, AuthError
from config import server_settings as settings
from app.core.logger import get_logger

log = get_logger("database")

class MemgraphClient:
    def __init__(self):
        self.uri = settings.memgraph_uri
        self.driver: AsyncDriver | None = None
        self._connected = False

    async def connect(self, retries: int = 5, delay: int = 2) -> None:
        """Nawiązuje asynchroniczne połączenie z bazą Memgraph, z ponawianiem prób."""
        for attempt in range(1, retries + 1):
            try:
                log.info(f"Próba połączenia z Memgraph ({self.uri}) - próba {attempt}/{retries}")
                self.driver = AsyncGraphDatabase.driver(self.uri, auth=None)
                await self.driver.verify_connectivity()
                self._connected = True
                log.info("Udało się nawiązać połączenie z bazą Memgraph.")
                return
            except (ServiceUnavailable, AuthError, Exception) as e:
                log.warning(f"Błąd połączenia z Memgraph: {e}")
                if attempt < retries:
                    log.info(f"Ponawianie próby za {delay} sekund...")
                    await asyncio.sleep(delay)
                else:
                    log.error("Nie udało się połączyć z Memgraph po kilku próbach.")
                    self._connected = False
                    self.driver = None

    async def verify_connectivity(self) -> bool:
        """Zwraca True, jeśli połączenie jest aktywne."""
        if not self.driver or not self._connected:
            return False
        try:
            await self.driver.verify_connectivity()
            return True
        except Exception:
            return False

    async def close(self):
        """Zamyka asynchronicznego drivera neo4j."""
        if self.driver:
            await self.driver.close()
            self._connected = False
            log.info("Zakończono połączenie z Memgraph.")
