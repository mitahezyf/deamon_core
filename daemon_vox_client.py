import json
import logging
import socket
import struct
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] daemon_vox_client - %(message)s",
)
log = logging.getLogger("daemon_vox_client")

# Klient do komunikacji z daemon_vox.py --server
# Użycie: python daemon_vox_client.py "tekst do wypowiedzenia"

HOST = "127.0.0.1"
PORT = 59721


def wyslij_request(text: str, output: str = "daemon_final.wav") -> dict:
    req = json.dumps({"text": text, "output": output}).encode("utf-8")
    t0 = time.perf_counter()

    with socket.create_connection((HOST, PORT), timeout=30) as s:
        s.sendall(struct.pack(">I", len(req)) + req)
        raw_len = s.recv(4)
        msg_len = struct.unpack(">I", raw_len)[0]
        data = b""
        while len(data) < msg_len:
            chunk = s.recv(msg_len - len(data))
            if not chunk:
                break
            data += chunk

    elapsed_total = time.perf_counter() - t0
    result = json.loads(data.decode("utf-8"))
    result["client_rtt"] = round(elapsed_total, 3)
    return result


def main():
    if len(sys.argv) < 2:
        log.error('Uzycie: python daemon_vox_client.py "tekst do wypowiedzenia"')
        log.error("Serwer musi byc uruchomiony: python daemon_vox.py --server")
        sys.exit(1)

    text = " ".join(sys.argv[1:])
    log.info("Wysylam do daemona: %s...", text[:60])

    try:
        result = wyslij_request(text)
        log.info("Latencja 1. chunk: %ss", result["latency_first_chunk"])
        log.info("Czas inference: %ss", result["total_time"])
        log.info("Czas audio: %ss", result["audio_duration"])
        log.info("RTT klient: %ss", result["client_rtt"])
        log.info("Plik: %s", result["output"])
    except ConnectionRefusedError:
        log.error("Brak polaczenia z serwerem na %s:%s", HOST, PORT)
        log.error("Uruchom najpierw: python daemon_vox.py --server")
        sys.exit(1)


if __name__ == "__main__":
    main()
