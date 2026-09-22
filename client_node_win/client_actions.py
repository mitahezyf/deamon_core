import re
import os
import subprocess
import ctypes
import psutil
import datetime
import urllib.request
import json
import logging
from typing import Tuple, Optional

import sys
from pathlib import Path
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import client_settings

log = logging.getLogger("client.actions")

VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF

def press_key(vk_code):
    try:
        ctypes.windll.user32.keybd_event(vk_code, 0, 0, 0)
        ctypes.windll.user32.keybd_event(vk_code, 0, 2, 0)
    except Exception:
        pass

class ActionExecutor:
    def __init__(self):
        self.action_matrix = [
            (re.compile(r"^(podgłośnij|głośniej)"), self.vol_up),
            (re.compile(r"^(ścisz|ciszej)"), self.vol_down),
            (re.compile(r"^(wycisz|zmutuj)"), self.vol_mute),
            (re.compile(r"^(odcisz)"), self.vol_unmute),
            (re.compile(r"^(otwórz|włącz|uruchom)\s+(przeglądarkę|chrome|edge)"), self.open_browser),
            (re.compile(r"^(otwórz|włącz|uruchom)\s+(kalkulator)"), self.open_calc),
            (re.compile(r"^(otwórz|włącz|uruchom)\s+(notatnik)"), self.open_notepad),
            (re.compile(r"^(otwórz|włącz|uruchom)\s+(menedżer|zadania)"), self.open_taskmgr),
            (re.compile(r"^(otwórz|włącz|uruchom)\s+(terminal|powershell|konsole)"), self.open_powershell),
            (re.compile(r"^(zablokuj)\s+(komputer|system)"), self.lock_system),
            (re.compile(r"^(która godzina|podaj czas)"), self.get_time),
            (re.compile(r"^(jaka jest data|który dzisiaj)"), self.get_date),
            (re.compile(r"^(stan systemu|użycie cpu|użycie ram)"), self.get_sys_status),
        ]
        
    def vol_up(self):
        for _ in range(5): press_key(VK_VOLUME_UP)
        return "Zwiększyłem głośność."

    def vol_down(self):
        for _ in range(5): press_key(VK_VOLUME_DOWN)
        return "Zmniejszyłem głośność."

    def vol_mute(self):
        press_key(VK_VOLUME_MUTE)
        return "Wyciszyłem system."
        
    def vol_unmute(self):
        press_key(VK_VOLUME_MUTE)
        return "Odciszyłem system."

    def open_browser(self):
        os.system("start https://www.google.com")
        return "Przeglądarka została uruchomiona."

    def open_calc(self):
        os.system("start calc")
        return "Otwieram kalkulator."

    def open_notepad(self):
        os.system("start notepad")
        return "Uruchamiam notatnik."

    def open_taskmgr(self):
        os.system("start taskmgr")
        return "Menedżer zadań gotowy."
        
    def open_powershell(self):
        os.system("start powershell")
        return "Otwieram środowisko powershell."

    def lock_system(self):
        os.system("rundll32.exe user32.dll,LockWorkStation")
        return "Zablokowałem stację roboczą."

    def get_time(self):
        now = datetime.datetime.now().strftime("%H:%M")
        return f"Teraz jest godzina {now}."

    def get_date(self):
        # Spolszczenie daty na szybko
        months = ["", "stycznia", "lutego", "marca", "kwietnia", "maja", "czerwca", "lipca", "sierpnia", "września", "października", "listopada", "grudnia"]
        now = datetime.datetime.now()
        return f"Dzisiaj jest {now.day} {months[now.month]} {now.year} roku."

    def get_sys_status(self):
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory().percent
        return f"Użycie procesora to {cpu} procent, a pamięci RAM {mem} procent."

    def match_regex(self, text: str) -> Optional[str]:
        text_lower = text.lower().strip()
        for pattern, func in self.action_matrix:
            if pattern.search(text_lower):
                log.info(f"ActionExecutor [Tier 0]: Dopasowano regex -> {func.__name__}")
                return func()
        return None

    def match_semantic(self, text: str) -> Optional[str]:
        sys_prompt = (
            "Jesteś klasyfikatorem intencji systemowych Windows. Poniżej lista dostępnych funkcji:\n"
            "vol_up, vol_down, vol_mute, vol_unmute, open_browser, open_calc, open_notepad, open_taskmgr, "
            "open_powershell, lock_system, get_time, get_date, get_sys_status.\n"
            "Dopasuj intencję użytkownika do jednej z funkcji. Zwróć WYŁĄCZNIE nazwę funkcji bez żadnego tekstu. "
            "Jeśli nie pasuje żadna, zwróć słowo UNKNOWN."
        )
        payload = {
            "model": client_settings.model_router,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": text}
            ],
            "options": {"temperature": 0.0, "num_predict": 10},
            "stream": False
        }
        url = f"{client_settings.daemon_ollama_host}/api/chat"
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'),
                                         headers={'Content-Type': 'application/json'}, method='POST')
            with urllib.request.urlopen(req, timeout=1.5) as response:
                resp_json = json.loads(response.read().decode('utf-8'))
                if "message" in resp_json and "content" in resp_json["message"]:
                    func_name = resp_json["message"]["content"].strip()
                    if hasattr(self, func_name) and callable(getattr(self, func_name)):
                        log.info(f"ActionExecutor [Tier 1]: Dopasowano semantycznie -> {func_name}")
                        return getattr(self, func_name)()
        except Exception as e:
            log.warning(f"Błąd podczas fallbacku Tier 1 (Ollama): {e}")
        return None

    def execute(self, text: str) -> Tuple[bool, str]:
        # Tier 0 - szybki
        res = self.match_regex(text)
        if res:
            return True, res
            
        # Tier 1 - semantyczny LLM
        res = self.match_semantic(text)
        if res:
            return True, res
            
        # Jeśli nic
        return False, ""
