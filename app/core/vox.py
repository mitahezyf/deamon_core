import time
import warnings
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torchaudio
from TTS.api import TTS

from app.core.config import settings
from app.core.logger import get_logger

# wycisza FutureWarning i UserWarning z zewnetrznych bibliotek (TTS, transformers)
# to ich wewnetrzny deprecated kod i nie naprawiamy go po stronie aplikacji
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="TTS")
warnings.filterwarnings("ignore", category=UserWarning, module="TTS")
warnings.filterwarnings("ignore", message=".*weights_only.*")
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*")

log = get_logger("vox")

# optymalizacje CUDA musza byc ustawione przed ladowaniem modelu
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# parametry syntezy dostosowane pod naturalna polska mowe
_SYNTH_PARAMS: dict = dict(
    temperature=0.75,
    length_penalty=1.0,
    repetition_penalty=5.0,
    top_k=50,
    top_p=0.85,
    speed=0.95,
    enable_text_splitting=True,
)

_SAMPLE_RATE = 24_000


class DaemonVox:
    # silnik TTS oparty na XTTS v2 z cachem embeddingow glosu

    def __init__(self) -> None:
        self._tts: TTS | None = None
        self._gpt_cond_latent = None
        self._speaker_embedding = None
        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"
        log.debug("DaemonVox zainicjalizowany, urządzenie: %s", self._device)

    # --- publiczne API ---

    def load(self) -> None:
        # laduje model i cache glosu, wywolac raz przy starcie serwera
        log.info("Ładowanie silnika głosu...")
        self._tts = self._zaladuj_model()
        probki = self._zbierz_probki()
        self._gpt_cond_latent, self._speaker_embedding = self._zaladuj_lub_zbuduj_cache(
            probki
        )
        log.info("Silnik głosu gotowy.")

    def warmup(self) -> None:
        # rozgrzewka CUDA, pierwsze wywolanie jest wolniejsze przez JIT kerneli
        self._assert_loaded()
        log.info("Rozgrzewanie CUDA (warmup)...")
        model = self._tts.synthesizer.tts_model  # type: ignore[union-attr]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for _ in model.inference_stream(
                text="Test.",
                language=settings.language,
                gpt_cond_latent=self._gpt_cond_latent,
                speaker_embedding=self._speaker_embedding,
                **_SYNTH_PARAMS,
            ):
                pass
        log.info("Warmup zakonczony - serwer gotowy.")

    def rebuild_cache(self) -> None:
        # przebudowuje cache embeddingow ze wszystkich probek
        log.info("Przebudowywanie cache embeddingów głosu...")
        if self._tts is None:
            self._tts = self._zaladuj_model()
        probki = self._zbierz_probki()
        self._zbuduj_cache(probki)
        # przeladowuje embeddingi do pamieci
        dane = torch.load(
            settings.cache_path,
            map_location=self._tts.synthesizer.tts_model.device,  # type: ignore[union-attr]
            weights_only=True,
        )
        self._gpt_cond_latent = dane["gpt_cond_latent"]
        self._speaker_embedding = dane["speaker_embedding"]
        log.info("Cache przebudowany i załadowany.")

    def stream_chunks(self, text: str) -> Iterator[np.ndarray]:
        # generator zwraca kolejne chunki PCM (numpy float32)
        # to pozwala odtwarzac audio zanim caly tekst zostanie wygenerowany
        self._assert_loaded()
        log.debug("Rozpoczynam syntezę strumieniową: %r", text[:60])
        model = self._tts.synthesizer.tts_model  # type: ignore[union-attr]
        chunk_count = 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for chunk in model.inference_stream(
                text=text,
                language=settings.language,
                gpt_cond_latent=self._gpt_cond_latent,
                speaker_embedding=self._speaker_embedding,
                **_SYNTH_PARAMS,
            ):
                chunk_count += 1
                log.debug(
                    "Chunk #%d wygenerowany (%d próbek)", chunk_count, chunk.shape[-1]
                )
                yield chunk.cpu().numpy()
        log.debug("Synteza zakończona, chunków: %d", chunk_count)

    def synthesize_to_file(self, text: str, output_path: Path) -> dict:
        # syntetyzuje tekst i zapisuje WAV, zwraca statystyki latencji
        log.info("Synteza do pliku: %s", output_path.name)
        t0 = time.perf_counter()
        t_first: float | None = None
        chunks: list[torch.Tensor] = []

        for chunk in self.stream_chunks(text):
            if t_first is None:
                t_first = time.perf_counter() - t0
                log.debug("Latencja pierwszego chunka: %.3fs", t_first)
            chunks.append(torch.from_numpy(chunk))

        wav = torch.cat(chunks, dim=-1).unsqueeze(0).float()
        torchaudio.save(str(output_path), wav, _SAMPLE_RATE)
        elapsed = time.perf_counter() - t0
        dur = wav.shape[-1] / _SAMPLE_RATE

        result = {
            "latency_first_chunk": round(t_first or 0.0, 3),
            "total_time": round(elapsed, 3),
            "audio_duration": round(dur, 2),
            "output": str(output_path),
        }
        log.info(
            "Plik zapisany | 1.chunk=%.3fs | total=%.3fs | audio=%.2fs",
            result["latency_first_chunk"],
            result["total_time"],
            result["audio_duration"],
        )
        return result

    # --- prywatne metody pomocnicze ---

    def _assert_loaded(self) -> None:
        if self._tts is None:
            raise RuntimeError("Najpierw wywołaj DaemonVox.load()")

    def _zaladuj_model(self) -> TTS:
        log.info("Ładowanie modelu XTTS v2 na %s...", self._device)
        tts = TTS(settings.tts_model).to(self._device)
        model = tts.synthesizer.tts_model
        model.eval()
        try:
            model.half()
            log.info("Tryb FP16 aktywny")
        except Exception as e:
            log.warning("FP16 niedostępne: %s", e)
        return tts

    def _zbierz_probki(self) -> list[str]:
        pliki = sorted(settings.samples_dir.glob("*.wav"))  # type: ignore[union-attr]
        if not pliki:
            raise FileNotFoundError(f"Brak plików .wav w {settings.samples_dir}")
        log.info("Próbki głosu: %s", [p.name for p in pliki])
        return [str(p) for p in pliki]

    def _zbuduj_cache(self, probki: list[str]) -> None:
        log.info("Obliczam embeddingi głosu z %d próbek...", len(probki))
        model = self._tts.synthesizer.tts_model  # type: ignore[union-attr]
        # tymczasowo float32, get_conditioning_latents nie obsluguje FP16
        model.float()
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=probki
        )
        try:
            model.half()
        except Exception:
            pass
        torch.save(
            {
                "gpt_cond_latent": gpt_cond_latent,
                "speaker_embedding": speaker_embedding,
            },
            settings.cache_path,
        )
        log.info("Cache zapisany: %s", settings.cache_path)

    def _zaladuj_lub_zbuduj_cache(self, probki: list[str]):
        if settings.cache_path.exists():  # type: ignore[union-attr]
            log.info("Ładowanie cache: %s", settings.cache_path.name)  # type: ignore[union-attr]
            dane = torch.load(
                settings.cache_path,
                map_location=self._tts.synthesizer.tts_model.device,  # type: ignore[union-attr]
                weights_only=True,
            )
            return dane["gpt_cond_latent"], dane["speaker_embedding"]
        log.warning("Cache nie znaleziony - buduje od zera...")
        self._zbuduj_cache(probki)
        dane = torch.load(
            settings.cache_path,
            map_location=self._tts.synthesizer.tts_model.device,  # type: ignore[union-attr]
            weights_only=True,
        )
        return dane["gpt_cond_latent"], dane["speaker_embedding"]
