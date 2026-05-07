"""
ASR Benchmarking Pipeline v2 — Indian Conversational Speech
Focus: Locality name entity extraction (Bangalore localities)
Baseline: Deepgram Nova-3 | Challengers: Faster-Whisper, Sarvam AI Saaras

Key improvements over v1:
  1. Retry logic with exponential backoff on API failures
  2. Per-condition EMR breakdown (quiet / noisy / phone / whispered)
  3. Phonetic similarity scoring (partial EMR credit via Levenshtein ratio)
  4. Latency percentiles (p50 / p95) instead of mean-only
  5. Confidence score extraction (Deepgram word-confidence passthrough)
  6. Expanded alias map with transliteration variants
  7. Per-sample result CSV export for further analysis
  8. CLI args (--audio_dir, --output_dir, --engines, --dry_run)
  9. Graceful Whisper medium fallback when large-v3 OOMs
 10. IndicConformer engine stub (AI4Bharat open-source option)
"""

import argparse
import csv
import os
import time
import json
import logging
import re
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import unicodedata

import numpy as np
import soundfile as sf
from jiwer import wer, cer
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import httpx
import dotenv

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("asr_bench_v2")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

BANGALORE_LOCALITIES = [
    "koramangala", "indiranagar", "whitefield", "electronic city", "marathahalli",
    "jayanagar", "rajajinagar", "hebbal", "yelahanka", "banashankari",
    "hsr layout", "btm layout", "majestic", "silk board", "bellandur",
    "sarjapur", "bommanahalli", "kr puram", "peenya", "yeshwanthpur",
    "byatarayanapura", "kadugondanahalli", "hesaraghatta", "chikkabanavara",
    "rajarajeshwarinagar", "kothanur dinne", "thanisandra", "doddanekundi",
    "kengeri upanagara", "thalaghattapura",
]

# ── Expanded alias map ─────────────────────────────────────────────────────────
# Each key is a known mis-transcription variant; value is the canonical locality.
# New in v2: added common Devanagari romanisation variants + extra Whisper splits.
LOCALITY_ALIASES = {
    # marathahalli
    "maratha halli": "marathahalli",
    "maratha hali": "marathahalli",
    "marathahali": "marathahalli",
    "maratahalli": "marathahalli",
    # koramangala
    "koramangal": "koramangala",
    "koramangal a": "koramangala",
    # indiranagar
    "indira nagar": "indiranagar",
    "indira nagara": "indiranagar",
    # rajarajeshwarinagar
    "raja rajeshwari nagar": "rajarajeshwarinagar",
    "raja rajeshwara nagar": "rajarajeshwarinagar",
    "rajarajeshwara nagar": "rajarajeshwarinagar",
    # byatarayanapura
    "byatara yanpura": "byatarayanapura",
    "byatara yana pura": "byatarayanapura",
    "byatarayana pura": "byatarayanapura",
    # kadugondanahalli
    "kadu gondana halli": "kadugondanahalli",
    "kadu go hana halli": "kadugondanahalli",
    "kadugondana halli": "kadugondanahalli",
    # doddanekundi
    "dodda ne kundy": "doddanekundi",
    "dodda ne condi": "doddanekundi",
    "dodda nekundi": "doddanekundi",
    # short forms
    "hsr": "hsr layout",
    "btm": "btm layout",
    "kr puram": "kr puram",           # identity, kept for alias lookup symmetry
    "kengeri upa nagara": "kengeri upanagara",
    "kengeri upanagar": "kengeri upanagara",
    # electronic city variants
    "electronic city phase": "electronic city",
    "electronic city phase 1": "electronic city",
    "electronic city phase 2": "electronic city",
    # yeshwanthpur
    "yeshwanth pur": "yeshwanthpur",
    "yeshwant pur": "yeshwanthpur",
    # hesaraghatta
    "hesara ghatta": "hesaraghatta",
    "hesara gatta": "hesaraghatta",
    # chikkabanavara
    "chikka banavara": "chikkabanavara",
    "chikka bana vara": "chikkabanavara",
    # thalaghattapura
    "thala ghattapura": "thalaghattapura",
    "thala ghatta pura": "thalaghattapura",
}


# ─────────────────────────────────────────────
# UTILITY: Levenshtein ratio (for partial credit)
# ─────────────────────────────────────────────

def _levenshtein(s1: str, s2: str) -> int:
    """Pure-Python edit distance."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev[j-1] + cost)
    return dp[n]


def phonetic_similarity(a: str, b: str) -> float:
    """Normalised edit distance similarity in [0, 1]."""
    if not a or not b:
        return 0.0
    dist = _levenshtein(a, b)
    return 1.0 - dist / max(len(a), len(b))


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class AudioSample:
    path: Path
    reference: str                      # ground-truth transcript
    locality: str                       # canonical locality name
    condition: str = "unknown"          # quiet / noisy / phone / whispered
    language: str = "hinglish"          # hindi / hinglish / kannada

@dataclass
class TranscriptionResult:
    engine: str
    sample: AudioSample
    hypothesis: str
    latency_ms: float
    confidence: Optional[float] = None  # NEW: word-level confidence (where available)
    error: Optional[str] = None

@dataclass
class EvalMetrics:
    engine: str
    wer: float
    cer: float
    emr: float                          # Entity Match Rate (strict)
    partial_emr: float                  # NEW: phonetic partial credit EMR
    avg_latency_ms: float
    p50_latency_ms: float               # NEW
    p95_latency_ms: float               # NEW
    samples_evaluated: int
    failure_modes: dict = field(default_factory=dict)
    condition_emr: dict = field(default_factory=dict)   # NEW: per-condition breakdown
    avg_confidence: Optional[float] = None              # NEW


# ─────────────────────────────────────────────
# AUDIO PROCESSOR
# ─────────────────────────────────────────────

class AudioProcessor:
    """Handles audio loading, normalisation, and optional noise injection."""

    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr

    def load(self, path: Path) -> tuple[np.ndarray, int]:
        audio, sr = sf.read(str(path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32), sr

    def normalise(self, audio: np.ndarray) -> np.ndarray:
        peak = np.abs(audio).max()
        return audio / peak if peak > 0 else audio

    def inject_noise(self, audio: np.ndarray, snr_db: float = 10.0) -> np.ndarray:
        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        return (audio + noise).astype(np.float32)

    def duration_seconds(self, path: Path) -> float:
        audio, sr = self.load(path)
        return len(audio) / sr


# ─────────────────────────────────────────────
# RETRY HELPER (new in v2)
# ─────────────────────────────────────────────

def _with_retry(fn, retries: int = 3, base_delay: float = 1.0):
    """
    Call fn(); on exception, retry with exponential backoff.
    Returns (result, attempts_used).
    """
    for attempt in range(1, retries + 1):
        try:
            return fn(), attempt
        except httpx.HTTPStatusError as exc:
            # Don't retry on auth errors
            if exc.response.status_code in (401, 403):
                raise
            if attempt == retries:
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            log.warning(f"HTTP {exc.response.status_code} — retrying in {delay:.1f}s (attempt {attempt}/{retries})")
            time.sleep(delay)
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            if attempt == retries:
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            log.warning(f"Network error ({exc}) — retrying in {delay:.1f}s (attempt {attempt}/{retries})")
            time.sleep(delay)


# ─────────────────────────────────────────────
# ASR ENGINE BASE
# ─────────────────────────────────────────────

class ASREngine(ABC):
    """Abstract base for all ASR engines."""

    name: str = "base"

    @abstractmethod
    def transcribe(self, audio_path: Path) -> tuple[str, float, Optional[float]]:
        """Returns (transcript, latency_ms, confidence_or_None)."""
        ...

    def run_batch(self, samples: list[AudioSample]) -> list[TranscriptionResult]:
        results = []
        for sample in samples:
            try:
                transcript, latency, confidence = self.transcribe(sample.path)
                results.append(TranscriptionResult(
                    engine=self.name,
                    sample=sample,
                    hypothesis=transcript.lower().strip(),
                    latency_ms=latency,
                    confidence=confidence,
                ))
            except Exception as exc:
                log.warning(f"[{self.name}] Failed on {sample.path.name}: {exc}")
                results.append(TranscriptionResult(
                    engine=self.name, sample=sample,
                    hypothesis="", latency_ms=0.0, error=str(exc),
                ))
        return results


# ─────────────────────────────────────────────
# ENGINE: DEEPGRAM (baseline)
# ─────────────────────────────────────────────

class DeepgramEngine(ASREngine):
    name = "Deepgram Nova-3"

    def __init__(self, api_key: Optional[str] = None, retries: int = 3):
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY", "")
        self.retries = retries
        if not self.api_key:
            log.warning("DEEPGRAM_API_KEY not set — engine will return mock results")

    def transcribe(self, audio_path: Path) -> tuple[str, float, Optional[float]]:
        if not self.api_key:
            text, lat = self._mock_transcribe(audio_path)
            return text, lat, None

        url = (
            "https://api.deepgram.com/v1/listen"
            "?model=nova-3&language=hi&punctuate=true"
            "&smart_format=true&utterances=false"
        )
        headers = {"Authorization": f"Token {self.api_key}", "Content-Type": "audio/wav"}
        audio_bytes = audio_path.read_bytes()

        t0 = time.perf_counter()
        def _call():
            return httpx.post(url, headers=headers, content=audio_bytes, timeout=30)

        resp, _ = _with_retry(_call, retries=self.retries)
        latency_ms = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        data = resp.json()

        alt = data["results"]["channels"][0]["alternatives"][0]
        transcript = alt.get("transcript", "")

        # Average word confidence — NEW
        words = alt.get("words", [])
        avg_conf = float(np.mean([w["confidence"] for w in words])) if words else None

        return transcript, latency_ms, avg_conf

    def _mock_transcribe(self, audio_path: Path) -> tuple[str, float]:
        mock_map = {
            "koramangala": "haan main koramangala mein rehta hoon",
            "marathahalli": "mera ghar marathahalli ke paas hai",
            "indiranagar": "main indiranagar se hoon",
            "whitefield": "whitefield mein kaam karta hoon",
            "electronic_city": "electronic city mein office hai",
        }
        stem = audio_path.stem.lower().replace(" ", "_")
        for k, v in mock_map.items():
            if k in stem:
                return v, np.random.uniform(200, 500)
        return f"haan main {audio_path.stem.lower()} mein rehta hoon", np.random.uniform(200, 500)


# ─────────────────────────────────────────────
# ENGINE: FASTER-WHISPER (local, open-source)
# ─────────────────────────────────────────────

class FasterWhisperEngine(ASREngine):
    name = "Faster-Whisper large-v3"

    # v2: try large-v3 first, fall back to medium if OOM
    MODEL_FALLBACK_CHAIN = ["large-v3", "medium"]

    def __init__(self, model_size: str = "large-v3", device: str = "cpu", compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            log.warning("faster-whisper not installed; using mock mode. Install with: pip install faster-whisper")
            self._model = "mock"
            return

        for size in [self.model_size] + [m for m in self.MODEL_FALLBACK_CHAIN if m != self.model_size]:
            try:
                log.info(f"Loading Faster-Whisper {size} on {self.device} ({self.compute_type})...")
                self._model = WhisperModel(size, device=self.device, compute_type=self.compute_type)
                self.name = f"Faster-Whisper {size}"
                log.info(f"  ✓ Loaded {size}")
                return
            except (RuntimeError, Exception) as exc:
                if "out of memory" in str(exc).lower() or "oom" in str(exc).lower():
                    log.warning(f"OOM loading {size} — trying smaller model")
                    continue
                raise

        log.warning("All Whisper model sizes failed; falling back to mock mode")
        self._model = "mock"

    def transcribe(self, audio_path: Path) -> tuple[str, float, Optional[float]]:
        self._load_model()
        if self._model == "mock":
            text, lat = self._mock_transcribe(audio_path)
            return text, lat, None
        t0 = time.perf_counter()
        segments, info = self._model.transcribe(
            str(audio_path), language="hi", beam_size=5,
            vad_filter=True, word_timestamps=True,   # v2: request word timestamps for confidence
        )
        seg_list = list(segments)
        transcript = " ".join(seg.text for seg in seg_list).strip()
        latency_ms = (time.perf_counter() - t0) * 1000

        # Extract avg word probability as confidence proxy
        all_words = [w for seg in seg_list for w in (seg.words or [])]
        avg_conf = float(np.mean([w.probability for w in all_words])) if all_words else None

        return transcript, latency_ms, avg_conf

    def _mock_transcribe(self, audio_path: Path) -> tuple[str, float]:
        stem = audio_path.stem.lower()
        whisper_errors = {
            "marathahalli": "maratha halli",
            "koramangala": "koramangal",
            "rajarajeshwarinagar": "raja rajeshwari nagar",
            "byatarayanapura": "byatara yanpura",
            "kadugondanahalli": "kadu gondana halli",
            "doddanekundi": "dodda ne condi",
        }
        for k, v in whisper_errors.items():
            if k in stem:
                return f"haan main {v} mein rehta hoon", np.random.uniform(800, 2500)
        return f"haan main {stem} mein rehta hoon", np.random.uniform(800, 2500)


# ─────────────────────────────────────────────
# ENGINE: SARVAM AI — Saaras (India-native)
# ─────────────────────────────────────────────

class SarvamEngine(ASREngine):
    name = "Sarvam Saaras-v2"

    def __init__(self, api_key: Optional[str] = None, retries: int = 3):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY", "")
        self.retries = retries
        if not self.api_key:
            log.warning("SARVAM_API_KEY not set — engine will return mock results")

    def transcribe(self, audio_path: Path) -> tuple[str, float, Optional[float]]:
        if not self.api_key:
            text, lat = self._mock_transcribe(audio_path)
            return text, lat, None

        url = "https://api.sarvam.ai/speech-to-text"
        headers = {"api-subscription-key": self.api_key}
        t0 = time.perf_counter()

        def _call():
            with open(audio_path, "rb") as f:
                files = {"file": (audio_path.name, f, "audio/wav")}
                data = {
                    "model": "saaras:v2",
                    "language_code": "hi-IN",
                    "with_timestamps": "false",
                }
                return httpx.post(url, headers=headers, files=files, data=data, timeout=30)

        resp, attempts = _with_retry(_call, retries=self.retries)
        if attempts > 1:
            log.info(f"  [Sarvam] Succeeded after {attempts} attempts on {audio_path.name}")
        latency_ms = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        return resp.json().get("transcript", ""), latency_ms, None

    def _mock_transcribe(self, audio_path: Path) -> tuple[str, float]:
        stem = audio_path.stem.lower()
        return f"haan main {stem} mein rehta hoon", np.random.uniform(150, 400)


# ─────────────────────────────────────────────
# ENGINE: IndicConformer (AI4Bharat) — NEW stub
# ─────────────────────────────────────────────

class IndicConformerEngine(ASREngine):
    """
    AI4Bharat IndicConformer — open-source Indian ASR.
    Requires: pip install ai4bharat-transliteration transformers
    Docs: https://github.com/AI4Bharat/IndicASR
    """
    name = "IndicConformer (AI4Bharat)"

    def __init__(self, model_id: str = "ai4bharat/indicconformer_hi", device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self._pipe = None

    def _load_model(self):
        if self._pipe is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline
            log.info(f"Loading IndicConformer from {self.model_id}...")
            self._pipe = hf_pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                device=0 if self.device == "cuda" else -1,
            )
            log.info("  ✓ IndicConformer loaded")
        except (ImportError, Exception) as exc:
            log.warning(f"IndicConformer unavailable ({exc}); mock mode active")
            self._pipe = "mock"

    def transcribe(self, audio_path: Path) -> tuple[str, float, Optional[float]]:
        self._load_model()
        if self._pipe == "mock":
            return f"haan main {audio_path.stem.lower()} mein rehta hoon", np.random.uniform(300, 900), None
        t0 = time.perf_counter()
        result = self._pipe(str(audio_path))
        latency_ms = (time.perf_counter() - t0) * 1000
        return result.get("text", ""), latency_ms, None


# ─────────────────────────────────────────────
# EVALUATOR
# ─────────────────────────────────────────────

class Evaluator:
    """Computes WER, CER, EMR, partial-EMR and failure modes."""

    NOISE_KEYWORDS = ["noisy", "traffic", "street", "cafe"]
    CODESW_KEYWORDS = ["hinglish", "mixed"]

    def _normalise_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        # Unicode normalise (handles Devanagari diacritics if present)
        text = unicodedata.normalize("NFC", text)
        for alias, canonical in LOCALITY_ALIASES.items():
            text = text.replace(alias, canonical)
        return text

    def entity_match_rate(self, results: list[TranscriptionResult]) -> tuple[float, float]:
        """
        Returns (strict_emr, partial_emr).
        strict_emr: exact match after alias normalisation.
        partial_emr: gives 0.5 credit when phonetic_similarity >= 0.7.
        """
        strict_total = 0.0
        partial_total = 0.0
        valid = [r for r in results if not r.error]

        for r in valid:
            hyp = self._normalise_text(r.hypothesis)
            locality = r.sample.locality.lower()
            candidates = {locality} | {k for k, v in LOCALITY_ALIASES.items() if v == locality}

            if any(c in hyp for c in candidates):
                strict_total += 1.0
                partial_total += 1.0
            else:
                # Try phonetic partial credit against each word in hypothesis
                hyp_words = hyp.split()
                loc_words = locality.split()
                best_sim = max(
                    (phonetic_similarity(lw, hw) for lw in loc_words for hw in hyp_words),
                    default=0.0,
                )
                partial_total += 0.5 if best_sim >= 0.7 else 0.0

        n = len(valid)
        return (strict_total / n if n else 0.0,
                partial_total / n if n else 0.0)

    def condition_emr(self, results: list[TranscriptionResult]) -> dict[str, float]:
        """EMR broken down by audio condition (quiet / noisy / phone / whispered)."""
        from collections import defaultdict
        buckets: dict[str, list[TranscriptionResult]] = defaultdict(list)
        for r in results:
            if not r.error:
                buckets[r.sample.condition].append(r)

        out = {}
        for cond, items in sorted(buckets.items()):
            strict, _ = self.entity_match_rate(items)
            out[cond] = round(strict, 3)
        return out

    def compute_metrics(self, results: list[TranscriptionResult], engine_name: str) -> EvalMetrics:
        valid = [r for r in results if not r.error and r.hypothesis]
        if not valid:
            return EvalMetrics(
                engine=engine_name, wer=1.0, cer=1.0,
                emr=0.0, partial_emr=0.0, avg_latency_ms=0.0,
                p50_latency_ms=0.0, p95_latency_ms=0.0, samples_evaluated=0,
            )

        refs = [self._normalise_text(r.sample.reference) for r in valid]
        hyps = [self._normalise_text(r.hypothesis) for r in valid]

        _wer = wer(refs, hyps)
        _cer = cer(refs, hyps)
        strict_emr, partial_emr = self.entity_match_rate(valid)

        latencies = np.array([r.latency_ms for r in valid])
        _avg_lat = float(np.mean(latencies))
        _p50 = float(np.percentile(latencies, 50))
        _p95 = float(np.percentile(latencies, 95))

        confs = [r.confidence for r in valid if r.confidence is not None]
        avg_conf = float(np.mean(confs)) if confs else None

        failure_modes = self._analyse_failures(valid)
        cond_emr = self.condition_emr(valid)

        return EvalMetrics(
            engine=engine_name,
            wer=round(_wer, 4),
            cer=round(_cer, 4),
            emr=round(strict_emr, 4),
            partial_emr=round(partial_emr, 4),
            avg_latency_ms=round(_avg_lat, 1),
            p50_latency_ms=round(_p50, 1),
            p95_latency_ms=round(_p95, 1),
            samples_evaluated=len(valid),
            failure_modes=failure_modes,
            condition_emr=cond_emr,
            avg_confidence=round(avg_conf, 3) if avg_conf else None,
        )

    def _analyse_failures(self, results: list[TranscriptionResult]) -> dict:
        modes = {"noise": 0, "code_switching": 0, "compound_names": 0, "accent": 0}
        for r in results:
            hyp = self._normalise_text(r.hypothesis)
            locality = r.sample.locality.lower()
            if any(c in hyp for c in ({locality} | {k for k, v in LOCALITY_ALIASES.items() if v == locality})):
                continue  # not a failure
            if r.sample.condition in self.NOISE_KEYWORDS:
                modes["noise"] += 1
            if r.sample.language in self.CODESW_KEYWORDS:
                modes["code_switching"] += 1
            if len(locality.split()) == 1 and len(locality) > 10:
                modes["compound_names"] += 1
            if locality[:4] in hyp:
                modes["accent"] += 1
        return modes


# ─────────────────────────────────────────────
# REPORTER
# ─────────────────────────────────────────────

class Reporter:
    COLORS = {
        "Deepgram Nova-3": "#4A90D9",
        "Faster-Whisper large-v3": "#E07B39",
        "Faster-Whisper medium": "#E07B39",
        "Sarvam Saaras-v2": "#27AE60",
        "IndicConformer (AI4Bharat)": "#9B59B6",
    }

    def _color(self, engine: str) -> str:
        for k, v in self.COLORS.items():
            if k in engine:
                return v
        return "#888888"

    def print_table(self, metrics: list[EvalMetrics]):
        header = (
            f"{'Engine':<30} {'WER':>6} {'CER':>6} {'EMR':>6} "
            f"{'pEMR':>6} {'AvgLat':>8} {'P50':>8} {'P95':>8} {'Samples':>8}"
        )
        print("\n" + "=" * 90)
        print(header)
        print("-" * 90)
        for m in metrics:
            conf_str = f"{m.avg_confidence:.2f}" if m.avg_confidence else "  N/A"
            print(
                f"{m.engine:<30} {m.wer:>6.3f} {m.cer:>6.3f} {m.emr:>6.3f} "
                f"{m.partial_emr:>6.3f} {m.avg_latency_ms:>8.1f} "
                f"{m.p50_latency_ms:>8.1f} {m.p95_latency_ms:>8.1f} {m.samples_evaluated:>8}"
            )
        print("=" * 90 + "\n")

    def print_condition_table(self, metrics: list[EvalMetrics]):
        """Print per-condition EMR breakdown."""
        conditions = sorted({c for m in metrics for c in m.condition_emr})
        if not conditions:
            return
        col_w = 12
        header = f"{'Engine':<30}" + "".join(f"{c[:col_w]:>{col_w}}" for c in conditions)
        print("\nEMR by Condition")
        print("=" * (30 + col_w * len(conditions)))
        print(header)
        print("-" * (30 + col_w * len(conditions)))
        for m in metrics:
            row = f"{m.engine:<30}"
            for c in conditions:
                val = m.condition_emr.get(c, float("nan"))
                row += f"{val:>{col_w}.3f}" if not np.isnan(val) else f"{'N/A':>{col_w}}"
            print(row)
        print("=" * (30 + col_w * len(conditions)) + "\n")

    def plot_comparison(self, metrics: list[EvalMetrics], output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        engines = [m.engine for m in metrics]
        colors = [self._color(e) for e in engines]
        short = [e.split()[0] for e in engines]

        fig, axes = plt.subplots(1, 5, figsize=(20, 5))   # v2: 5 panels (added partial EMR)
        fig.patch.set_facecolor("#0F1117")
        for ax in axes:
            ax.set_facecolor("#1A1D27")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#333")

        metric_data = [
            ("WER ↓",       [m.wer for m in metrics],           True),
            ("CER ↓",       [m.cer for m in metrics],           True),
            ("EMR ↑",       [m.emr for m in metrics],           False),
            ("Partial EMR ↑", [m.partial_emr for m in metrics], False),
            ("Latency ms ↓",[m.avg_latency_ms for m in metrics],True),
        ]

        for ax, (label, values, lower_better) in zip(axes, metric_data):
            bars = ax.bar(range(len(engines)), values, color=colors, width=0.6,
                          edgecolor="white", linewidth=0.5)
            ax.set_xticks(range(len(engines)))
            ax.set_xticklabels(short, fontsize=9, color="white")
            ax.set_title(label, color="white", fontsize=11, fontweight="bold", pad=8)
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}" if val < 100 else f"{val:.0f}",
                    ha="center", va="bottom", color="white", fontsize=8, fontweight="bold",
                )

        fig.suptitle("ASR Benchmark v2 — Bangalore Locality Names", color="white",
                     fontsize=14, fontweight="bold", y=1.02)
        patches = [mpatches.Patch(color=self._color(e), label=e) for e in engines]
        fig.legend(handles=patches, loc="lower center", ncol=len(engines), framealpha=0,
                   labelcolor="white", fontsize=9, bbox_to_anchor=(0.5, -0.08))

        plt.tight_layout()
        out = output_dir / "asr_benchmark_chart_v2.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="#0F1117")
        log.info(f"Chart saved → {out}")
        plt.close()

    def plot_condition_heatmap(self, metrics: list[EvalMetrics], output_dir: Path):
        """NEW: heatmap of EMR by engine × condition."""
        conditions = sorted({c for m in metrics for c in m.condition_emr})
        if not conditions:
            return
        data = np.array([[m.condition_emr.get(c, 0.0) for c in conditions] for m in metrics])
        engine_labels = [m.engine.split()[0] for m in metrics]

        fig, ax = plt.subplots(figsize=(max(6, len(conditions) * 1.5), max(3, len(metrics) * 1.2)))
        fig.patch.set_facecolor("#0F1117")
        ax.set_facecolor("#1A1D27")

        im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels([c.title() for c in conditions], color="white", fontsize=10)
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels(engine_labels, color="white", fontsize=10)
        ax.set_title("EMR by Engine × Condition", color="white", fontsize=12, fontweight="bold")

        for i in range(len(metrics)):
            for j in range(len(conditions)):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                        color="black" if data[i, j] > 0.5 else "white", fontsize=9, fontweight="bold")

        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
        cbar.set_label("EMR", color="white")

        plt.tight_layout()
        out = output_dir / "condition_heatmap.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="#0F1117")
        log.info(f"Heatmap saved → {out}")
        plt.close()

    def plot_latency_percentiles(self, metrics: list[EvalMetrics], output_dir: Path):
        """NEW: grouped bar chart for avg / p50 / p95 latency."""
        engines = [m.engine.split()[0] for m in metrics]
        x = np.arange(len(engines))
        width = 0.25
        colors = [self._color(m.engine) for m in metrics]

        fig, ax = plt.subplots(figsize=(max(6, len(engines) * 2), 5))
        fig.patch.set_facecolor("#0F1117")
        ax.set_facecolor("#1A1D27")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333")

        avgs = [m.avg_latency_ms for m in metrics]
        p50s = [m.p50_latency_ms for m in metrics]
        p95s = [m.p95_latency_ms for m in metrics]

        b1 = ax.bar(x - width, avgs, width, label="Avg", color=colors, alpha=0.6, edgecolor="white", linewidth=0.5)
        b2 = ax.bar(x,         p50s, width, label="P50", color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
        b3 = ax.bar(x + width, p95s, width, label="P95", color=colors, alpha=1.0, edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(engines, color="white", fontsize=10)
        ax.set_ylabel("Latency (ms)", color="white")
        ax.set_title("Latency Percentiles (Avg / P50 / P95)", color="white", fontsize=12, fontweight="bold")
        ax.legend(labelcolor="white", framealpha=0, fontsize=9)

        for bars in [b1, b2, b3]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 5, f"{h:.0f}",
                        ha="center", va="bottom", color="white", fontsize=7)

        plt.tight_layout()
        out = output_dir / "latency_percentiles.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="#0F1117")
        log.info(f"Latency chart saved → {out}")
        plt.close()

    def plot_failure_modes(self, metrics: list[EvalMetrics], output_dir: Path):
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
        if len(metrics) == 1:
            axes = [axes]
        fig.patch.set_facecolor("#0F1117")

        for ax, m in zip(axes, metrics):
            ax.set_facecolor("#1A1D27")
            modes = m.failure_modes
            if not any(modes.values()):
                ax.text(0.5, 0.5, "No failures", ha="center", va="center", color="white")
                ax.set_title(m.engine.split()[0], color="white")
                continue
            labels = [k.replace("_", " ").title() for k in modes.keys()]
            vals = list(modes.values())
            wedge_colors = ["#E74C3C", "#F39C12", "#9B59B6", "#1ABC9C"]
            wedges, texts, autotexts = ax.pie(
                vals, labels=labels, autopct="%1.0f%%",
                colors=wedge_colors[:len(labels)], startangle=90,
                textprops={"color": "white", "fontsize": 9},
            )
            for at in autotexts:
                at.set_color("white")
            ax.set_title(m.engine.split()[0], color="white", fontsize=11, fontweight="bold")

        fig.suptitle("Failure Mode Distribution", color="white", fontsize=13, fontweight="bold")
        plt.tight_layout()
        out = output_dir / "failure_modes.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="#0F1117")
        log.info(f"Failure chart saved → {out}")
        plt.close()

    def save_json(self, metrics: list[EvalMetrics], output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        data = [asdict(m) for m in metrics]
        out = output_dir / "results.json"
        out.write_text(json.dumps(data, indent=2))
        log.info(f"Results JSON saved → {out}")

    def save_per_sample_csv(self, all_results: list[TranscriptionResult], output_dir: Path):
        """NEW: export per-sample detail for deeper post-hoc analysis."""
        output_dir.mkdir(parents=True, exist_ok=True)
        evaluator = Evaluator()
        out = output_dir / "per_sample_results.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "engine", "file", "locality", "condition", "language",
                "reference", "hypothesis", "entity_match",
                "latency_ms", "confidence", "error",
            ])
            writer.writeheader()
            for r in all_results:
                hyp_norm = evaluator._normalise_text(r.hypothesis)
                locality = r.sample.locality.lower()
                candidates = {locality} | {k for k, v in LOCALITY_ALIASES.items() if v == locality}
                entity_match = any(c in hyp_norm for c in candidates)
                writer.writerow({
                    "engine": r.engine,
                    "file": r.sample.path.name,
                    "locality": r.sample.locality,
                    "condition": r.sample.condition,
                    "language": r.sample.language,
                    "reference": r.sample.reference,
                    "hypothesis": r.hypothesis,
                    "entity_match": int(entity_match),
                    "latency_ms": round(r.latency_ms, 1),
                    "confidence": round(r.confidence, 3) if r.confidence else "",
                    "error": r.error or "",
                })
        log.info(f"Per-sample CSV saved → {out}")


# ─────────────────────────────────────────────
# SAMPLE LOADER
# ─────────────────────────────────────────────

def load_samples(audio_dir: Path, manifest_path: Optional[Path] = None) -> list[AudioSample]:
    if manifest_path and manifest_path.exists():
        data = json.loads(manifest_path.read_text())
        return [
            AudioSample(
                path=audio_dir / d["file"],
                reference=d["reference"],
                locality=d["locality"],
                condition=d.get("condition", "unknown"),
                language=d.get("language", "hinglish"),
            )
            for d in data
            if (audio_dir / d["file"]).exists()
        ]

    # Fallback: infer from filename (NN_locality_condition.wav)
    samples = []
    for wav in sorted(audio_dir.glob("*.wav")):
        parts = wav.stem.lower().split("_")
        # Skip leading index if present
        start = 1 if parts and parts[0].isdigit() else 0
        locality_parts = []
        condition = "unknown"
        for i, p in enumerate(parts[start:], start=start):
            if p in ("quiet", "noisy", "phone", "whispered"):
                condition = p
                break
            locality_parts.append(p)
        locality = " ".join(locality_parts) if locality_parts else wav.stem.lower()
        samples.append(AudioSample(
            path=wav,
            reference=f"haan main {locality} mein rehta hoon",
            locality=locality,
            condition=condition,
        ))
    return samples


# ─────────────────────────────────────────────
# PIPELINE RUNNER
# ─────────────────────────────────────────────

def run_benchmark(
    audio_dir: str = "audio_samples",
    manifest: str = "audio_samples/manifest.json",
    output_dir: str = "output",
    engines: Optional[list[ASREngine]] = None,
    dry_run: bool = False,
):
    audio_path = Path(audio_dir)
    manifest_path = Path(manifest)
    out_path = Path(output_dir)

    samples = load_samples(audio_path, manifest_path)
    if not samples:
        log.error(f"No audio samples found in {audio_path}. Run mock_data_generator.py first.")
        return

    log.info(f"Loaded {len(samples)} audio samples")
    if dry_run:
        log.info("[DRY RUN] Listing samples only — no inference will run.")
        for s in samples:
            log.info(f"  {s.path.name}  locality={s.locality}  condition={s.condition}")
        return

    if engines is None:
        engines = [
            DeepgramEngine(),
            FasterWhisperEngine(),
            SarvamEngine(),
        ]

    evaluator = Evaluator()
    reporter = Reporter()
    all_metrics: list[EvalMetrics] = []
    all_results: list[TranscriptionResult] = []

    for engine in engines:
        log.info(f"\n── Running engine: {engine.name} ──")
        results = engine.run_batch(samples)
        all_results.extend(results)
        metrics = evaluator.compute_metrics(results, engine.name)
        all_metrics.append(metrics)
        log.info(
            f"  WER={metrics.wer:.3f}  CER={metrics.cer:.3f}  "
            f"EMR={metrics.emr:.3f}  pEMR={metrics.partial_emr:.3f}  "
            f"Latency avg={metrics.avg_latency_ms:.0f}ms  "
            f"p50={metrics.p50_latency_ms:.0f}ms  p95={metrics.p95_latency_ms:.0f}ms"
        )

    reporter.print_table(all_metrics)
    reporter.print_condition_table(all_metrics)
    reporter.plot_comparison(all_metrics, out_path)
    reporter.plot_failure_modes(all_metrics, out_path)
    reporter.plot_condition_heatmap(all_metrics, out_path)
    reporter.plot_latency_percentiles(all_metrics, out_path)
    reporter.save_json(all_metrics, out_path)
    reporter.save_per_sample_csv(all_results, out_path)

    log.info("\n✓ Benchmark complete. Results saved to output/")
    return all_metrics


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ASR Benchmark Pipeline v2")
    p.add_argument("--audio_dir", default="audio_samples",
                   help="Directory containing WAV files (default: audio_samples)")
    p.add_argument("--manifest", default="audio_samples/manifest.json",
                   help="Path to manifest.json (default: audio_samples/manifest.json)")
    p.add_argument("--output_dir", default="output",
                   help="Output directory for charts and results (default: output)")
    p.add_argument("--engines", nargs="+",
                   choices=["deepgram", "whisper", "sarvam", "indicconformer"],
                   default=["deepgram", "whisper", "sarvam"],
                   help="Engines to benchmark")
    p.add_argument("--whisper_size", default="large-v3",
                   choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                   help="Whisper model size (default: large-v3)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="Device for local inference (default: cpu)")
    p.add_argument("--dry_run", action="store_true",
                   help="List samples only; no inference")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    engine_map = {
        "deepgram": lambda: DeepgramEngine(),
        "whisper":  lambda: FasterWhisperEngine(model_size=args.whisper_size, device=args.device),
        "sarvam":   lambda: SarvamEngine(),
        "indicconformer": lambda: IndicConformerEngine(device=args.device),
    }
    selected_engines = [engine_map[e]() for e in args.engines]

    run_benchmark(
        audio_dir=args.audio_dir,
        manifest=args.manifest,
        output_dir=args.output_dir,
        engines=selected_engines,
        dry_run=args.dry_run,
    )
