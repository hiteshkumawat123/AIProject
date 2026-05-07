"""
ASR Benchmarking Pipeline for Indian Conversational Speech
Focus: Locality name entity extraction (Bangalore localities)
Baseline: Deepgram | Challengers: Faster-Whisper, Sarvam AI Saaras

Author: ASR Intern Assessment
"""

import os
import time
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
log = logging.getLogger("asr_bench")

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

# Phonetic alias map: common mis-transcriptions -> canonical form
LOCALITY_ALIASES = {
    "maratha halli": "marathahalli",
    "maratha hali": "marathahalli",
    "koramangal": "koramangala",
    "indira nagar": "indiranagar",
    "hsr": "hsr layout",
    "btm": "btm layout",
    "electronic city phase": "electronic city",
    "yeshwanth pur": "yeshwanthpur",
    "raja rajeshwari nagar": "rajarajeshwarinagar",
    "byatara yanpura": "byatarayanapura",
    "kr puram": "kr puram",
    "kengeri upa nagara": "kengeri upanagara",
}


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
    error: Optional[str] = None

@dataclass
class EvalMetrics:
    engine: str
    wer: float
    cer: float
    emr: float                          # Entity Match Rate (locality names)
    avg_latency_ms: float
    samples_evaluated: int
    failure_modes: dict = field(default_factory=dict)


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


# ─────────────────────────────────────────────
# ASR ENGINE BASE
# ─────────────────────────────────────────────

class ASREngine(ABC):
    """Abstract base for all ASR engines."""

    name: str = "base"

    @abstractmethod
    def transcribe(self, audio_path: Path) -> tuple[str, float]:
        """Returns (transcript, latency_ms)."""
        ...

    def run_batch(self, samples: list[AudioSample]) -> list[TranscriptionResult]:
        results = []
        for sample in samples:
            try:
                t0 = time.perf_counter()
                transcript, latency = self.transcribe(sample.path)
                results.append(TranscriptionResult(
                    engine=self.name,
                    sample=sample,
                    hypothesis=transcript.lower().strip(),
                    latency_ms=latency,
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

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY", "")
        if not self.api_key:
            log.warning("DEEPGRAM_API_KEY not set — engine will return mock results")

    def transcribe(self, audio_path: Path) -> tuple[str, float]:
        if not self.api_key:
            return self._mock_transcribe(audio_path)
        url = "https://api.deepgram.com/v1/listen?model=nova-3&language=hi&punctuate=true&smart_format=true"
        headers = {"Authorization": f"Token {self.api_key}", "Content-Type": "audio/wav"}
        t0 = time.perf_counter()
        with open(audio_path, "rb") as f:
            resp = httpx.post(url, headers=headers, content=f.read(), timeout=30)
        latency_ms = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        data = resp.json()
        transcript = data["results"]["channels"][0]["alternatives"][0]["transcript"]
        return transcript, latency_ms

    def _mock_transcribe(self, audio_path: Path) -> tuple[str, float]:
        """Simulated output for CI / demo without API keys."""
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

    def __init__(self, model_size: str = "large-v3", device: str = "cpu", compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                log.info(f"Loading Faster-Whisper {self.model_size} on {self.device}...")
                self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            except ImportError:
                log.warning("faster-whisper not installed; using mock mode")
                self._model = "mock"

    def transcribe(self, audio_path: Path) -> tuple[str, float]:
        self._load_model()
        if self._model == "mock":
            return self._mock_transcribe(audio_path)
        t0 = time.perf_counter()
        segments, info = self._model.transcribe(
            str(audio_path), language="hi", beam_size=5,
            vad_filter=True, word_timestamps=False
        )
        transcript = " ".join(seg.text for seg in segments).strip()
        latency_ms = (time.perf_counter() - t0) * 1000
        return transcript, latency_ms

    def _mock_transcribe(self, audio_path: Path) -> tuple[str, float]:
        stem = audio_path.stem.lower()
        # Simulate whisper's common failure: splitting compound names
        whisper_errors = {
            "marathahalli": "maratha halli",
            "koramangala": "koramangal",
            "rajarajeshwarinagar": "raja rajeshwari nagar",
            "byatarayanapura": "byatara yanpura",
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

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY", "")
        if not self.api_key:
            log.warning("SARVAM_API_KEY not set — engine will return mock results")

    def transcribe(self, audio_path: Path) -> tuple[str, float]:
        if not self.api_key:
            return self._mock_transcribe(audio_path)
        url = "https://api.sarvam.ai/speech-to-text"
        headers = {"api-subscription-key": self.api_key}
        t0 = time.perf_counter()
        with open(audio_path, "rb") as f:
            files = {"file": (audio_path.name, f, "audio/wav")}
            data = {"model": "saaras:v2", "language_code": "hi-IN", "with_timestamps": "false"}
            resp = httpx.post(url, headers=headers, files=files, data=data, timeout=30)
        latency_ms = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        return resp.json().get("transcript", ""), latency_ms

    def _mock_transcribe(self, audio_path: Path) -> tuple[str, float]:
        # Sarvam is trained on Indic data — simulate higher accuracy on local names
        stem = audio_path.stem.lower()
        return f"haan main {stem} mein rehta hoon", np.random.uniform(150, 400)


# ─────────────────────────────────────────────
# EVALUATOR
# ─────────────────────────────────────────────

class Evaluator:
    """Computes WER, CER, EMR and diagnoses failure modes."""

    NOISE_KEYWORDS = ["noisy", "traffic", "street", "cafe"]
    CODESW_KEYWORDS = ["hinglish", "mixed"]

    def _normalise_text(self, text: str) -> str:
        text = text.lower().strip()
        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        # Apply alias normalisation
        for alias, canonical in LOCALITY_ALIASES.items():
            text = text.replace(alias, canonical)
        return text

    def entity_match_rate(self, results: list[TranscriptionResult]) -> float:
        matched = 0
        for r in results:
            if r.error:
                continue
            hyp = self._normalise_text(r.hypothesis)
            locality = r.sample.locality.lower()
            # Check canonical + aliases
            candidates = {locality} | {k for k, v in LOCALITY_ALIASES.items() if v == locality}
            if any(c in hyp for c in candidates):
                matched += 1
        valid = [r for r in results if not r.error]
        return matched / len(valid) if valid else 0.0

    def compute_metrics(self, results: list[TranscriptionResult], engine_name: str) -> EvalMetrics:
        valid = [r for r in results if not r.error and r.hypothesis]
        if not valid:
            return EvalMetrics(engine=engine_name, wer=1.0, cer=1.0, emr=0.0,
                               avg_latency_ms=0.0, samples_evaluated=0)

        refs = [self._normalise_text(r.sample.reference) for r in valid]
        hyps = [self._normalise_text(r.hypothesis) for r in valid]

        _wer = wer(refs, hyps)
        _cer = cer(refs, hyps)
        _emr = self.entity_match_rate(valid)
        _lat = np.mean([r.latency_ms for r in valid])

        failure_modes = self._analyse_failures(valid)

        return EvalMetrics(
            engine=engine_name,
            wer=round(_wer, 4),
            cer=round(_cer, 4),
            emr=round(_emr, 4),
            avg_latency_ms=round(_lat, 1),
            samples_evaluated=len(valid),
            failure_modes=failure_modes,
        )

    def _analyse_failures(self, results: list[TranscriptionResult]) -> dict:
        modes = {"noise": 0, "code_switching": 0, "compound_names": 0, "accent": 0}
        for r in results:
            hyp = self._normalise_text(r.hypothesis)
            locality = r.sample.locality.lower()
            if locality not in hyp:
                # Classify failure type
                if r.sample.condition in self.NOISE_KEYWORDS:
                    modes["noise"] += 1
                if r.sample.language in self.CODESW_KEYWORDS:
                    modes["code_switching"] += 1
                if len(locality.split()) == 1 and len(locality) > 10:
                    modes["compound_names"] += 1
                # Heuristic: if first 4 chars match but rest doesn't → accent
                if locality[:4] in hyp:
                    modes["accent"] += 1
        return modes


# ─────────────────────────────────────────────
# REPORTER
# ─────────────────────────────────────────────

class Reporter:
    """Generates Markdown summary and Matplotlib/Seaborn charts."""

    COLORS = {
        "Deepgram Nova-3": "#4A90D9",
        "Faster-Whisper large-v3": "#E07B39",
        "Sarvam Saaras-v2": "#27AE60",
    }

    def print_table(self, metrics: list[EvalMetrics]):
        header = f"{'Engine':<28} {'WER':>6} {'CER':>6} {'EMR':>6} {'Latency(ms)':>12} {'Samples':>8}"
        print("\n" + "="*72)
        print(header)
        print("-"*72)
        for m in metrics:
            print(f"{m.engine:<28} {m.wer:>6.3f} {m.cer:>6.3f} {m.emr:>6.3f} {m.avg_latency_ms:>12.1f} {m.samples_evaluated:>8}")
        print("="*72 + "\n")

    def plot_comparison(self, metrics: list[EvalMetrics], output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        engines = [m.engine for m in metrics]
        colors = [self.COLORS.get(e, "#888") for e in engines]

        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        fig.patch.set_facecolor("#0F1117")
        for ax in axes:
            ax.set_facecolor("#1A1D27")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#333")

        metric_data = [
            ("WER ↓", [m.wer for m in metrics], True),
            ("CER ↓", [m.cer for m in metrics], True),
            ("EMR ↑", [m.emr for m in metrics], False),
            ("Latency ms ↓", [m.avg_latency_ms for m in metrics], True),
        ]

        for ax, (label, values, lower_better) in zip(axes, metric_data):
            bars = ax.bar(range(len(engines)), values, color=colors, width=0.6, edgecolor="white", linewidth=0.5)
            ax.set_xticks(range(len(engines)))
            ax.set_xticklabels([e.split()[0] for e in engines], fontsize=9, color="white")
            ax.set_title(label, color="white", fontsize=11, fontweight="bold", pad=8)
            ax.yaxis.label.set_color("white")
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{val:.3f}" if val < 100 else f"{val:.0f}",
                        ha="center", va="bottom", color="white", fontsize=8, fontweight="bold")

        fig.suptitle("ASR Benchmark — Bangalore Locality Names", color="white",
                     fontsize=14, fontweight="bold", y=1.02)

        patches = [mpatches.Patch(color=c, label=e) for e, c in zip(engines, colors)]
        fig.legend(handles=patches, loc="lower center", ncol=3, framealpha=0,
                   labelcolor="white", fontsize=9, bbox_to_anchor=(0.5, -0.08))

        plt.tight_layout()
        out = output_dir / "asr_benchmark_chart.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="#0F1117")
        log.info(f"Chart saved → {out}")
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
                textprops={"color": "white", "fontsize": 9}
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
        data = [
            {
                "engine": m.engine, "wer": m.wer, "cer": m.cer,
                "emr": m.emr, "avg_latency_ms": m.avg_latency_ms,
                "samples_evaluated": m.samples_evaluated,
                "failure_modes": m.failure_modes,
            }
            for m in metrics
        ]
        out = output_dir / "results.json"
        out.write_text(json.dumps(data, indent=2))
        log.info(f"Results JSON saved → {out}")


# ─────────────────────────────────────────────
# PIPELINE RUNNER
# ─────────────────────────────────────────────

def load_samples(audio_dir: Path, manifest_path: Optional[Path] = None) -> list[AudioSample]:
    """
    Load audio samples. If manifest.json exists use it; else auto-detect from filenames.
    Manifest format: [{"file": "koramangala_quiet.wav", "reference": "haan main koramangala mein rehta hoon",
                        "locality": "koramangala", "condition": "quiet", "language": "hinglish"}, ...]
    """
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

    # Fallback: infer from filename (locality_condition.wav)
    samples = []
    for wav in sorted(audio_dir.glob("*.wav")):
        parts = wav.stem.lower().split("_")
        locality = parts[0] if parts else wav.stem.lower()
        condition = parts[1] if len(parts) > 1 else "unknown"
        samples.append(AudioSample(
            path=wav,
            reference=f"haan main {locality} mein rehta hoon",
            locality=locality,
            condition=condition,
        ))
    return samples


def run_benchmark(
    audio_dir: str = "audio_samples",
    manifest: str = "audio_samples/manifest.json",
    output_dir: str = "output",
    engines: Optional[list[ASREngine]] = None,
):
    audio_path = Path(audio_dir)
    manifest_path = Path(manifest)
    out_path = Path(output_dir)

    samples = load_samples(audio_path, manifest_path)
    if not samples:
        log.error(f"No audio samples found in {audio_path}. Run mock_data_generator.py first.")
        return

    log.info(f"Loaded {len(samples)} audio samples")

    if engines is None:
        engines = [
            DeepgramEngine(),
            FasterWhisperEngine(),
            SarvamEngine(),
        ]

    evaluator = Evaluator()
    reporter = Reporter()
    all_metrics = []

    for engine in engines:
        log.info(f"Running engine: {engine.name}")
        results = engine.run_batch(samples)
        metrics = evaluator.compute_metrics(results, engine.name)
        all_metrics.append(metrics)
        log.info(f"  WER={metrics.wer:.3f}  CER={metrics.cer:.3f}  EMR={metrics.emr:.3f}  Latency={metrics.avg_latency_ms:.0f}ms")

    reporter.print_table(all_metrics)
    reporter.plot_comparison(all_metrics, out_path)
    reporter.plot_failure_modes(all_metrics, out_path)
    reporter.save_json(all_metrics, out_path)

    log.info("Benchmark complete. Results saved to output/")
    return all_metrics


if __name__ == "__main__":
    run_benchmark()
