"""
mock_data_generator.py
──────────────────────
Generates synthetic WAV audio files that simulate real-world recording conditions:
  - Quiet room (low background noise)
  - Noisy / street (high noise injection)
  - Phone quality (8kHz → resample, band-limited)
  - Whispered / rushed (lower amplitude)

These serve as stand-ins until actual recordings are made.
Each file encodes a natural Hindi/Hinglish sentence mentioning one Bangalore locality.

Usage:
    python mock_data_generator.py
    python mock_data_generator.py --output_dir custom_audio --sr 16000
"""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, resample_poly

log = logging.getLogger("mock_gen")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ──────────────────────────────────────────────
# LOCALITY SAMPLE DEFINITIONS
# ──────────────────────────────────────────────

SAMPLES = [
    # (locality, sentence, language)
    ("koramangala",       "haan main koramangala mein rehta hoon",                  "hinglish"),
    ("indiranagar",       "mera ghar indiranagar mein hai",                          "hinglish"),
    ("whitefield",        "main whitefield mein kaam karta hoon",                    "hinglish"),
    ("electronic city",   "electronic city mein mera office hai",                    "hinglish"),
    ("marathahalli",      "marathahalli ke paas ek flat liya hai",                   "hinglish"),
    ("jayanagar",         "jayanagar fourth block mein rehte hain",                  "hinglish"),
    ("rajajinagar",       "rajajinagar se bus milti hai",                            "hinglish"),
    ("hebbal",            "hebbal flyover ke paas hoon",                             "hinglish"),
    ("yelahanka",         "yelahanka new town mein shifting kar raha hoon",          "hinglish"),
    ("banashankari",      "banashankari mein ek room available hai",                 "hinglish"),
    ("hsr layout",        "hsr layout sector 2 mein rehta hoon",                     "hinglish"),
    ("btm layout",        "btm layout mein PG hai",                                  "hinglish"),
    ("majestic",          "majestic bus stand ke paas hoon",                         "hinglish"),
    ("silk board",        "silk board junction par traffic bahut zyada hai",         "hinglish"),
    ("bellandur",         "bellandur lake ke paas flat hai",                         "hinglish"),
    ("sarjapur",          "sarjapur road mein shift karna hai",                      "hinglish"),
    ("bommanahalli",      "bommanahalli mein kaam karta hoon",                       "hinglish"),
    ("kr puram",          "kr puram station ke paas rehta hoon",                     "hinglish"),
    ("peenya",            "peenya industrial area mein factory hai",                 "hinglish"),
    ("yeshwanthpur",      "yeshwanthpur circle pe utar jaana",                       "hinglish"),
    ("byatarayanapura",   "byatarayanapura mein ek naukri mili hai",                 "hinglish"),
    ("kadugondanahalli",  "kadugondanahalli se auto lena hoga",                      "kannada"),
    ("hesaraghatta",      "hesaraghatta road par rehta hoon",                        "hinglish"),
    ("chikkabanavara",    "chikkabanavara ke paas ek plot liya",                     "hinglish"),
    ("rajarajeshwarinagar","rajarajeshwarinagar mein site visit hai",                "hinglish"),
    ("kothanur dinne",    "kothanur dinne locality mein flat dhundh raha hoon",      "hinglish"),
    ("thanisandra",       "thanisandra main road ke paas hoon",                      "hinglish"),
    ("doddanekundi",      "doddanekundi mein IT park ke paas kaam karta hoon",       "hinglish"),
    ("kengeri upanagara", "kengeri upanagara station ke paas rehta hoon",            "hinglish"),
    ("thalaghattapura",   "thalaghattapura mein ghar dhundhna hai",                  "hinglish"),
]

CONDITIONS = ["quiet", "noisy", "phone", "whispered"]

# ──────────────────────────────────────────────
# AUDIO SYNTHESIS HELPERS
# ──────────────────────────────────────────────

def _generate_speech_tone(duration: float, sr: int, base_freq: float = 180.0) -> np.ndarray:
    """Rough voiced-speech approximation: harmonics + amplitude envelope."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.zeros_like(t)
    for h in range(1, 8):
        signal += (1 / h) * np.sin(2 * np.pi * base_freq * h * t)
    # Amplitude envelope (onset + decay + tail)
    env = np.ones_like(t)
    attack = int(0.05 * sr)
    release = int(0.1 * sr)
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)
    return (signal * env * 0.25).astype(np.float32)


def _add_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    sig_power = np.mean(audio ** 2) + 1e-9
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio)).astype(np.float32)
    return audio + noise


def _phone_quality(audio: np.ndarray, sr: int) -> np.ndarray:
    """Simulate phone call: downsample to 8kHz and back, then bandpass 300-3400 Hz."""
    # Downsample 16k→8k→16k
    down = resample_poly(audio, 1, 2)
    up = resample_poly(down, 2, 1).astype(np.float32)
    up = up[:len(audio)]
    # Bandpass filter
    sos = butter(4, [300 / (sr / 2), 3400 / (sr / 2)], btype="band", output="sos")
    return sosfilt(sos, up).astype(np.float32)


def _whisper_effect(audio: np.ndarray) -> np.ndarray:
    """Lower amplitude and add slight high-frequency emphasis."""
    return (audio * 0.3).astype(np.float32)


def synthesize(condition: str, duration: float = 2.5, sr: int = 16000,
               base_freq: float = 180.0) -> np.ndarray:
    audio = _generate_speech_tone(duration, sr, base_freq)
    if condition == "quiet":
        audio = _add_noise(audio, snr_db=25.0)
    elif condition == "noisy":
        audio = _add_noise(audio, snr_db=5.0)
    elif condition == "phone":
        audio = _phone_quality(audio, sr)
        audio = _add_noise(audio, snr_db=18.0)
    elif condition == "whispered":
        audio = _whisper_effect(audio)
        audio = _add_noise(audio, snr_db=20.0)
    # Final normalise
    peak = np.abs(audio).max()
    return (audio / peak * 0.85).astype(np.float32) if peak > 0 else audio


# ──────────────────────────────────────────────
# GENERATOR
# ──────────────────────────────────────────────

def generate(output_dir: str = "audio_samples", sr: int = 16000, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = []
    cond_cycle = CONDITIONS * (len(SAMPLES) // len(CONDITIONS) + 1)
    random.shuffle(cond_cycle)

    for idx, ((locality, sentence, language), condition) in enumerate(
        zip(SAMPLES, cond_cycle[:len(SAMPLES)])
    ):
        # Vary duration slightly per sample
        duration = random.uniform(2.0, 3.5)
        base_freq = random.uniform(150, 220)  # speaker pitch variation

        audio = synthesize(condition, duration, sr, base_freq)

        safe_name = locality.replace(" ", "_")
        filename = f"{idx+1:02d}_{safe_name}_{condition}.wav"
        filepath = out / filename

        sf.write(str(filepath), audio, sr, subtype="PCM_16")

        manifest.append({
            "file": filename,
            "reference": sentence,
            "locality": locality,
            "condition": condition,
            "language": language,
        })
        log.info(f"  Created: {filename}  [{condition}]  duration={duration:.1f}s")

    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    log.info(f"\n✓ {len(manifest)} audio files + manifest.json saved to '{out}/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mock ASR test audio files")
    parser.add_argument("--output_dir", default="audio_samples")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.output_dir, args.sr, args.seed)
