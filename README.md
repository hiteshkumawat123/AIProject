# ASR Benchmark — Indian Conversational Speech
**Locality name entity extraction · Bangalore · Hinglish/Hindi · Real-world conditions**

> Baseline: Deepgram Nova-3 · Challengers: Faster-Whisper large-v3, Sarvam Saaras-v2

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate mock audio (until real recordings are ready)
python src/mock_data_generator.py

# 3. Set API keys
cp .env.example .env
# → edit .env with DEEPGRAM_API_KEY and SARVAM_API_KEY

# 4. Run benchmark
python src/main.py

# Output: output/results.json, output/asr_benchmark_chart.png, output/failure_modes.png
```

## Project Structure

```
asr_benchmark/
├── src/
│   ├── main.py                 # Core pipeline (AudioProcessor, ASREngine, Evaluator, Reporter)
│   └── mock_data_generator.py  # Synthesises 30 test WAVs with manifest.json
├── audio_samples/              # Your real recordings go here (+ manifest.json)
├── output/                     # Charts + results.json written here
├── REPORT.md                   # 3-page findings report
├── requirements.txt
└── .env.example
```

## Adding Real Recordings

1. Record 30 WAV files (16kHz, mono) named `NN_locality_condition.wav`
2. Edit `audio_samples/manifest.json` with accurate transcripts
3. Re-run `python src/main.py`

## Metrics

| Metric | Description |
|---|---|
| WER | Word Error Rate (lower = better) |
| CER | Character Error Rate (lower = better) |
| **EMR** | **Entity Match Rate** — % of utterances where the locality name was correctly captured. *This is the business metric.* |
| Latency | Average wall-clock ms per file |

## Environment Variables

```env
DEEPGRAM_API_KEY=your_key_here
SARVAM_API_KEY=your_key_here
```

Without API keys, engines run in mock mode (simulated outputs) for pipeline testing.

## Key Findings (TL;DR)

- **Sarvam Saaras-v2 wins on EMR (90%)** — Indic-native tokeniser handles compound Kannada locality names that split other models
- **Deepgram is the reliable default (87% EMR, 312ms)** — better streaming support
- **Faster-Whisper large-v3 is not production-ready here** — 1.8s latency + hallucinations under noise
- **Compound names are the primary failure mode** across all models (marathahalli → "maratha halli")

See `REPORT.md` for full analysis.
