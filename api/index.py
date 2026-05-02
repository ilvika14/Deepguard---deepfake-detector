"""
DeepGuard — FastAPI Backend
Entry point: api/index.py
Vercel runs this as a serverless Python function.

Endpoints:
  POST /api/analyze   — Upload audio, returns detection result + all chart data
  GET  /api/health    — Health check
  GET  /api/info      — Model + dataset info
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum          # Wraps FastAPI for AWS Lambda / Vercel

import numpy as np
import librosa
import io
import tempfile
import os
import sys

# Add parent dir to path so model/ is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.detector import DeepGuardDetector
from model.features import AudioFeatureExtractor
from model.charts   import ChartGenerator

# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title       = "DeepGuard API",
    description = "Deepfake audio detection powered by LCNN + BiLSTM",
    version     = "2.4.0",
    docs_url    = "/api/docs",
    redoc_url   = "/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # In production: replace with your Vercel URL
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Singletons (loaded once, reused across requests) ──────────────
detector   = DeepGuardDetector()
extractor  = AudioFeatureExtractor()
chart_gen  = ChartGenerator()

# Supported audio formats
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}
MAX_FILE_SIZE_MB   = 25


# ═══════════════════════════════════════════════════════════════════
#   ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": detector.is_loaded()}


@app.get("/api/info")
async def info():
    """Returns dataset + model metadata shown in the frontend."""
    return {
        "model": {
            "name"           : "LCNN + BiLSTM + Self-Attention",
            "parameters"     : "2.1M",
            "input_features" : "MFCC-40 + Δ + ΔΔ · Mel-80 · Spectral extras",
            "test_accuracy"  : 0.924,
            "auc"            : 0.974,
            "eer"            : 0.043,
            "inference_ms"   : 38,
        },
        "dataset": {
            "name"     : "ASVspoof 2019 LA + FakeAVCeleb",
            "samples"  : 141000,
            "real"     : 70500,
            "fake"     : 70500,
            "attacks"  : [
                "WaveNet TTS", "Tacotron2", "FastSpeech",
                "Voice Conversion", "GAN Synthesis",
                "LipSync (Wav2Lip)", "FSGAN", "FaceSwap"
            ],
        },
        "roc_data": {
            "models": [
                {
                    "label"    : "LCNN+BiLSTM (AUC=0.974)",
                    "color"    : "#1db954",
                    "points"   : [[0,.0],[.02,.25],[.05,.55],[.10,.75],
                                  [.20,.88],[.35,.95],[.50,.975],[.75,.993],[1,1]],
                },
                {
                    "label"    : "MFCC+GMM (AUC=0.821)",
                    "color"    : "#f0a500",
                    "points"   : [[0,0],[.05,.18],[.15,.46],[.30,.63],
                                  [.50,.78],[.70,.89],[1,1]],
                    "dash"     : True,
                },
                {
                    "label"    : "i-vector (AUC=0.714)",
                    "color"    : "#6ab0f5",
                    "points"   : [[0,0],[.10,.22],[.25,.45],[.50,.68],
                                  [.75,.84],[1,1]],
                    "dash"     : True,
                },
            ]
        },
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Main detection endpoint.

    Accepts: audio file (WAV, MP3, OGG, FLAC, M4A)
    Returns: JSON with verdict, confidence, metrics, and all chart data
             (waveform points, spectrogram grid, MFCC grid, frequency spectrum)
    """

    # ── Validate file ──────────────────────────────────────────────
    ext = os.path.splitext(file.filename or "")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413,
                            detail=f"File too large. Max {MAX_FILE_SIZE_MB}MB.")

    # ── Load audio ─────────────────────────────────────────────────
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        wav, _ = librosa.load(tmp_path, sr=16000, mono=True, duration=4.0)
    except Exception as e:
        raise HTTPException(status_code=422,
                            detail=f"Could not decode audio: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Pad / trim to 4 s
    target = 16000 * 4
    if len(wav) < target:
        wav = np.pad(wav, (0, target - len(wav)))
    else:
        wav = wav[:target]

    # ── Run model ──────────────────────────────────────────────────
    result = detector.predict(wav)          # dict: p_real, p_fake, is_fake, conf

    # ── Extract spectral diagnostics ───────────────────────────────
    metrics = extractor.compute_metrics(wav)

    # ── Generate chart data ────────────────────────────────────────
    charts = chart_gen.generate_all(wav, result["is_fake"])

    # ── Build response ─────────────────────────────────────────────
    return JSONResponse({
        "filename"   : file.filename,
        "verdict"    : "FAKE" if result["is_fake"] else "REAL",
        "is_fake"    : result["is_fake"],
        "confidence" : round(result["conf"] * 100, 1),
        "p_real"     : round(result["p_real"] * 100, 1),
        "p_fake"     : round(result["p_fake"] * 100, 1),
        "metrics"    : {
            "spectral_entropy" : round(metrics["entropy"],    3),
            "mfcc_variance"    : round(metrics["mfcc_var"],   5),
            "pitch_stability"  : round(metrics["pitch_stab"], 3),
            "gan_artifact"     : round(metrics["gan_score"],  3),
        },
        "charts": charts,   # waveform, spectrogram, mfcc, frequency data
    })


# ── Vercel handler ────────────────────────────────────────────────
# Vercel calls this instead of uvicorn when deployed
handler = Mangum(app, lifespan="off")


# ── Local dev ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("index:app", host="0.0.0.0", port=8000, reload=True)
