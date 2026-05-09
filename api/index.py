"""
DeepGuard — FastAPI Backend
Entry point: api/index.py

Endpoints:
  POST /api/analyze   — Upload audio and get deepfake detection
  GET  /api/health    — Health check
  GET  /api/info      — Model + dataset info
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import numpy as np
import librosa
import tempfile
import os
import sys

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.detector import DeepGuardDetector
from model.features import AudioFeatureExtractor
from model.charts import ChartGenerator


# ═══════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════

app = FastAPI(
    title="DeepGuard API",
    description="Deepfake audio detection powered by LCNN + BiLSTM",
    version="2.4.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ═══════════════════════════════════════════════════════════════════
# CORS
# ═══════════════════════════════════════════════════════════════════

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════
# LOAD MODELS / HELPERS
# ═══════════════════════════════════════════════════════════════════

detector = DeepGuardDetector()
extractor = AudioFeatureExtractor()
chart_gen = ChartGenerator()

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

ALLOWED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".ogg",
    ".flac",
    ".m4a",
    ".webm"
}

MAX_FILE_SIZE_MB = 25

# ═══════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "message": "DeepGuard API running successfully"
    }


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": detector.is_loaded()
    }


@app.get("/api/info")
async def info():
    return {
        "model": {
            "name": "LCNN + BiLSTM + Self-Attention",
            "parameters": "2.1M",
            "input_features": "MFCC-40 + Delta + DeltaDelta",
            "test_accuracy": 0.924,
            "auc": 0.974,
            "eer": 0.043,
            "inference_ms": 38,
        },
        "dataset": {
            "name": "ASVspoof 2019 LA + FakeAVCeleb",
            "samples": 141000,
            "real": 70500,
            "fake": 70500,
        }
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):

    # ──────────────────────────────────────────────────────────────
    # VALIDATE FILE
    # ──────────────────────────────────────────────────────────────

    ext = os.path.splitext(file.filename or "")[-1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'"
        )

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max {MAX_FILE_SIZE_MB}MB"
        )

    # ──────────────────────────────────────────────────────────────
    # LOAD AUDIO
    # ──────────────────────────────────────────────────────────────

    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=ext,
            delete=False
        ) as tmp:

            tmp.write(contents)
            tmp_path = tmp.name

        wav, sr = librosa.load(
            tmp_path,
            sr=16000,
            mono=True,
            duration=4.0
        )

    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Could not decode audio: {str(e)}"
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # ──────────────────────────────────────────────────────────────
    # PAD / TRIM AUDIO
    # ──────────────────────────────────────────────────────────────

    target_length = 16000 * 4

    if len(wav) < target_length:
        wav = np.pad(
            wav,
            (0, target_length - len(wav))
        )
    else:
        wav = wav[:target_length]

    # ──────────────────────────────────────────────────────────────
    # MODEL PREDICTION
    # ──────────────────────────────────────────────────────────────

    result = detector.predict(wav)

    # ──────────────────────────────────────────────────────────────
    # METRICS
    # ──────────────────────────────────────────────────────────────

    metrics = extractor.compute_metrics(wav)

    # ──────────────────────────────────────────────────────────────
    # CHARTS
    # ──────────────────────────────────────────────────────────────

    charts = chart_gen.generate_all(
        wav,
        result["is_fake"]
    )

    # ──────────────────────────────────────────────────────────────
    # RESPONSE
    # ──────────────────────────────────────────────────────────────

    return JSONResponse({
        "filename": file.filename,
        "verdict": "FAKE" if result["is_fake"] else "REAL",
        "is_fake": result["is_fake"],

        "confidence": round(result["conf"] * 100, 1),

        "p_real": round(result["p_real"] * 100, 1),
        "p_fake": round(result["p_fake"] * 100, 1),

        "metrics": {
            "spectral_entropy": round(metrics["entropy"], 3),
            "mfcc_variance": round(metrics["mfcc_var"], 5),
            "pitch_stability": round(metrics["pitch_stab"], 3),
            "gan_artifact": round(metrics["gan_score"], 3),
        },

        "charts": charts
    })


# ═══════════════════════════════════════════════════════════════════
# LOCAL DEVELOPMENT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.index:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )