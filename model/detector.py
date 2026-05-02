"""
model/detector.py
Loads the trained LCNN+BiLSTM model.
Falls back to heuristic scoring if no checkpoint exists (demo mode).
"""

import os
import numpy as np

# Try to import torch; fall back gracefully on Vercel free tier
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ── Config ────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
DURATION    = 4.0
N_MFCC      = 40
N_MELS      = 80
N_FFT       = 512
HOP_LENGTH  = 160
WIN_LENGTH  = 400
FMAX        = 8000
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
DROPOUT     = 0.3
INPUT_DIM   = 205
DEVICE      = "cpu"           # Vercel serverless = CPU only

CKPT_PATH   = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "checkpoints", "deepguard_best.pt"
)


# ── Architecture ──────────────────────────────────────────────────
if TORCH_AVAILABLE:

    class MaxFeatureMap2D(nn.Module):
        def forward(self, x):
            a, b = x.chunk(2, dim=1)
            return torch.max(a, b)

    class LCNNBlock(nn.Module):
        def __init__(self, ic, oc, k=3, s=1, p=1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(ic, oc*2, k, s, p, bias=False),
                nn.BatchNorm2d(oc*2),
                MaxFeatureMap2D(),
                nn.Dropout2d(0.1),
            )
        def forward(self, x): return self.net(x)

    class AttentionPool(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.a = nn.Sequential(nn.Linear(d, d//2), nn.Tanh(), nn.Linear(d//2, 1))
        def forward(self, x):
            w = torch.softmax(self.a(x), dim=1)
            return (w * x).sum(1)

    class _Net(nn.Module):
        def __init__(self, input_dim=INPUT_DIM):
            super().__init__()
            self.cnn = nn.Sequential(
                LCNNBlock(1, 16), nn.MaxPool2d((2,1),(2,1)),
                LCNNBlock(16,32), nn.MaxPool2d((2,1),(2,1)),
                LCNNBlock(32,64), nn.MaxPool2d((2,1),(2,1)),
                LCNNBlock(64,128),nn.MaxPool2d((2,1),(2,1)),
            )
            cnn_dim = (input_dim // 16) * 128
            self.proj = nn.Sequential(
                nn.Linear(cnn_dim, LSTM_HIDDEN*2),
                nn.LayerNorm(LSTM_HIDDEN*2), nn.GELU(), nn.Dropout(DROPOUT),
            )
            self.lstm = nn.LSTM(LSTM_HIDDEN*2, LSTM_HIDDEN, LSTM_LAYERS,
                                batch_first=True, dropout=DROPOUT if LSTM_LAYERS>1 else 0,
                                bidirectional=True)
            self.pool = AttentionPool(LSTM_HIDDEN*2)
            self.head = nn.Sequential(
                nn.Linear(LSTM_HIDDEN*2, 256), nn.LayerNorm(256),
                nn.GELU(), nn.Dropout(DROPOUT),
                nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 2),
            )

        def forward(self, x):
            B, C, T = x.shape
            o = self.cnn(x.unsqueeze(1))
            B2,ch,cf,ct = o.shape
            o = o.permute(0,3,1,2).reshape(B2, ct, ch*cf)
            o = self.proj(o)
            o, _ = self.lstm(o)
            o = self.pool(o)
            return self.head(o)


# ── Detector class ────────────────────────────────────────────────
class DeepGuardDetector:
    """
    Wraps the model with a clean predict() API.
    Falls back to spectral heuristics if torch/checkpoint unavailable.
    """

    def __init__(self):
        self._model = None
        self._mode  = "heuristic"

        if not TORCH_AVAILABLE:
            print("[DeepGuard] torch not available — using heuristic mode.")
            return

        try:
            net = _Net(INPUT_DIM).to(DEVICE)
            if os.path.exists(CKPT_PATH):
                ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
                net.load_state_dict(ckpt["state_dict"])
                print(f"[DeepGuard] Loaded checkpoint — "
                      f"val_acc={ckpt.get('val_acc', '?')}")
                self._mode = "model"
            else:
                print("[DeepGuard] No checkpoint — using heuristic mode.")
            net.eval()
            self._model = net
        except Exception as e:
            print(f"[DeepGuard] Model init failed ({e}) — heuristic mode.")

    def is_loaded(self) -> bool:
        return self._mode == "model"

    def predict(self, wav: np.ndarray) -> dict:
        """Returns dict with keys: p_real, p_fake, is_fake, conf."""
        if self._mode == "model" and self._model is not None:
            return self._model_predict(wav)
        return self._heuristic_predict(wav)

    # ── Model inference ───────────────────────────────────────────
    def _model_predict(self, wav):
        from model.features import AudioFeatureExtractor
        feat = AudioFeatureExtractor().wav_to_tensor(wav)
        x    = feat.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = F.softmax(self._model(x), dim=-1).squeeze().cpu().numpy()
        p_fake = float(probs[1])
        p_real = float(probs[0])
        return {"p_real": p_real, "p_fake": p_fake,
                "is_fake": p_fake > 0.5, "conf": max(p_real, p_fake)}

    # ── Heuristic fallback (no model weights needed) ──────────────
    def _heuristic_predict(self, wav):
        """
        Uses spectral features directly when no trained model is available.
        Suitable for demo/development.
        """
        import librosa

        # Spectral entropy
        spec     = np.abs(librosa.stft(wav, n_fft=N_FFT, hop_length=HOP_LENGTH))
        ps       = spec**2 / (spec**2).sum(0, keepdims=True) + 1e-10
        entropy  = float(-np.mean(ps * np.log(ps)))

        # MFCC variance
        mfcc     = librosa.feature.mfcc(y=wav, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
                                         hop_length=HOP_LENGTH)
        mfcc_var = float(np.var(mfcc))

        # Pitch stability
        try:
            f0, _, _ = librosa.pyin(wav, fmin=80, fmax=400, sr=SAMPLE_RATE)
            f0v = f0[~np.isnan(f0)]
            pitch_stab = (1 - np.std(f0v)/(np.mean(f0v)+1e-8)) if len(f0v)>1 else 0.5
        except Exception:
            pitch_stab = 0.5

        # GAN score: energy in 4–8kHz band relative to total
        freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)
        band  = (freqs >= 4000) & (freqs <= 8000)
        gan_score = float(np.clip(
            (spec[band].mean() / (spec.mean()+1e-10) - 0.10) / 0.35, 0, 1
        )) if band.any() else 0.0

        # Combine into a fake-probability score
        fake_score = np.mean([
            1 - np.clip(entropy  / 0.75, 0, 1),      # low entropy → fake
            1 - np.clip(mfcc_var / 0.030, 0, 1),     # low variance → fake
            np.clip((pitch_stab - 0.88) / 0.12, 0,1),# high stability → fake
            np.clip(gan_score, 0, 1),                  # artifact → fake
        ])

        # Add small noise so demo samples vary naturally
        noise     = float(np.random.RandomState(abs(hash(wav.tobytes())) % 9999)
                          .randn() * 0.04)
        p_fake    = float(np.clip(fake_score + noise, 0.02, 0.98))
        p_real    = 1.0 - p_fake
        is_fake   = p_fake > 0.5
        conf      = max(p_real, p_fake)

        return {"p_real": p_real, "p_fake": p_fake,
                "is_fake": is_fake, "conf": conf}
