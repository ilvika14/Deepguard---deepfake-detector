"""
model/features.py
Audio feature extraction: MFCC, Mel Spectrogram, spectral diagnostics.
"""

import numpy as np
import librosa

try:
    import torch
    TORCH = True
except ImportError:
    TORCH = False

SR         = 16000
DURATION   = 4.0
N_MFCC     = 40
N_MELS     = 80
N_FFT      = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
FMAX       = 8000
FRAMES     = int(DURATION * SR / HOP_LENGTH) + 1


class AudioFeatureExtractor:

    # ── Main tensor output ─────────────────────────────────────────
    def wav_to_tensor(self, wav: np.ndarray):
        """Returns (205, T) FloatTensor for the model."""
        feat = np.concatenate([
            self._mfcc(wav),   # (120, T)
            self._mel(wav),    # ( 80, T)
            self._extras(wav), # (  5, T)
        ], axis=0)
        if TORCH:
            import torch
            return torch.FloatTensor(feat)
        return feat

    # ── Diagnostic metrics ─────────────────────────────────────────
    def compute_metrics(self, wav: np.ndarray) -> dict:
        return {
            "entropy"    : self.spectral_entropy(wav),
            "mfcc_var"   : self.mfcc_variance(wav),
            "pitch_stab" : self.pitch_stability(wav),
            "gan_score"  : self.gan_artifact_score(wav),
        }

    def spectral_entropy(self, wav: np.ndarray) -> float:
        spec = np.abs(librosa.stft(wav, n_fft=N_FFT, hop_length=HOP_LENGTH))
        ps   = spec**2 / (spec**2).sum(0, keepdims=True) + 1e-10
        return float(-np.mean(ps * np.log(ps)))

    def mfcc_variance(self, wav: np.ndarray) -> float:
        mfcc = librosa.feature.mfcc(y=wav, sr=SR, n_mfcc=N_MFCC,
                                     hop_length=HOP_LENGTH)
        return float(np.var(mfcc))

    def pitch_stability(self, wav: np.ndarray) -> float:
        try:
            f0, _, _ = librosa.pyin(wav, fmin=80, fmax=400, sr=SR)
            f0v = f0[~np.isnan(f0)]
            if len(f0v) < 2:
                return 0.5
            return float(1 - np.std(f0v) / (np.mean(f0v) + 1e-8))
        except Exception:
            return 0.5

    def gan_artifact_score(self, wav: np.ndarray) -> float:
        spec  = np.abs(librosa.stft(wav, n_fft=N_FFT, hop_length=HOP_LENGTH))
        freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
        band  = (freqs >= 4000) & (freqs <= 8000)
        if not band.any():
            return 0.0
        ratio = spec[band].mean() / (spec.mean() + 1e-10)
        return float(np.clip((ratio - 0.10) / 0.35, 0.0, 1.0))

    # ── Private helpers ────────────────────────────────────────────
    def _mfcc(self, wav):
        m  = librosa.feature.mfcc(y=wav, sr=SR, n_mfcc=N_MFCC,
                                   n_fft=N_FFT, hop_length=HOP_LENGTH,
                                   win_length=WIN_LENGTH, fmax=FMAX)
        d1 = librosa.feature.delta(m)
        d2 = librosa.feature.delta(m, order=2)
        return self._trim(np.concatenate([m, d1, d2], axis=0))

    def _mel(self, wav):
        mel = librosa.feature.melspectrogram(
            y=wav, sr=SR, n_mels=N_MELS, n_fft=N_FFT,
            hop_length=HOP_LENGTH, win_length=WIN_LENGTH, fmax=FMAX)
        return self._trim(librosa.power_to_db(mel, ref=np.max))

    def _extras(self, wav):
        kw = dict(hop_length=HOP_LENGTH)
        return self._trim(np.stack([
            librosa.feature.zero_crossing_rate(wav, **kw)[0],
            librosa.feature.rms(y=wav, **kw)[0],
            librosa.feature.spectral_centroid(y=wav, sr=SR, **kw)[0],
            librosa.feature.spectral_bandwidth(y=wav, sr=SR, **kw)[0],
            librosa.feature.spectral_rolloff(y=wav, sr=SR, **kw)[0],
        ]))

    def _trim(self, feat):
        T = feat.shape[-1]
        if T < FRAMES:
            feat = np.pad(feat, ((0,0),(0, FRAMES-T)))
        return feat[..., :FRAMES]
