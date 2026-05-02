"""
model/charts.py
Generates all chart data as plain Python lists/dicts.
The frontend (Chart.js) draws everything — no matplotlib on server.
This keeps the Vercel function small and fast.
"""

import numpy as np
import librosa

SR         = 16000
DURATION   = 4.0
N_MELS     = 80
N_MFCC     = 40
N_FFT      = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
FMAX       = 8000
WAVEFORM_POINTS = 400   # downsample waveform to this many points


def _ref_real(seed=42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    t   = np.linspace(0, DURATION, int(SR * DURATION))
    wav = (np.sin(2*np.pi*200*t)*.28 + np.sin(2*np.pi*450*t)*.18
           + rng.randn(len(t))*.30)
    return (wav * np.sin(np.pi * t / DURATION)).astype(np.float32)

def _ref_fake(seed=77) -> np.ndarray:
    rng = np.random.RandomState(seed)
    t   = np.linspace(0, DURATION, int(SR * DURATION))
    wav = (np.sin(2*np.pi*200*t)*.35 + np.sin(2*np.pi*400*t)*.35
           + rng.randn(len(t))*.04)
    return (wav * np.sin(np.pi * t / DURATION)).astype(np.float32)


def _downsample_waveform(wav: np.ndarray, n: int = WAVEFORM_POINTS) -> list:
    """Reduce waveform to n evenly-spaced amplitude values."""
    indices = np.linspace(0, len(wav)-1, n, dtype=int)
    return [round(float(wav[i]), 5) for i in indices]


def _mel_grid(wav: np.ndarray, cols: int = 80, rows: int = 40) -> list:
    """
    Returns a (rows × cols) grid of mel energy values [0..1].
    Frontend maps this to an inferno-style heatmap.
    """
    mel = librosa.feature.melspectrogram(
        y=wav, sr=SR, n_mels=rows, n_fft=N_FFT,
        hop_length=HOP_LENGTH, fmax=FMAX)
    log_mel = librosa.power_to_db(mel, ref=np.max)   # (rows, T)
    # Normalise to [0, 1]
    lo, hi  = log_mel.min(), log_mel.max()
    norm    = (log_mel - lo) / (hi - lo + 1e-8)
    # Resize to (rows, cols)
    from scipy.ndimage import zoom
    factor = (rows / norm.shape[0], cols / norm.shape[1])
    try:
        from scipy.ndimage import zoom as _zoom
        resized = _zoom(norm, factor, order=1)
    except Exception:
        # Manual nearest-neighbour fallback
        ri = np.linspace(0, norm.shape[0]-1, rows).astype(int)
        ci = np.linspace(0, norm.shape[1]-1, cols).astype(int)
        resized = norm[np.ix_(ri, ci)]
    return resized.clip(0,1).round(3).tolist()


def _mfcc_grid(wav: np.ndarray, cols: int = 60) -> list:
    """
    Returns a (40 × cols) MFCC grid normalised to [-1, 1].
    """
    mfcc = librosa.feature.mfcc(y=wav, sr=SR, n_mfcc=N_MFCC,
                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    # Normalise per coefficient
    mu  = mfcc.mean(1, keepdims=True)
    std = mfcc.std(1,  keepdims=True) + 1e-8
    norm = np.clip((mfcc - mu) / std / 3, -1, 1)   # (40, T)
    # Resize
    ri = np.linspace(0, norm.shape[0]-1, N_MFCC).astype(int)
    ci = np.linspace(0, norm.shape[1]-1, cols).astype(int)
    return norm[np.ix_(ri, ci)].round(3).tolist()


def _freq_spectrum(wav: np.ndarray) -> dict:
    """Power spectrum at fixed octave-band frequencies."""
    fft   = np.abs(np.fft.rfft(wav, n=N_FFT))
    freqs = np.fft.rfftfreq(N_FFT, 1/SR)
    power = 20 * np.log10(fft + 1e-10)

    bands = [100, 200, 400, 800, 1600, 3200, 6400, 12800]
    out   = []
    for f in bands:
        idx = np.argmin(np.abs(freqs - f))
        out.append(round(float(power[idx]), 2))
    return {"labels": ["100", "200", "400", "800", "1.6k", "3.2k", "6.4k", "12.8k"],
            "values": out}


def _band_energy(wav: np.ndarray) -> list:
    """Average power (dB) in 6 frequency bands."""
    spec  = np.abs(librosa.stft(wav, n_fft=N_FFT, hop_length=HOP_LENGTH))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    power = 20 * np.log10(spec.mean(1) + 1e-10)

    boundaries = [(0,300),(300,1000),(1000,3000),(3000,6000),(6000,12000),(12000,SR//2)]
    out = []
    for lo, hi in boundaries:
        mask = (freqs >= lo) & (freqs < hi)
        out.append(round(float(power[mask].mean()) if mask.any() else 0, 2))
    return out


class ChartGenerator:
    """Generates all chart data for one request."""

    def generate_all(self, wav: np.ndarray, is_fake: bool) -> dict:
        real_ref = _ref_real()
        fake_ref = _ref_fake()

        return {
            # ── Waveforms ─────────────────────────────────────
            "waveforms": {
                "user" : _downsample_waveform(wav),
                "real" : _downsample_waveform(real_ref),
                "fake" : _downsample_waveform(fake_ref),
            },

            # ── Mel spectrograms ──────────────────────────────
            "spectrograms": {
                "user" : _mel_grid(wav),
                "real" : _mel_grid(real_ref),
                "fake" : _mel_grid(fake_ref),
            },

            # ── MFCC heatmaps ─────────────────────────────────
            "mfcc": {
                "user" : _mfcc_grid(wav),
                "real" : _mfcc_grid(real_ref),
                "fake" : _mfcc_grid(fake_ref),
            },

            # ── Frequency analysis ────────────────────────────
            "frequency": {
                "spectrum": {
                    "user" : _freq_spectrum(wav),
                    "real" : _freq_spectrum(real_ref),
                    "fake" : _freq_spectrum(fake_ref),
                },
                "bands": {
                    "labels" : ["0–300Hz","300–1k","1k–3k","3k–6k","6k–12k","12k+"],
                    "user"   : _band_energy(wav),
                    "real"   : _band_energy(real_ref),
                    "fake"   : _band_energy(fake_ref),
                },
            },
        }
