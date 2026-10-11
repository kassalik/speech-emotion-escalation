"""Lightweight audio front-end (NumPy / SciPy only) mirroring the training pipeline in src/.

Kept dependency-free of PyTorch and librosa so it runs on small devices (e.g. Raspberry Pi 5).
"""
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt


def load_audio(path_or_array, sr=None, target_sr=16000):
    """File path or (array, sr) -> float32 mono at target_sr."""
    if isinstance(path_or_array, (str, bytes)) or hasattr(path_or_array, "__fspath__"):
        y, sr = sf.read(path_or_array, dtype="float32", always_2d=True)
        y = y.mean(axis=1)
    else:
        y = np.asarray(path_or_array, dtype=np.float32)
        if y.ndim == 2:  # (samples, channels) or (channels, samples)
            y = y.mean(axis=1 if y.shape[0] > y.shape[1] else 0)
        if np.issubdtype(np.asarray(path_or_array).dtype, np.integer):
            y = y / np.iinfo(np.asarray(path_or_array).dtype).max
    return resample(y, sr, target_sr)


def resample(y, sr_in, sr_out):
    if sr_in == sr_out:
        return y.astype(np.float32)
    g = gcd(int(sr_in), int(sr_out))
    return resample_poly(y, sr_out // g, sr_in // g).astype(np.float32)


def _frame_rms_db(y, frame=400, hop=160):
    """librosa-compatible centred frame RMS in dB relative to the loudest frame (amin 1e-5, top_db 80)."""
    yp = np.pad(y, frame // 2)
    n = 1 + max(len(yp) - frame, 0) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(yp[idx] ** 2, axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-5)) - 20 * np.log10(max(rms.max(), 1e-5))
    return np.maximum(db, db.max() - 80.0)


def trim_silence(y, margin_db=8.0, top_db_range=(15.0, 40.0), frame=400, hop=160):
    """Trim leading/trailing silence with a threshold adapted to the clip's own noise floor."""
    db = _frame_rms_db(y, frame, hop)
    top_db = float(np.clip(-np.percentile(db, 10) - margin_db, *top_db_range))
    voiced = np.flatnonzero(db > -top_db)
    if len(voiced) == 0:
        return y[:0]
    return y[voiced[0] * hop: min(len(y), (voiced[-1] + 1) * hop)]


def normalise_loudness(y, target_dbfs=-23.0, peak_limit=0.99):
    y = y * (10 ** (target_dbfs / 20) / (np.sqrt(np.mean(y ** 2)) + 1e-9))
    peak = np.max(np.abs(y)) if len(y) else 0.0
    return y * (peak_limit / peak) if peak > peak_limit else y


def fix_length(y, n):
    """Centre crop or symmetric zero-pad; returns (clip, speech_start, speech_length)."""
    if len(y) >= n:
        start = (len(y) - n) // 2
        return y[start:start + n], 0, n
    left = (n - len(y)) // 2
    return np.pad(y, (left, n - len(y) - left)), left, len(y)


def telephone(y, sr=16000, phone_sr=8000, band=(300, 3400), mu=255):
    """Narrow-band telephone channel: 8 kHz, band-pass, G.711 mu-law 8-bit, back to sr."""
    nb = resample(y, sr, phone_sr).astype(np.float64)
    nb = sosfiltfilt(butter(4, band, btype="bandpass", fs=phone_sr, output="sos"), nb)
    nb = np.clip(nb, -1, 1)
    nb = np.sign(nb) * np.log1p(mu * np.abs(nb)) / np.log1p(mu)
    nb = np.round((nb + 1) / 2 * 255) / 255 * 2 - 1
    nb = np.sign(nb) * ((1 + mu) ** np.abs(nb) - 1) / mu
    out = resample(nb, phone_sr, sr)[:len(y)]
    return np.pad(out, (0, len(y) - len(out))).astype(np.float32)


def logmel_db(y, mel_fb, window, n_fft=400, hop=160, top_db=80.0):
    """Power log-mel in dB, identical in definition to torchaudio MelSpectrogram + AmplitudeToDB(top_db)."""
    yp = np.pad(y.astype(np.float32), n_fft // 2, mode="reflect")
    n = 1 + (len(yp) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    power = np.abs(np.fft.rfft(yp[idx] * window, n=n_fft, axis=1)) ** 2   # [frames, n_fft/2+1]
    mel = power @ mel_fb                                                  # [frames, n_mels]
    db = 10 * np.log10(np.maximum(mel, 1e-10)).T                          # [n_mels, frames]
    return np.maximum(db, db.max() - top_db).astype(np.float32)
