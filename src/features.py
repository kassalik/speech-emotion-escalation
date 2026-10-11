"""Step 5: feature extraction.

Handcrafted (baseline models): per-clip mean/std of 40 MFCC + delta + delta-delta, plus prosodic
and spectral statistics, computed on the speech region only (zero padding excluded).
Log-mel (CNN): 64-band log-mel spectrogram of the full 3 s window, with the same torchaudio
transform the CNN uses at training time.

No scaling is fitted here; scalers are fitted on the training split only in Steps 6-7.

Outputs (row order = metadata.csv), for VERSION in {clean, phone}:
  data/processed/handcrafted_{VERSION}.csv   [N, 259] + path
  data/processed/logmel_{VERSION}.npy        float16 [N, 64, 301]
"""
import sys

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from joblib import Parallel, delayed

from config import METADATA_CSV, PROCESSED, SAMPLE_RATE

N_FFT = 400        # 25 ms window
HOP = 160          # 10 ms hop
N_MELS = 64
N_MFCC = 40
F0_RANGE = (65, 500)
VERSIONS = ("clean", "phone")


def make_logmel(device="cpu"):
    """Waveform [B, T] -> log-mel [B, 64, frames] in dB, floored 80 dB below each clip's peak."""
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, win_length=N_FFT, hop_length=HOP,
        f_min=20.0, f_max=SAMPLE_RATE / 2, n_mels=N_MELS, power=2.0).to(device)
    to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0).to(device)

    def transform(wave):
        return torch.stack([to_db(m) for m in mel(wave)])  # per-clip top_db floor
    return transform


def _stats(name, x):
    return {f"{name}_mean": float(np.mean(x)), f"{name}_std": float(np.std(x))}


def handcrafted(y):
    """Feature dict for one speech segment (float32, 16 kHz)."""
    feats = {}
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(librosa.feature.melspectrogram(
        S=S, sr=SAMPLE_RATE, n_mels=N_MELS)), n_mfcc=N_MFCC)
    width = min(9, mfcc.shape[1] - (1 - mfcc.shape[1] % 2))  # delta needs an odd width <= frames
    d1 = librosa.feature.delta(mfcc, width=width)
    d2 = librosa.feature.delta(mfcc, order=2, width=width)
    for prefix, M in (("mfcc", mfcc), ("mfcc_d", d1), ("mfcc_dd", d2)):
        for k in range(N_MFCC):
            feats.update(_stats(f"{prefix}{k:02d}", M[k]))

    f0, voiced, _ = librosa.pyin(y, fmin=F0_RANGE[0], fmax=F0_RANGE[1], sr=SAMPLE_RATE,
                                 frame_length=1024, hop_length=HOP)
    f0v = f0[voiced & np.isfinite(f0)]
    if len(f0v) >= 3:
        p5, p95 = np.percentile(f0v, [5, 95])
        feats.update({"f0_mean": f0v.mean(), "f0_std": f0v.std(), "f0_p5": p5, "f0_p95": p95,
                      "f0_range": p95 - p5})
    else:  # no reliable pitch (e.g. whispered or very short clip)
        feats.update({"f0_mean": np.nan, "f0_std": np.nan, "f0_p5": np.nan, "f0_p95": np.nan,
                      "f0_range": np.nan})
    feats["voiced_frac"] = float(np.mean(voiced))

    mag = np.sqrt(S)
    rms_db = librosa.amplitude_to_db(librosa.feature.rms(S=mag, frame_length=N_FFT), ref=1.0)[0]
    feats.update(_stats("rms_db", rms_db))
    feats["rms_db_p95"] = float(np.percentile(rms_db, 95))
    feats.update(_stats("zcr", librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)[0]))
    feats.update(_stats("centroid", librosa.feature.spectral_centroid(S=mag, sr=SAMPLE_RATE)[0]))
    feats.update(_stats("rolloff", librosa.feature.spectral_rolloff(S=mag, sr=SAMPLE_RATE)[0]))
    feats.update(_stats("bandwidth", librosa.feature.spectral_bandwidth(S=mag, sr=SAMPLE_RATE)[0]))
    feats.update(_stats("flatness", librosa.feature.spectral_flatness(S=mag)[0]))
    return feats


def _clip_features(npy, i, start, length):
    audio = np.load(npy, mmap_mode="r")
    return handcrafted(audio[i, start:start + length].astype(np.float32))


def extract_handcrafted(version, n_jobs=-1):
    meta = pd.read_csv(METADATA_CSV)
    idx = pd.read_csv(PROCESSED / "audio_index.csv")
    npy = PROCESSED / f"audio_{version}.npy"
    rows = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_clip_features)(npy, i, s, l)
        for i, (s, l) in enumerate(zip(idx["speech_start"], idx["speech_length"])))
    df = pd.DataFrame(rows)
    df.insert(0, "path", meta["path"])
    df.to_csv(PROCESSED / f"handcrafted_{version}.csv", index=False)
    return df


def extract_logmel(version, batch=256):
    audio = np.load(PROCESSED / f"audio_{version}.npy", mmap_mode="r")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = make_logmel(device)
    n_frames = audio.shape[1] // HOP + 1
    out = np.lib.format.open_memmap(PROCESSED / f"logmel_{version}.npy", "w+", np.float16,
                                    (len(audio), N_MELS, n_frames))
    with torch.no_grad():
        for lo in range(0, len(audio), batch):
            wave = torch.from_numpy(audio[lo:lo + batch].astype(np.float32)).to(device)
            out[lo:lo + batch] = transform(wave).cpu().numpy().astype(np.float16)
    out.flush()
    return out.shape


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for v in VERSIONS:
        if which in ("all", "logmel"):
            print(f"log-mel {v}:", extract_logmel(v), flush=True)
        if which in ("all", "handcrafted"):
            print(f"handcrafted {v}:", extract_handcrafted(v).shape, flush=True)
