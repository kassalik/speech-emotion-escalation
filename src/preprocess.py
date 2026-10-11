"""Step 4: audio pre-processing.

Every clip -> 16 kHz mono -> trim leading/trailing silence (noise-floor adaptive)
-> RMS loudness normalisation -> centre crop / zero-pad to 3 s.  A second "telephone" version
simulates a narrow-band call: 8 kHz resample, 300-3400 Hz band-pass, G.711 mu-law companding,
back to 16 kHz.

Outputs (row order = metadata.csv):
  data/processed/audio_clean.npy   float16 [N, 48000]
  data/processed/audio_phone.npy   float16 [N, 48000]
  data/processed/audio_index.csv   speech start/length inside the 3 s window
"""
import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import butter, sosfiltfilt

from config import CLIP_SECONDS, METADATA_CSV, PROCESSED, ROOT, SAMPLE_RATE

N_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)
TRIM_MARGIN_DB = 8       # edge frames less than 8 dB above the clip's noise floor count as silence
TRIM_TOP_DB_RANGE = (15, 40)
TARGET_RMS_DBFS = -23.0   # loudness target (speech crest factor is ~15-25 dB)
PEAK_LIMIT = 0.99
PHONE_SR = 8000
PHONE_BAND = (300, 3400)
MU = 255                  # G.711 mu-law

_PHONE_SOS = butter(4, PHONE_BAND, btype="bandpass", fs=PHONE_SR, output="sos")


def trim_silence(y):
    """Trim edges with a threshold adapted to the clip's own noise floor.

    CREMA-D has a noise floor only ~25 dB below the speech peak, so a fixed top_db misses it.
    """
    db = librosa.amplitude_to_db(librosa.feature.rms(y=y, frame_length=400, hop_length=160)[0], ref=np.max)
    floor = np.percentile(db, 10)
    top_db = float(np.clip(-floor - TRIM_MARGIN_DB, *TRIM_TOP_DB_RANGE))
    trimmed, _ = librosa.effects.trim(y, top_db=top_db, frame_length=400, hop_length=160)
    return trimmed


def normalise_loudness(y):
    rms = np.sqrt(np.mean(y ** 2)) + 1e-9
    y = y * (10 ** (TARGET_RMS_DBFS / 20) / rms)
    peak = np.max(np.abs(y))
    return y * (PEAK_LIMIT / peak) if peak > PEAK_LIMIT else y


def fix_length(y, n=N_SAMPLES):
    """Centre crop or symmetric zero-pad; returns (clip, speech_start, speech_length)."""
    if len(y) >= n:
        start = (len(y) - n) // 2
        return y[start:start + n], 0, n
    pad = n - len(y)
    left = pad // 2
    return np.pad(y, (left, pad - left)), left, len(y)


def telephone(y):
    """Narrow-band telephone channel simulation on a 16 kHz signal."""
    nb = librosa.resample(y, orig_sr=SAMPLE_RATE, target_sr=PHONE_SR, res_type="soxr_hq")
    nb = sosfiltfilt(_PHONE_SOS, nb)
    nb = np.clip(nb, -1, 1)
    nb = np.sign(nb) * np.log1p(MU * np.abs(nb)) / np.log1p(MU)          # mu-law encode
    nb = np.round((nb + 1) / 2 * 255) / 255 * 2 - 1                      # 8-bit quantise
    nb = np.sign(nb) * ((1 + MU) ** np.abs(nb) - 1) / MU                 # mu-law decode
    return librosa.resample(nb, orig_sr=PHONE_SR, target_sr=SAMPLE_RATE, res_type="soxr_hq")


def process_clip(path):
    y, _ = librosa.load(ROOT / path, sr=SAMPLE_RATE, mono=True, res_type="soxr_hq")
    y = trim_silence(y)
    trimmed_sec = len(y) / SAMPLE_RATE
    y = normalise_loudness(y)
    clean, start, length = fix_length(y)
    phone = telephone(clean)[:N_SAMPLES]
    phone = np.pad(phone, (0, N_SAMPLES - len(phone)))
    # keep the speech region at the same loudness as the clean version
    seg = slice(start, start + length)
    phone *= np.sqrt(np.mean(clean[seg] ** 2)) / (np.sqrt(np.mean(phone[seg] ** 2)) + 1e-9)
    phone = np.clip(phone, -PEAK_LIMIT, PEAK_LIMIT)
    return clean.astype(np.float16), phone.astype(np.float16), start, length, trimmed_sec


def run(n_jobs=-1, chunk=500):
    meta = pd.read_csv(METADATA_CSV)
    n = len(meta)
    clean_mm = np.lib.format.open_memmap(PROCESSED / "audio_clean.npy", "w+", np.float16, (n, N_SAMPLES))
    phone_mm = np.lib.format.open_memmap(PROCESSED / "audio_phone.npy", "w+", np.float16, (n, N_SAMPLES))
    index = []
    with Parallel(n_jobs=n_jobs) as pool:
        for lo in range(0, n, chunk):
            out = pool(delayed(process_clip)(p) for p in meta["path"].iloc[lo:lo + chunk])
            for i, (c, ph, start, length, tsec) in enumerate(out, lo):
                clean_mm[i], phone_mm[i] = c, ph
                index.append({"speech_start": start, "speech_length": length, "trimmed_sec": round(tsec, 3)})
            print(f"  {min(lo + chunk, n)}/{n} clips", flush=True)
    clean_mm.flush(); phone_mm.flush()
    idx = pd.DataFrame(index)
    idx.insert(0, "path", meta["path"])
    idx.to_csv(PROCESSED / "audio_index.csv", index=False)
    return idx


if __name__ == "__main__":
    run()
