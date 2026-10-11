"""Step 4: before/after figures and example audio for the pre-processing pipeline."""
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

import plotstyle
from config import FIGURES, METADATA_CSV, PROCESSED, RESULTS, ROOT, SAMPLE_RATE
from preprocess import trim_silence

EXAMPLES = ["data/raw/crema-d/1001_DFA_ANG_XX.wav", "data/raw/ravdess/Actor_01/03-01-05-02-01-01-01.wav"]


def _spec(ax, y, title):
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=160)), ref=np.max)
    librosa.display.specshow(S, sr=SAMPLE_RATE, hop_length=160, x_axis="time", y_axis="hz",
                             cmap="magma", vmin=-80, vmax=0, ax=ax)
    ax.set_ylim(0, 8000)
    ax.set_title(title, fontsize=10)
    ax.grid(False)


def before_after(meta, clean, phone):
    import librosa.display  # noqa: F401
    fig, axes = plt.subplots(len(EXAMPLES), 3, figsize=(12, 3.0 * len(EXAMPLES)))
    for row, path in zip(axes, EXAMPLES):
        i = meta.index[meta["path"] == path][0]
        raw, sr = librosa.load(ROOT / path, sr=None, mono=True)
        raw16 = librosa.resample(raw, orig_sr=sr, target_sr=SAMPLE_RATE)
        kept = trim_silence(raw16)
        start = next(k for k in range(len(raw16)) if np.array_equal(raw16[k:k + len(kept)], kept))
        t = np.arange(len(raw16)) / SAMPLE_RATE
        ax = row[0]
        ax.plot(t, raw16, color=plotstyle.SERIES[0], linewidth=0.5)
        ax.axvspan(0, start / SAMPLE_RATE, color=plotstyle.GRID, alpha=0.9, lw=0)
        ax.axvspan((start + len(kept)) / SAMPLE_RATE, t[-1], color=plotstyle.GRID, alpha=0.9, lw=0)
        ax.set_title(f"Raw ({meta.dataset[i]}, {meta.emotion[i]}, {sr // 1000} kHz) · grey = trimmed",
                     fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.grid(False)
        _spec(row[1], clean[i].astype(np.float32), "Clean: 16 kHz, trimmed, normalised, 3 s")
        _spec(row[2], phone[i].astype(np.float32), "Telephone: 300–3400 Hz, mu-law")
        for ax in row[1:]:
            ax.set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(FIGURES / "04_preprocessing_before_after.png")
    plt.close(fig)


def length_hist(meta, idx):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bins = np.arange(0.4, 5.1, 0.1)
    for ds, color in zip(["CREMA-D", "RAVDESS"], plotstyle.SERIES):
        ax.hist(idx.loc[meta["dataset"] == ds, "trimmed_sec"], bins=bins, color=color, alpha=0.85,
                label=ds, edgecolor=plotstyle.SURFACE, linewidth=0.8)
    ax.axvline(3.0, color=plotstyle.TEXT_2, linewidth=1.2, linestyle="--")
    cropped = (idx["trimmed_sec"] > 3).mean()
    ax.text(3.05, ax.get_ylim()[1] * 0.92, f"3 s window · {cropped:.0%} cropped", fontsize=9,
            color=plotstyle.TEXT_2)
    ax.set_xlabel("Speech length after silence trimming (seconds)")
    ax.set_ylabel("Clips")
    ax.set_title("Speech length after trimming")
    ax.legend(loc="center right")
    fig.savefig(FIGURES / "05_trimmed_length_hist.png")
    plt.close(fig)


def example_audio(meta, clean, phone):
    out = RESULTS / "audio_examples"
    out.mkdir(parents=True, exist_ok=True)
    for path in EXAMPLES:
        i = meta.index[meta["path"] == path][0]
        stem = f"{meta.dataset[i]}_{meta.emotion[i]}_{meta.speaker[i]}"
        sf.write(out / f"{stem}_clean.wav", clean[i].astype(np.float32), SAMPLE_RATE)
        sf.write(out / f"{stem}_phone.wav", phone[i].astype(np.float32), SAMPLE_RATE)


def main():
    plotstyle.apply()
    meta = pd.read_csv(METADATA_CSV)
    idx = pd.read_csv(PROCESSED / "audio_index.csv")
    clean = np.load(PROCESSED / "audio_clean.npy", mmap_mode="r")
    phone = np.load(PROCESSED / "audio_phone.npy", mmap_mode="r")
    before_after(meta, clean, phone)
    length_hist(meta, idx)
    example_audio(meta, clean, phone)
    print("Saved figures and example audio")


if __name__ == "__main__":
    main()
