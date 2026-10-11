"""Step 5: feature figures for the report."""
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plotstyle
from config import CLASSES, ESCALATE_CLASSES, FIGURES, METADATA_CSV, PROCESSED, RESULTS, SAMPLE_RATE
from features import HOP, N_MELS, N_MFCC

PAIR = [("data/raw/crema-d/1001_DFA_NEU_XX.wav", "Neutral"), ("data/raw/crema-d/1001_DFA_ANG_XX.wav", "Angry")]
BOX_FEATURES = [("f0_mean", "Mean pitch F0 (Hz)"), ("f0_std", "Pitch variability F0 std (Hz)"),
                ("rms_db_std", "Energy variability (dB std)"), ("centroid_mean", "Spectral centroid (Hz)")]


def angry_vs_neutral(meta, logmel, clean, idx):
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.8))
    for col, (path, label) in enumerate(PAIR):
        i = meta.index[meta["path"] == path][0]
        s, n = idx.speech_start[i], idx.speech_length[i]
        frames = slice(s // HOP, (s + n) // HOP + 1)
        ax = axes[0, col]
        im = ax.imshow(logmel[i][:, frames].astype(np.float32), origin="lower", aspect="auto",
                       cmap="magma", extent=[0, n / SAMPLE_RATE, 0, N_MELS])
        ax.set_title(f"{label} · log-mel spectrogram", fontsize=10)
        ax.set_ylabel("Mel band")
        ax.grid(False)
        y = clean[i, s:s + n].astype(np.float32)
        mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=400, hop_length=HOP, n_mels=N_MELS)
        ax = axes[1, col]
        im2 = ax.imshow(mfcc[1:], origin="lower", aspect="auto", cmap="coolwarm", vmin=-60, vmax=60,
                        extent=[0, n / SAMPLE_RATE, 1, N_MFCC])
        ax.set_title(f"{label} · MFCC 1–39", fontsize=10)
        ax.set_ylabel("MFCC coefficient")
        ax.set_xlabel("Time (s)")
        ax.grid(False)
    fig.suptitle("Neutral vs angry: same speaker (CREMA-D 1001), same sentence, speech region only",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.colorbar(im, ax=axes[0], label="dB", fraction=0.02, pad=0.01)
    fig.colorbar(im2, ax=axes[1], label="Coefficient value", fraction=0.02, pad=0.01)
    fig.savefig(FIGURES / "06_angry_vs_neutral_features.png")
    plt.close(fig)


def feature_boxplots(meta, hc):
    train = meta["split"] == "train"
    fig, axes = plt.subplots(1, len(BOX_FEATURES), figsize=(13, 3.6))
    colors = [plotstyle.ESCALATE if c in ESCALATE_CLASSES else plotstyle.NO_ESCALATE for c in CLASSES]
    for ax, (feat, title) in zip(axes, BOX_FEATURES):
        data = [hc.loc[train & (meta["emotion"] == c), feat].dropna() for c in CLASSES]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                        medianprops={"color": plotstyle.SURFACE, "linewidth": 1.6},
                        whiskerprops={"color": plotstyle.TEXT_2}, capprops={"color": plotstyle.TEXT_2})
        for patch, c in zip(bp["boxes"], colors):
            patch.set(facecolor=c, edgecolor=c)
        ax.set_xticks(range(1, len(CLASSES) + 1), [c[:3].capitalize() for c in CLASSES])
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=plotstyle.NO_ESCALATE),
               plt.Rectangle((0, 0), 1, 1, color=plotstyle.ESCALATE)]
    fig.legend(handles, ["No-escalate", "Escalate"], loc="upper right", ncols=2, bbox_to_anchor=(1, 1.04))
    fig.suptitle("Prosodic and spectral cues by emotion (training set, clean audio)", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "07_feature_boxplots.png")
    plt.close(fig)


def feature_summary(meta, hc_clean, hc_phone):
    """Per-emotion medians of key cues, and the clean-vs-phone shift of each cue."""
    cols = [f for f, _ in BOX_FEATURES] + ["voiced_frac", "zcr_mean"]
    train = meta["split"] == "train"
    by_emotion = hc_clean[train].groupby(meta.loc[train, "emotion"])[cols].median().reindex(CLASSES)
    shift = pd.DataFrame({"clean_median": hc_clean[cols].median(), "phone_median": hc_phone[cols].median()})
    by_emotion.round(2).to_csv(RESULTS / "feature_medians_by_emotion.csv")
    shift.round(2).to_csv(RESULTS / "feature_shift_clean_vs_phone.csv")
    return by_emotion, shift


def main():
    plotstyle.apply()
    meta = pd.read_csv(METADATA_CSV)
    idx = pd.read_csv(PROCESSED / "audio_index.csv")
    logmel = np.load(PROCESSED / "logmel_clean.npy", mmap_mode="r")
    clean = np.load(PROCESSED / "audio_clean.npy", mmap_mode="r")
    angry_vs_neutral(meta, logmel, clean, idx)
    hc_clean = pd.read_csv(PROCESSED / "handcrafted_clean.csv").drop(columns="path")
    hc_phone = pd.read_csv(PROCESSED / "handcrafted_phone.csv").drop(columns="path")
    feature_boxplots(meta, hc_clean)
    by_emotion, shift = feature_summary(meta, hc_clean, hc_phone)
    print(by_emotion.round(2).to_string())
    print(shift.round(2).to_string())


if __name__ == "__main__":
    main()
