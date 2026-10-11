"""Step 3: dataset summary figures and tables for the report."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plotstyle
from config import CLASSES, FIGURES, METADATA_CSV, RESULTS

SPLITS = ["train", "val", "test"]


def class_distribution(df):
    counts = pd.crosstab(df["emotion"], df["dataset"]).reindex(CLASSES)[["CREMA-D", "RAVDESS"]]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(CLASSES))
    w = 0.38
    for i, (ds, color) in enumerate(zip(counts.columns, plotstyle.SERIES)):
        ax.bar(x + (i - 0.5) * (w + 0.02), counts[ds], width=w, color=color, label=ds)
    ax.set_xticks(x, [c.capitalize() for c in CLASSES])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Clips")
    ax.set_title("Clips per emotion class and dataset")
    ax.set_ylim(0, 1500)
    ax.legend(loc="upper right", ncols=2)
    fig.savefig(FIGURES / "01_class_distribution.png")
    plt.close(fig)


def escalate_by_split(df):
    tab = pd.crosstab(df["split"], df["escalate"]).reindex(SPLITS)
    share = tab[1] / tab.sum(axis=1)
    fig, ax = plt.subplots(figsize=(7, 2.6))
    y = np.arange(len(SPLITS))[::-1]
    ax.barh(y, tab[0], color=plotstyle.NO_ESCALATE, height=0.55, label="No-escalate (neutral, happy)")
    ax.barh(y, tab[1], left=tab[0] + 25, color=plotstyle.ESCALATE, height=0.55,
            label="Escalate (angry, disgust, fear, sad)")
    for yi, split in zip(y, SPLITS):
        total = tab.loc[split].sum()
        ax.text(total + 60, yi, f"{total:,} clips · {share[split]:.0%} escalate",
                va="center", fontsize=9, color=plotstyle.TEXT_2)
    ax.set_yticks(y, [s.capitalize() for s in SPLITS])
    ax.set_xlim(0, tab.sum(axis=1).max() * 1.45)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Clips")
    ax.set_title("Speaker-independent split and Escalate share")
    ax.legend(loc="upper left", ncols=2, bbox_to_anchor=(0, -0.3))
    fig.savefig(FIGURES / "02_split_escalate.png")
    plt.close(fig)


def duration_hist(df):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bins = np.arange(1.0, 5.6, 0.1)
    for ds, color in zip(["CREMA-D", "RAVDESS"], plotstyle.SERIES):
        ax.hist(df.loc[df["dataset"] == ds, "duration"], bins=bins, color=color, alpha=0.85,
                label=ds, edgecolor=plotstyle.SURFACE, linewidth=0.8)
    ax.axvline(3.0, color=plotstyle.TEXT_2, linewidth=1.2, linestyle="--")
    ax.text(3.05, ax.get_ylim()[1] * 0.92, "3 s model input", fontsize=9, color=plotstyle.TEXT_2)
    ax.set_xlabel("Clip duration (seconds, before silence trimming)")
    ax.set_ylabel("Clips")
    ax.set_title("Clip duration by dataset")
    ax.legend(loc="upper right")
    fig.savefig(FIGURES / "03_duration_hist.png")
    plt.close(fig)


def summary_tables(df):
    split_class = pd.crosstab(df["split"], df["emotion"]).reindex(index=SPLITS, columns=CLASSES)
    split_class["total"] = split_class.sum(axis=1)
    speakers = df.groupby(["split", "dataset"])["speaker"].nunique().unstack().reindex(SPLITS)
    split_class.join(speakers.add_prefix("speakers_")).to_csv(RESULTS / "split_summary.csv")


if __name__ == "__main__":
    plotstyle.apply()
    FIGURES.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(METADATA_CSV)
    class_distribution(meta)
    escalate_by_split(meta)
    duration_hist(meta)
    summary_tables(meta)
    print("Figures saved to", FIGURES)
