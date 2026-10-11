"""Step 8 (c): grouped permutation importance of handcrafted feature families.

Each family's columns are shuffled together (same row permutation) on the clean validation set;
importance = drop in macro-F1, averaged over repeats.
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import plotstyle
from baseline import MODELS_DIR, load_split
from config import FIGURES, RESULTS, SEED

REPEATS = 10
FAMILIES = {
    "MFCC means": lambda c: c.startswith("mfcc") and not c.startswith("mfcc_d") and c.endswith("_mean"),
    "MFCC std": lambda c: c.startswith("mfcc") and not c.startswith("mfcc_d") and c.endswith("_std"),
    "MFCC deltas (Δ)": lambda c: c.startswith("mfcc_d") and not c.startswith("mfcc_dd"),
    "MFCC delta-deltas (ΔΔ)": lambda c: c.startswith("mfcc_dd"),
    "Pitch F0": lambda c: c.startswith("f0_"),
    "Voicing": lambda c: c == "voiced_frac",
    "Energy (RMS)": lambda c: c.startswith("rms_db"),
    "Zero-crossing rate": lambda c: c.startswith("zcr"),
    "Spectral shape": lambda c: c.split("_")[0] in {"centroid", "rolloff", "bandwidth", "flatness"},
}


def grouped_importance(model, X, y, rng):
    base = f1_score(y, model.predict(X), average="macro")
    rows = []
    for fam, rule in FAMILIES.items():
        cols = [c for c in X.columns if rule(c)]
        drops = []
        for _ in range(REPEATS):
            Xp = X.copy()
            Xp[cols] = X[cols].values[rng.permutation(len(X))]
            drops.append(base - f1_score(y, model.predict(Xp), average="macro"))
        rows.append({"family": fam, "n_features": len(cols), "f1_drop_mean": np.mean(drops),
                     "f1_drop_std": np.std(drops)})
    return base, pd.DataFrame(rows)


def plot(tables):
    order = tables["svm"].sort_values("f1_drop_mean")["family"].tolist()
    fig, ax = plt.subplots(figsize=(8, 4.4))
    y = np.arange(len(order))
    h = 0.38
    for k, (name, label) in enumerate([("svm", "SVM"), ("rf", "Random Forest")]):
        t = tables[name].set_index("family").loc[order]
        ax.barh(y + (0.5 - k) * (h + 0.02), t["f1_drop_mean"], height=h, xerr=t["f1_drop_std"],
                color=plotstyle.SERIES[k], label=label, error_kw={"ecolor": plotstyle.TEXT_2, "lw": 0.8})
    ax.set_yticks(y, [f"{f} ({int(tables['svm'].set_index('family').loc[f, 'n_features'])})" for f in order])
    ax.axvline(0, color=plotstyle.TEXT_2, lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Drop in validation macro-F1 when the feature family is shuffled")
    ax.set_title("Which handcrafted cues the baselines rely on (clean validation audio)")
    ax.legend(loc="lower right")
    fig.savefig(FIGURES / "13_feature_importance.png")
    plt.close(fig)


def main():
    plotstyle.apply()
    X, y, _ = load_split("val", "clean")
    rng = np.random.default_rng(SEED)
    tables = {}
    for name in ("svm", "rf"):
        model = joblib.load(MODELS_DIR / f"{name}.joblib")
        base, t = grouped_importance(model, X, y, rng)
        t.insert(0, "model", name)
        tables[name] = t
        print(f"{name}: baseline val macro-F1 {base:.3f}")
        print(t.sort_values("f1_drop_mean", ascending=False).round(4).to_string(index=False))
    pd.concat(tables.values()).to_csv(RESULTS / "feature_importance.csv", index=False)
    plot(tables)


if __name__ == "__main__":
    main()
