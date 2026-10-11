"""Step 8: final evaluation on the held-out test set + business mapping.

For every model the escalation threshold is chosen on the *validation* telephone audio as the
highest threshold that still catches >= 90% of Escalate clips, then applied unchanged to the test
set. CNN results are reported as mean +/- std over three seeds; seed 42 (fixed in advance) is the
"final model" used for the confusion matrix, trade-off curve and demo.
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import plotstyle
from baseline import ESC_IDX, MODELS_DIR, load_split, metrics_from_proba, plot_confusions
from config import CLASSES, FIGURES, METADATA_CSV, PROCESSED, RESULTS
from train_cnn import ARCHS, predict_proba

TARGET_RECALL = 0.90
CNN_SEEDS = [42, 7, 123]
FINAL = "cnn_s42"
MODELS = {  # key -> (display name, kind)
    "svm": ("SVM (clean-trained)", "sk"),
    "rf": ("Random Forest", "sk"),
    "svm_mc": ("SVM (multi-condition)", "sk"),
    **{f"cnn_s{s}": (f"CNN (seed {s})", "torch") for s in CNN_SEEDS},
    "crnn_s42": ("CNN + BiLSTM", "torch"),
}


def all_probas():
    """proba[model][split][version] -> [N, 6]"""
    meta = pd.read_csv(METADATA_CSV)
    out = {k: {"val": {}, "test": {}} for k in MODELS}
    for split in ("val", "test"):
        rows = np.where(meta["split"] == split)[0]
        for version in ("clean", "phone"):
            X, _, _ = load_split(split, version)
            mel = torch.from_numpy(np.load(PROCESSED / f"logmel_{version}.npy", mmap_mode="r")[rows]).cuda()
            for key, (_, kind) in MODELS.items():
                if kind == "sk":
                    out[key][split][version] = joblib.load(MODELS_DIR / f"{key}.joblib").predict_proba(X)
                else:
                    ck = torch.load(MODELS_DIR / f"{key}.pt", map_location="cuda")
                    model = ARCHS[ck["arch"]]().cuda()
                    model.load_state_dict(ck["state_dict"])
                    out[key][split][version] = predict_proba(model, mel, ck["mean"].cuda(), ck["std"].cuda())
    labels = {s: meta.loc[meta["split"] == s, "label"].values for s in ("val", "test")}
    return out, labels


def pick_threshold(p_esc, y_esc, target=TARGET_RECALL):
    """Highest threshold whose recall on these clips is >= target."""
    pos = np.sort(p_esc[y_esc == 1])
    return float(pos[int(np.floor((1 - target) * len(pos)))])


def business_metrics(p_esc, y_esc, t):
    flag = p_esc >= t
    n = len(y_esc)
    tp, fp = int((flag & (y_esc == 1)).sum()), int((flag & (y_esc == 0)).sum())
    fn = int((~flag & (y_esc == 1)).sum())
    return {"threshold": t, "recall": tp / (tp + fn), "precision": tp / max(tp + fp, 1),
            "alerts_per_100": 100 * flag.mean(), "false_alerts_per_100": 100 * fp / n,
            "missed_per_100": 100 * fn / n, "auc": roc_auc_score(y_esc, p_esc)}


def evaluate_all(proba, labels):
    y_esc = {s: np.isin(labels[s], ESC_IDX).astype(int) for s in labels}
    results = {}
    for key in MODELS:
        t = pick_threshold(proba[key]["val"]["phone"][:, ESC_IDX].sum(1), y_esc["val"])
        results[key] = {"threshold_from_val_phone": t}
        for version in ("clean", "phone"):
            p = proba[key]["test"][version]
            m = metrics_from_proba(p, labels["test"])
            m["business"] = business_metrics(p[:, ESC_IDX].sum(1), y_esc["test"], t)
            results[key][version] = m
    results["flag_every_call"] = {"recall": 1.0, "precision": float(y_esc["test"].mean()),
                                  "alerts_per_100": 100.0, "missed_per_100": 0.0,
                                  "false_alerts_per_100": 100 * float(1 - y_esc["test"].mean())}
    return results


def summary_table(results):
    rows = []

    def row(name, version, ms):
        def agg(f):
            vals = np.array([f(m) for m in ms])
            return vals.mean(), (vals.std(ddof=1) if len(vals) > 1 else np.nan)
        r = {"model": name, "audio": version}
        for col, f in [("accuracy", lambda m: m["accuracy"]), ("macro_f1", lambda m: m["macro_f1"]),
                       ("esc_auc", lambda m: m["business"]["auc"]),
                       ("esc_recall", lambda m: m["business"]["recall"]),
                       ("esc_precision", lambda m: m["business"]["precision"]),
                       ("alerts_per_100", lambda m: m["business"]["alerts_per_100"]),
                       ("false_alerts_per_100", lambda m: m["business"]["false_alerts_per_100"]),
                       ("missed_per_100", lambda m: m["business"]["missed_per_100"])]:
            r[col], r[col + "_std"] = agg(f)
        rows.append(r)

    for version in ("clean", "phone"):
        for key in ("svm", "rf", "svm_mc"):
            row(MODELS[key][0], version, [results[key][version]])
        row("CNN (3 seeds)", version, [results[f"cnn_s{s}"][version] for s in CNN_SEEDS])
        row("CNN + BiLSTM", version, [results["crnn_s42"][version]])
    return pd.DataFrame(rows)


def fig_model_comparison(table):
    models = ["SVM (clean-trained)", "Random Forest", "SVM (multi-condition)", "CNN (3 seeds)", "CNN + BiLSTM"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(models))
    w = 0.38
    for k, (version, color) in enumerate([("clean", plotstyle.SERIES[0]), ("phone", plotstyle.SERIES[1])]):
        t = table[table["audio"] == version].set_index("model").loc[models]
        xs = x + (k - 0.5) * (w + 0.02)
        ax.bar(xs, t["macro_f1"], width=w, color=color, label=f"{version.capitalize()} test audio")
        multi = t["macro_f1_std"].notna().values
        ax.errorbar(xs[multi], t["macro_f1"].values[multi], yerr=t["macro_f1_std"].values[multi], fmt="none",
                    ecolor=plotstyle.TEXT_2, lw=1, capsize=3)
        for xi, v in zip(x, t["macro_f1"]):
            ax.text(xi + (k - 0.5) * (w + 0.02), v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color=plotstyle.TEXT_2)
    ax.axhline(1 / 6, color=plotstyle.TEXT_2, ls="--", lw=1)
    ax.text(len(models) - 0.5, 1 / 6 + 0.01, "chance", ha="right", fontsize=8, color=plotstyle.TEXT_2)
    ax.set_xticks(x, ["SVM\n(clean-trained)", "Random\nForest", "SVM\n(multi-condition)", "CNN\n(mean of 3 seeds)", "CNN +\nBiLSTM"])
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("6-class macro-F1")
    ax.grid(axis="x", visible=False)
    ax.set_title("Held-out test set: clean vs telephone audio")
    ax.legend(loc="upper left", ncols=2)
    fig.savefig(FIGURES / "10_model_comparison_test.png")
    plt.close(fig)


def fig_tradeoff(proba, labels, results):
    y = np.isin(labels["test"], ESC_IDX).astype(int)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot([0, 100], [0, 1], color=plotstyle.GRID, lw=1.5, ls="--")
    ax.text(62, 0.55, "random flagging", rotation=29, fontsize=8, color=plotstyle.TEXT_2)
    base = 100 * y.mean()
    ax.plot([0, base, 100], [0, 1, 1], color=plotstyle.TEXT_2, lw=1, ls=":")
    ax.text(base - 1, 1.025, f"perfect model ({base:.0f} alerts)", ha="right", fontsize=8, color=plotstyle.TEXT_2)
    for key, color in [("svm", plotstyle.SERIES[2]), ("svm_mc", plotstyle.SERIES[3]), (FINAL, plotstyle.SERIES[1])]:
        p = proba[key]["test"]["phone"][:, ESC_IDX].sum(1)
        ts = np.unique(np.r_[p, 1.01])
        alerts = [100 * (p >= t).mean() for t in ts]
        recall = [((p >= t) & (y == 1)).sum() / y.sum() for t in ts]
        label = "CNN (final, seed 42)" if key == FINAL else MODELS[key][0]
        ax.plot(alerts, recall, color=color, lw=2, label=label)
        b = results[key]["phone"]["business"]
        ax.plot(b["alerts_per_100"], b["recall"], "o", ms=8, color=color, mec=plotstyle.SURFACE, mew=2)
    b = results[FINAL]["phone"]["business"]
    ax.annotate(f"Operating point (threshold {b['threshold']:.2f} from validation)\n"
                f"catches {b['recall']:.0%} · {b['alerts_per_100']:.0f} alerts / 100 calls",
                (b["alerts_per_100"], b["recall"]), xytext=(18, 0.25), fontsize=8, color=plotstyle.TEXT,
                arrowprops={"arrowstyle": "->", "color": plotstyle.TEXT_2})
    ax.plot(100, 1, "s", ms=7, color=plotstyle.TEXT_2)
    ax.text(98, 1.025, "flag every call", ha="right", fontsize=8, color=plotstyle.TEXT_2)
    ax.set_xlim(0, 102)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Alerts per 100 calls")
    ax.set_ylabel("Share of frustrated callers caught (recall)")
    ax.set_title("Escalation trade-off on telephone test audio")
    ax.legend(loc="lower right")
    fig.savefig(FIGURES / "12_escalation_tradeoff_test_phone.png")
    plt.close(fig)


def main():
    plotstyle.apply()
    proba, labels = all_probas()
    results = evaluate_all(proba, labels)
    with open(RESULTS / "metrics_step8_test.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    table = summary_table(results)
    table.to_csv(RESULTS / "test_results.csv", index=False)
    plot_confusions(results[FINAL], FINAL, split="test")
    fig_model_comparison(table)
    fig_tradeoff(proba, labels, results)
    np.savez_compressed(RESULTS / "test_probabilities.npz",
                        **{f"{k}_{v}": proba[k]["test"][v] for k in MODELS for v in ("clean", "phone")},
                        labels=labels["test"])
    pd.set_option("display.width", 200)
    show = table[["model", "audio", "accuracy", "macro_f1", "macro_f1_std", "esc_auc", "esc_recall",
                  "esc_precision", "alerts_per_100", "false_alerts_per_100", "missed_per_100"]]
    print(show.round(3).to_string(index=False))
    print("flag every call:", {k: round(v, 3) for k, v in results["flag_every_call"].items()})
    for key in MODELS:
        print(f"{key}: threshold {results[key]['threshold_from_val_phone']:.3f}")


if __name__ == "__main__":
    main()
