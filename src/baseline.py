"""Step 6: baseline models (SVM, Random Forest) on handcrafted features.

Trained on clean training audio. Hyper-parameters chosen by speaker-grouped 5-fold CV on the
training split (macro-F1), then reported on the validation split for clean and telephone audio.
The test split is not touched here (Step 8).
"""
import json
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import plotstyle
from config import CLASSES, ESCALATE_CLASSES, FIGURES, METADATA_CSV, PROCESSED, RESULTS, SEED

MODELS_DIR = RESULTS / "models"
ESC_IDX = [CLASSES.index(c) for c in CLASSES if c in ESCALATE_CLASSES]

SEARCH = {
    "svm": (
        Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                  ("clf", SVC(kernel="rbf", class_weight="balanced", random_state=SEED))]),
        {"clf__C": [1, 3, 10, 30], "clf__gamma": [0.001, 0.002, 0.004, 0.008]},
    ),
    "rf": (
        Pipeline([("impute", SimpleImputer(strategy="median")),
                  ("clf", RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                                                 random_state=SEED, n_jobs=1))]),
        {"clf__max_features": ["sqrt", 0.15], "clf__min_samples_leaf": [1, 3, 8]},
    ),
}


def load_split(split, version):
    meta = pd.read_csv(METADATA_CSV)
    X = pd.read_csv(PROCESSED / f"handcrafted_{version}.csv").drop(columns="path")
    m = (meta["split"] == split).values
    return X[m], meta.loc[m, "label"].values, meta.loc[m, "speaker"].values


def escalate_scores(proba):
    return proba[:, ESC_IDX].sum(axis=1)


def evaluate(model, X, y, threshold=0.5):
    return metrics_from_proba(model.predict_proba(X), y, threshold)


def metrics_from_proba(proba, y, threshold=0.5):
    pred = proba.argmax(axis=1)
    y_esc = np.isin(y, ESC_IDX).astype(int)
    p_esc = escalate_scores(proba)
    esc_pred = (p_esc >= threshold).astype(int)
    report = classification_report(y, pred, target_names=CLASSES, output_dict=True, zero_division=0)
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "per_class_f1": {c: report[c]["f1-score"] for c in CLASSES},
        "per_class_recall": {c: report[c]["recall"] for c in CLASSES},
        "escalate": {
            "threshold": threshold,
            "recall": recall_score(y_esc, esc_pred),
            "precision": precision_score(y_esc, esc_pred),
            "f1": f1_score(y_esc, esc_pred),
            "roc_auc": roc_auc_score(y_esc, p_esc),
            "alerts_per_100": 100 * esc_pred.mean(),
        },
        "confusion": confusion_matrix(y, pred).tolist(),
    }


def plot_confusions(results, name, split="val"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, version in zip(axes, ["clean", "phone"]):
        cm = np.array(results[version]["confusion"], dtype=float)
        cm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > 0.55 else plotstyle.TEXT)
        labels = [c.capitalize() for c in CLASSES]
        ax.set_xticks(range(len(CLASSES)), labels, rotation=30)
        ax.set_yticks(range(len(CLASSES)), labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        r = results[version]
        ax.set_title(f"{version.capitalize()} audio · acc {r['accuracy']:.1%} · macro-F1 {r['macro_f1']:.3f}",
                     fontsize=10)
        ax.grid(False)
    split_label = {"val": "validation", "test": "held-out test"}[split]
    fig.suptitle(f"{name.upper()} confusion matrices ({split_label} set, row-normalised recall)",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{'08' if split == 'val' else '11'}_confusion_{name}_{split}.png")
    plt.close(fig)


def main():
    plotstyle.apply()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X_tr, y_tr, g_tr = load_split("train", "clean")
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    summary = {}
    for name, (pipe, grid) in SEARCH.items():
        t0 = time.time()
        search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=False)
        search.fit(X_tr, y_tr, groups=g_tr)
        cv_table = pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score"]]
        cv_table.sort_values("mean_test_score", ascending=False).to_csv(
            RESULTS / f"cv_{name}.csv", index=False)
        best = search.best_params_
        model = pipe.set_params(**best)
        if name == "svm":
            model.set_params(clf__probability=True)  # Platt scaling for escalation scores
        else:
            model.set_params(clf__n_jobs=-1)
        model.fit(X_tr, y_tr)
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        res = {"best_params": best, "cv_macro_f1": search.best_score_,
               "cv_macro_f1_std": float(cv_table["std_test_score"][search.best_index_]),
               "fit_seconds": round(time.time() - t0, 1)}
        for version in ["clean", "phone"]:
            X_v, y_v, _ = load_split("val", version)
            res[version] = evaluate(model, X_v, y_v)
        plot_confusions(res, name)
        summary[name] = res
        print(f"{name}: best {best} | CV macro-F1 {search.best_score_:.3f} | "
              f"val clean acc {res['clean']['accuracy']:.3f} F1 {res['clean']['macro_f1']:.3f} | "
              f"val phone acc {res['phone']['accuracy']:.3f} F1 {res['phone']['macro_f1']:.3f} | "
              f"{res['fit_seconds']}s", flush=True)

    _, y_v, _ = load_split("val", "clean")
    summary["always_escalate_reference"] = {
        "escalate_recall": 1.0, "escalate_precision": float(np.isin(y_v, ESC_IDX).mean()),
        "alerts_per_100": 100.0, "accuracy_6class_majority": float(np.bincount(y_v).max() / len(y_v))}
    with open(RESULTS / "metrics_step6_baseline.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)


def train_multicondition():
    """SVM trained on clean + telephone training features, with the Step 6 hyper-parameters."""
    best = json.load(open(RESULTS / "metrics_step6_baseline.json"))["svm"]["best_params"]
    X_c, y, _ = load_split("train", "clean")
    X_p, _, _ = load_split("train", "phone")
    X_tr, y_tr = pd.concat([X_c, X_p], ignore_index=True), np.concatenate([y, y])
    model = SEARCH["svm"][0].set_params(**best, clf__probability=True)
    t0 = time.time()
    model.fit(X_tr, y_tr)
    joblib.dump(model, MODELS_DIR / "svm_mc.joblib")
    res = {"best_params": best, "train_rows": len(X_tr), "fit_seconds": round(time.time() - t0, 1)}
    for version in ["clean", "phone"]:
        X_v, y_v, _ = load_split("val", version)
        res[version] = evaluate(model, X_v, y_v)
    with open(RESULTS / "metrics_step8_svm_mc_val.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"svm_mc: val clean acc {res['clean']['accuracy']:.3f} F1 {res['clean']['macro_f1']:.3f} | "
          f"phone acc {res['phone']['accuracy']:.3f} F1 {res['phone']['macro_f1']:.3f} | {res['fit_seconds']}s")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "multicondition":
        train_multicondition()
    else:
        main()
