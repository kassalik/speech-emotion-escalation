"""Step 9: emotion / escalation prediction for any .wav file.

  python predict.py FILE.wav [--phone] [--plot] [--model results/models/cnn_s42.pt]

Short clips (<= 3 s): same pipeline as training (trim, normalise, pad to 3 s).
Longer audio (a call): 3 s windows every 1.5 s. Windows without speech (energy more than 35 dB
below the loudest window, or below -55 dBFS) are skipped. Each speech window is loudness-normalised
and scored. The call is flagged ESCALATE when two consecutive speech windows (about 4.5 s of
speech) average at or above the threshold chosen on validation data in Step 8.
"""
import argparse
import json
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch

import plotstyle
from baseline import ESC_IDX
from config import CLASSES, RESULTS, SAMPLE_RATE
from features import make_logmel
from preprocess import N_SAMPLES, PEAK_LIMIT, normalise_loudness, process_clip, telephone, trim_silence
from train_cnn import ARCHS

HOP_SECONDS = 1.5
VAD_REL_DB, VAD_ABS_DBFS = 35.0, -55.0
DEFAULT_MODEL = RESULTS / "models" / "cnn_s42.pt"


def load_model(path):
    ck = torch.load(path, map_location="cpu")
    model = ARCHS[ck["arch"]]()
    model.load_state_dict(ck["state_dict"])
    model.eval()
    threshold = json.load(open(RESULTS / "metrics_step8_test.json"))[Path(path).stem]["threshold_from_val_phone"]
    return model, ck["mean"], ck["std"], threshold


def score(model, mean, std, clips):
    """clips: [N, 48000] float32 -> class probabilities [N, 6]."""
    with torch.no_grad():
        mel = make_logmel()(torch.from_numpy(np.asarray(clips, dtype=np.float32)))
        return torch.softmax(model(((mel - mean) / std).unsqueeze(1)), dim=1).numpy()


def prepare_window(seg, phone):
    """A full 3 s window of a call, processed like Step 4 (no trimming or padding needed)."""
    clean = normalise_loudness(seg)
    if not phone:
        return clean
    ph = telephone(clean)[:N_SAMPLES]
    ph = np.pad(ph, (0, N_SAMPLES - len(ph)))
    ph *= np.sqrt(np.mean(clean ** 2)) / (np.sqrt(np.mean(ph ** 2)) + 1e-9)
    return np.clip(ph, -PEAK_LIMIT, PEAK_LIMIT)


def windows(y):
    hop = int(HOP_SECONDS * SAMPLE_RATE)
    starts = list(range(0, max(len(y) - N_SAMPLES, 0) + 1, hop))
    if starts[-1] + N_SAMPLES < len(y):
        starts.append(len(y) - N_SAMPLES)
    segs = [y[s:s + N_SAMPLES] for s in starts]
    rms_db = np.array([20 * np.log10(np.sqrt(np.mean(s ** 2)) + 1e-9) for s in segs])
    speech = (rms_db >= rms_db.max() - VAD_REL_DB) & (rms_db >= VAD_ABS_DBFS)
    return np.array(starts) / SAMPLE_RATE, segs, speech


def call_verdict(p_esc, speech, threshold):
    idx = np.where(speech)[0]
    if len(idx) == 0:
        return False, float("nan"), None
    s = p_esc[idx]
    pair = (s[:-1] + s[1:]) / 2 if len(s) > 1 else s
    k = int(np.argmax(pair))
    return bool(pair[k] >= threshold), float(pair[k]), int(idx[k])


def plot_timeline(path, y, starts, p_esc, speech, emotions, threshold, out, truth=None):
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(10, 4.8), sharex=True, gridspec_kw={"height_ratios": [1, 2]})
    t = np.arange(len(y)) / SAMPLE_RATE
    a0.plot(t, y, color=plotstyle.SERIES[0], lw=0.4)
    a0.set_yticks([])
    a0.grid(False)
    a0.set_title(f"{Path(path).name}: escalation timeline", loc="left")
    if truth is not None:
        for _, r in truth.iterrows():
            a0.axvspan(r["start"], r["end"], color=plotstyle.ESCALATE if r["escalate"] else plotstyle.NO_ESCALATE,
                       alpha=0.12, lw=0)
            a0.text((r["start"] + r["end"]) / 2, a0.get_ylim()[1] * 0.75, r["emotion"], ha="center", fontsize=8,
                    color=plotstyle.TEXT_2)
    centres = starts + N_SAMPLES / SAMPLE_RATE / 2
    a1.axhline(threshold, color=plotstyle.TEXT_2, ls="--", lw=1)
    a1.text(centres[0], threshold + 0.03, f"threshold {threshold:.2f}", fontsize=8, color=plotstyle.TEXT_2)
    a1.plot(centres[speech], p_esc[speech], color=plotstyle.ESCALATE, lw=2, marker="o", ms=5,
            mec=plotstyle.SURFACE, mew=1.5)
    if (~speech).any():
        a1.plot(centres[~speech], np.zeros((~speech).sum()), "x", color=plotstyle.TEXT_2, ms=6, label="no speech (skipped)")
        a1.legend(loc="upper left")
    for c, p, e, s in zip(centres, p_esc, emotions, speech):
        if s:
            a1.text(c, min(p + 0.06, 1.02), e[:3], ha="center", fontsize=7, color=plotstyle.TEXT_2)
    a1.set_ylim(0, 1.1)
    a1.set_ylabel("P(escalate)")
    a1.set_xlabel("Time (s) · each point = a 3 s window centred there")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def predict_file(path, phone=False, plot=False, model_path=DEFAULT_MODEL, truth=None, quiet=False):
    model, mean, std, threshold = load_model(model_path)
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    dur = len(y) / SAMPLE_RATE
    out = {"file": str(path), "seconds": round(dur, 2), "phone": phone, "threshold": threshold}
    say = (lambda *a: None) if quiet else print
    say(f"File: {Path(path).name}  ({dur:.1f} s, telephone channel: {'on' if phone else 'off'})")
    trimmed = trim_silence(y)
    if len(trimmed) <= N_SAMPLES:
        clean, ph, *_ = process_clip(Path(path).resolve())  # identical to the training pipeline
        p = score(model, mean, std, [(ph if phone else clean).astype(np.float32)])[0]
        p_esc = float(p[ESC_IDX].sum())
        out.update({"mode": "clip", "emotion": CLASSES[int(p.argmax())], "probabilities": dict(zip(CLASSES, p.round(3).tolist())),
                    "p_escalate": round(p_esc, 3), "escalate": p_esc >= threshold})
        say("Emotion probabilities: " + "  ".join(f"{c} {v:.2f}" for c, v in sorted(zip(CLASSES, p), key=lambda x: -x[1])))
        say(f"Predicted emotion: {out['emotion'].upper()} | P(escalate) {p_esc:.2f} | "
            f"Decision: {'ESCALATE' if out['escalate'] else 'OK'} (threshold {threshold:.2f})")
        return out
    starts, segs, speech = windows(y)
    probs = score(model, mean, std, [prepare_window(s, phone) for s in segs])
    p_esc = probs[:, ESC_IDX].sum(1)
    emotions = [CLASSES[i] for i in probs.argmax(1)]
    flag, peak, k = call_verdict(p_esc, speech, threshold)
    say(f"{'Window':>8}  {'Emotion':<8} P(escalate)")
    for st, e, p, s in zip(starts, emotions, p_esc, speech):
        mark = "  no speech" if not s else ("  <" if p >= threshold else "")
        say(f"{int(st // 60)}:{st % 60:04.1f}  {(e if s else '-'):<8} {p:.2f}{mark}")
    when = f" at {int(starts[k] // 60)}:{starts[k] % 60:04.1f}-{int((starts[k] + 4.5) // 60)}:{(starts[k] + 4.5) % 60:04.1f}" if k is not None else ""
    say(f"Call verdict: {'ESCALATE' if flag else 'OK'} (peak 2-window score {peak:.2f}{when}, threshold {threshold:.2f})")
    out.update({"mode": "call", "escalate": flag, "peak_score": round(peak, 3),
                "windows": [{"start": float(s), "speech": bool(v), "emotion": e, "p_escalate": round(float(p), 3)}
                            for s, v, e, p in zip(starts, speech, emotions, p_esc)]})
    if plot:
        plotstyle.apply()
        png = Path(path).with_name(Path(path).stem + ("_phone" if phone else "") + "_timeline.png")
        plot_timeline(path, y, starts, p_esc, speech, emotions, threshold, png, truth)
        out["timeline"] = str(png)
        say(f"Timeline saved: {png.relative_to(RESULTS.parent) if png.is_relative_to(RESULTS.parent) else png}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Speech emotion / escalation prediction")
    ap.add_argument("file")
    ap.add_argument("--phone", action="store_true", help="simulate a telephone channel first")
    ap.add_argument("--plot", action="store_true", help="save a timeline PNG for long audio")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    a = ap.parse_args()
    predict_file(a.file, a.phone, a.plot, a.model)
