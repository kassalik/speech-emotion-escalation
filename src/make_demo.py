"""Step 9: demo material from held-out test speakers, plus the predictions on it.

Simulated support call: raw clips of ONE unseen test speaker (the first CREMA-D test speaker in
sorted order, not hand-picked), joined in a calm -> frustrated order:
  neutral x3 -> disgust -> angry x3
"""
import contextlib
import io
import json

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from config import ESCALATE_CLASSES, METADATA_CSV, RESULTS, ROOT, SAMPLE_RATE
from predict import predict_file

DEMO = RESULTS / "demo"
SCRIPT = [("neutral", 3), ("disgust", 1), ("angry", 3)]


def build_call(meta):
    test = meta[(meta["split"] == "test") & (meta["dataset"] == "CREMA-D")]
    speaker = sorted(test["speaker"].unique())[0]
    clips = test[test["speaker"] == speaker]
    audio, rows, t = [], [], 0.0
    for emotion, n in SCRIPT:
        pool = clips[clips["emotion"] == emotion].sort_values(["intensity", "sentence"])
        if emotion == "angry":  # include the high-intensity take when available
            pool = pool.sort_values("intensity", key=lambda s: s.ne("high"))
        for path in pool["path"].iloc[:n]:
            y, _ = librosa.load(ROOT / path, sr=SAMPLE_RATE)
            audio.append(y)
            rows.append({"start": t, "end": t + len(y) / SAMPLE_RATE, "emotion": emotion,
                         "escalate": emotion in ESCALATE_CLASSES, "source": path})
            t += len(y) / SAMPLE_RATE
    call = np.concatenate(audio)
    sf.write(DEMO / "simulated_call.wav", call, SAMPLE_RATE)
    truth = pd.DataFrame(rows)
    truth.to_csv(DEMO / "simulated_call_truth.csv", index=False)
    return speaker, truth


def main():
    DEMO.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(METADATA_CSV)
    speaker, truth = build_call(meta)
    log, results = io.StringIO(), {}
    with contextlib.redirect_stdout(log):
        print(f"=== Simulated call: test speaker {speaker} (never seen in training), "
              f"{len(truth)} clips, {truth['end'].iloc[-1]:.1f} s ===")
        print("Ground truth: " + " -> ".join(truth["emotion"]))
        for phone in (False, True):
            print()
            results[f"call_{'phone' if phone else 'clean'}"] = predict_file(
                DEMO / "simulated_call.wav", phone=phone, plot=True, truth=truth)
        for emotion in ("neutral", "angry"):
            src = meta[(meta["split"] == "test") & (meta["speaker"] == speaker) & (meta["emotion"] == emotion)]
            src = src[~src["path"].isin(truth["source"])]["path"].iloc[0]
            dst = DEMO / f"single_{emotion}_{speaker}.wav"
            sf.write(dst, librosa.load(ROOT / src, sr=SAMPLE_RATE)[0], SAMPLE_RATE)
            for phone in (False, True):
                print()
                print(f"--- Single clip, true emotion: {emotion} ---")
                results[f"single_{emotion}_{'phone' if phone else 'clean'}"] = predict_file(dst, phone=phone)
    (DEMO / "demo_output.txt").write_text(log.getvalue())
    with open(DEMO / "demo_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(log.getvalue())


if __name__ == "__main__":
    main()
