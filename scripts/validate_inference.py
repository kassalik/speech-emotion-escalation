"""Check that the lightweight inference/ package reproduces the research pipeline.

1. Front-end: NumPy log-mel vs stored torchaudio log-mels (same pre-processed audio).
2. End to end: raw WAV -> inference/ (NumPy/SciPy pre-processing, NumPy log-mel, ONNX) vs the Step 8 test-set
   probabilities of the PyTorch model, on every test clip, clean and telephone.
Writes results/inference_parity.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from config import METADATA_CSV, PROCESSED, RESULTS  # noqa: E402
from inference import EscalationPredictor  # noqa: E402
from inference.audio import load_audio, logmel_db  # noqa: E402


def main():
    p = EscalationPredictor(threads=4)
    meta = pd.read_csv(METADATA_CSV)
    test_rows = np.flatnonzero(meta["split"] == "test")
    y = meta["label"].values[test_rows]
    report = {}

    # 1. front-end parity on identical pre-processed audio
    audio = np.load(PROCESSED / "audio_clean.npy", mmap_mode="r")
    ref_mel = np.load(PROCESSED / "logmel_clean.npy", mmap_mode="r")
    sample = test_rows[:200]
    diffs = [np.abs(logmel_db(audio[i].astype(np.float32), p.mel_fb, p.window) - ref_mel[i].astype(np.float32))
             for i in sample]
    report["frontend_logmel_abs_diff_db"] = {"mean": float(np.mean([d.mean() for d in diffs])),
                                             "max": float(np.max([d.max() for d in diffs]))}

    # 2. end-to-end parity from raw files
    ref = np.load(RESULTS / "test_probabilities.npz")
    esc = p.esc_idx
    for version in ("clean", "phone"):
        light = np.concatenate([
            p.score([p.prepare_clip(load_audio(ROOT / meta["path"][i], target_sr=p.sr), version == "phone")
                     for i in test_rows[lo:lo + 128]])
            for lo in range(0, len(test_rows), 128)])
        print(f"{version}: scored {len(light)} clips", flush=True)
        torch_p = ref[f"cnn_s42_{version}"]
        pe_l, pe_t = light[:, esc].sum(1), torch_p[:, esc].sum(1)
        report[version] = {
            "top1_agreement": float((light.argmax(1) == torch_p.argmax(1)).mean()),
            "escalate_decision_agreement": float(((pe_l >= p.threshold) == (pe_t >= p.threshold)).mean()),
            "p_escalate_abs_diff_mean": float(np.abs(pe_l - pe_t).mean()),
            "p_escalate_abs_diff_max": float(np.abs(pe_l - pe_t).max()),
            "accuracy_light": float(accuracy_score(y, light.argmax(1))),
            "accuracy_pytorch": float(accuracy_score(y, torch_p.argmax(1))),
            "macro_f1_light": float(f1_score(y, light.argmax(1), average="macro")),
            "macro_f1_pytorch": float(f1_score(y, torch_p.argmax(1), average="macro")),
        }
    (RESULTS / "inference_parity.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
