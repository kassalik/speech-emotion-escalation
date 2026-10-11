"""Export the release CNN to ONNX for lightweight / edge inference.

Writes to models/:
  ser_cnn.onnx    log-mel dB [N, 64, 301] -> class probabilities [N, 6] (per-band standardisation built in)
  frontend.npz    mel filterbank and STFT window taken from torchaudio, so the NumPy front-end matches training
  config.json     labels, escalation threshold and pre-processing constants

Usage: python src/export_onnx.py [results/models/cnn_s42.pt]
"""
import json
import sys

import numpy as np
import torch
import torchaudio

from config import CLASSES, ESCALATE_CLASSES, RESULTS, ROOT, SAMPLE_RATE
from features import HOP, N_FFT, N_MELS
from predict import HOP_SECONDS, VAD_ABS_DBFS, VAD_REL_DB
from preprocess import (MU, N_SAMPLES, PEAK_LIMIT, PHONE_BAND, PHONE_SR, TARGET_RMS_DBFS, TRIM_MARGIN_DB,
                        TRIM_TOP_DB_RANGE)
from train_cnn import ARCHS

MODELS = ROOT / "models"
N_FRAMES = N_SAMPLES // HOP + 1


class Deployable(torch.nn.Module):
    """Standardise log-mel with training statistics, run the CNN, return probabilities."""

    def __init__(self, net, mean, std):
        super().__init__()
        self.net = net
        self.register_buffer("mean", mean.view(1, -1, 1))
        self.register_buffer("std", std.view(1, -1, 1))

    def forward(self, logmel_db):
        x = ((logmel_db - self.mean) / self.std).unsqueeze(1)
        return torch.softmax(self.net(x), dim=1)


def strip_debug_metadata(path):
    """Remove exporter debug info (stack traces with local file paths) from the ONNX graph."""
    import onnx
    m = onnx.load(str(path))
    m.doc_string = ""
    graphs = [m.graph] + [f for f in m.functions]
    for g in graphs:
        for node in g.node:
            node.doc_string = ""
            del node.metadata_props[:]
    onnx.save(m, str(path))


def main(ckpt_path=RESULTS / "models" / "cnn_s42.pt"):
    MODELS.mkdir(exist_ok=True)
    ck = torch.load(ckpt_path, map_location="cpu")
    net = ARCHS[ck["arch"]]()
    net.load_state_dict(ck["state_dict"])
    model = Deployable(net, ck["mean"].flatten(), ck["std"].flatten()).eval()

    example = torch.randn(2, N_MELS, N_FRAMES) * 10 - 30
    onnx_path = MODELS / "ser_cnn.onnx"
    torch.onnx.export(model, (example,), onnx_path, input_names=["logmel_db"], output_names=["probabilities"],
                      dynamic_shapes={"logmel_db": {0: torch.export.Dim("batch", min=1, max=256)}},
                      opset_version=18, dynamo=True, external_data=False)
    strip_debug_metadata(onnx_path)

    fb = torchaudio.functional.melscale_fbanks(n_freqs=N_FFT // 2 + 1, f_min=20.0, f_max=SAMPLE_RATE / 2,
                                               n_mels=N_MELS, sample_rate=SAMPLE_RATE, norm=None, mel_scale="htk")
    np.savez(MODELS / "frontend.npz", mel_fb=fb.numpy().astype(np.float32),
             window=torch.hann_window(N_FFT).numpy().astype(np.float32))

    stem = ckpt_path.stem
    threshold = json.loads((RESULTS / "metrics_step8_test.json").read_text())[stem]["threshold_from_val_phone"]
    config = {
        "model": "ser_cnn.onnx", "source_checkpoint": stem, "classes": CLASSES,
        "escalate_classes": [c for c in CLASSES if c in ESCALATE_CLASSES], "threshold": round(threshold, 4),
        "sample_rate": SAMPLE_RATE, "clip_samples": N_SAMPLES, "n_fft": N_FFT, "hop": HOP, "n_mels": N_MELS,
        "top_db": 80.0, "trim_margin_db": TRIM_MARGIN_DB, "trim_top_db_range": list(TRIM_TOP_DB_RANGE),
        "target_rms_dbfs": TARGET_RMS_DBFS, "peak_limit": PEAK_LIMIT, "phone_sr": PHONE_SR,
        "phone_band": list(PHONE_BAND), "mu": MU, "window_hop_seconds": HOP_SECONDS,
        "vad_rel_db": VAD_REL_DB, "vad_abs_dbfs": VAD_ABS_DBFS, "licence": "CC BY-NC-SA 4.0",
    }
    (MODELS / "config.json").write_text(json.dumps(config, indent=2))

    # parity check: ONNX vs PyTorch on random log-mels
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = (torch.randn(8, N_MELS, N_FRAMES) * 12 - 35)
    with torch.no_grad():
        ref = model(x).numpy()
    out = sess.run(None, {"logmel_db": x.numpy()})[0]
    print(f"Exported {onnx_path.name} ({onnx_path.stat().st_size / 1e6:.2f} MB) | "
          f"max |ONNX - PyTorch| = {np.abs(out - ref).max():.2e} | threshold {threshold:.3f}")


if __name__ == "__main__":
    main(ROOT / sys.argv[1] if len(sys.argv) > 1 else RESULTS / "models" / "cnn_s42.pt")
