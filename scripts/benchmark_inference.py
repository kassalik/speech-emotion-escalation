"""CPU latency benchmark of the lightweight inference/ package (per 3 s window).

Usage: python scripts/benchmark_inference.py [--threads 1 4] [--runs 200]
Writes results/benchmark_<label or CPU architecture>.json.
Run it on the target device (e.g. Raspberry Pi 5) for real numbers.
"""
import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference import EscalationPredictor  # noqa: E402
from inference.audio import load_audio, logmel_db  # noqa: E402


def timed(fn, runs):
    for _ in range(10):  # warm-up
        fn()
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--label", default=None, help="device label for the output file, e.g. rpi5")
    a = ap.parse_args()
    clip = load_audio(ROOT / "results" / "demo" / "single_angry_CRE_1001.wav")
    report = {"device": a.label or platform.machine(), "machine": platform.machine(), "processor": platform.processor(),
              "python": platform.python_version(), "window_seconds": 3.0, "results": {}}
    for th in a.threads:
        t0 = time.perf_counter()
        p = EscalationPredictor(threads=th)
        load_ms = (time.perf_counter() - t0) * 1000
        x = p.prepare_clip(clip)
        mel = logmel_db(x, p.mel_fb, p.window)[None]
        r = {"model_load_ms": load_ms,
             "preprocess": timed(lambda: p.prepare_clip(clip), a.runs),
             "frontend_logmel": timed(lambda: logmel_db(x, p.mel_fb, p.window), a.runs),
             "onnx_model": timed(lambda: p.session.run(None, {"logmel_db": mel}), a.runs),
             "end_to_end_window": timed(lambda: p.score([p.prepare_clip(clip)]), a.runs)}
        r["real_time_factor"] = r["end_to_end_window"]["median_ms"] / 3000.0
        report["results"][f"{th}_threads"] = r
    report["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    out = ROOT / "results" / f"benchmark_{report['device']}.json"
    out.write_text(json.dumps(report, indent=2))
    for k, r in report["results"].items():
        print(f"{k}: preprocess {r['preprocess']['median_ms']:.1f} ms | log-mel {r['frontend_logmel']['median_ms']:.1f} ms"
              f" | ONNX {r['onnx_model']['median_ms']:.1f} ms | end-to-end {r['end_to_end_window']['median_ms']:.1f} ms"
              f" (p95 {r['end_to_end_window']['p95_ms']:.1f}) | real-time factor {r['real_time_factor']:.3f}"
              f" | load {r['model_load_ms']:.0f} ms")
    print(f"peak memory {report['peak_rss_mb']:.0f} MB | {report['machine']} | saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
