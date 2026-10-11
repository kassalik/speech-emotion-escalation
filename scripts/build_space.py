"""Assemble a ready-to-upload Hugging Face Space (Gradio) in dist/hf-space/.

Usage: python scripts/build_space.py
Then create a Gradio Space on huggingface.co and upload the folder contents (or push them with git).
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "hf-space"
EXAMPLES = ["single_angry_CRE_1001.wav", "single_neutral_CRE_1001.wav", "simulated_call.wav"]

SPACE_README = """---
title: Call Escalation Detector
emoji: 📞
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: {gradio_version}
python_version: "3.11"
app_file: app.py
pinned: false
license: cc-by-nc-sa-4.0
short_description: Speech emotion recognition for call escalation (ONNX)
---

# Call Escalation Detector

Record or upload speech: a CNN on log-mel spectrograms recognises six emotions and flags negative ones (angry,
disgust, fear, sad) for early escalation. Longer audio is scored in 3 s windows with a call-level verdict and timeline.

- Model: 1.2 M-parameter CNN exported to ONNX, running on CPU with ONNX Runtime and a NumPy front-end.
- Training data: CREMA-D (ODbL) and RAVDESS (CC BY-NC-SA 4.0), with telephone-channel augmentation.
- Held-out test (17 unseen speakers, telephone audio): macro-F1 0.64, 90% escalate recall at 85% precision.
- Trained on acted speech: real-world accuracy will be lower. Decision support only, not for automated decisions
  about people. Audio is processed in memory and not stored.

Model weights: CC BY-NC-SA 4.0 (derived from RAVDESS). Example clips: CREMA-D speaker 1001 (attribution, ODbL/DbCL).

Citations: Cao et al. (2014), CREMA-D, IEEE Trans. Affective Computing 5(4). Livingstone & Russo (2018), RAVDESS,
PLoS ONE 13(5).
"""


def main():
    import gradio
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "examples").mkdir(parents=True)
    shutil.copy(ROOT / "app" / "app.py", OUT / "app.py")
    shutil.copytree(ROOT / "inference", OUT / "inference", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "models", OUT / "models")
    for f in EXAMPLES:
        shutil.copy(ROOT / "results" / "demo" / f, OUT / "examples" / f)
    reqs = (ROOT / "requirements-inference.txt").read_text().splitlines()
    reqs = [r for r in reqs if r and not r.startswith("#")] + ["matplotlib>=3.8"]
    (OUT / "requirements.txt").write_text("\n".join(reqs) + "\n")
    (OUT / "README.md").write_text(SPACE_README.format(gradio_version=gradio.__version__))
    files = sorted(p.relative_to(OUT).as_posix() for p in OUT.rglob("*") if p.is_file())
    size = sum((OUT / f).stat().st_size for f in files) / 1e6
    print(f"Built {OUT.relative_to(ROOT)} ({len(files)} files, {size:.1f} MB):")
    print("\n".join(f"  {f}" for f in files))


if __name__ == "__main__":
    main()
