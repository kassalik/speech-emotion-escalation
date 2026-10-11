"""Gradio demo: speech emotion recognition and escalation decision (ONNX Runtime, CPU).

Run locally:  python app/app.py        (then open http://127.0.0.1:7860)
"""
import sys
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "inference").exists() else HERE.parent      # Space layout vs repository layout
sys.path.insert(0, str(ROOT))
from inference import EscalationPredictor  # noqa: E402

MODEL_DIR = ROOT / "models"
EXAMPLE_DIR = ROOT / "examples" if (ROOT / "examples").exists() else ROOT / "results" / "demo"
predictor = EscalationPredictor(MODEL_DIR)

ORANGE, BLUE, GREY, INK = "#eb6834", "#2a78d6", "#52514e", "#0b0b0b"


def timeline_figure(result):
    w = result["windows"]
    centres = [x["start"] + 1.5 for x in w]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axhline(predictor.threshold, color=GREY, ls="--", lw=1)
    ax.text(centres[0], predictor.threshold + 0.03, f"threshold {predictor.threshold:.2f}", fontsize=8, color=GREY)
    sp = [i for i, x in enumerate(w) if x["speech"]]
    ax.plot([centres[i] for i in sp], [w[i]["p_escalate"] for i in sp], color=ORANGE, lw=2, marker="o", ms=5)
    for i in sp:
        ax.text(centres[i], min(w[i]["p_escalate"] + 0.06, 1.02), w[i]["emotion"][:3], ha="center", fontsize=7,
                color=GREY)
    silent = [i for i, x in enumerate(w) if not x["speech"]]
    if silent:
        ax.plot([centres[i] for i in silent], [0] * len(silent), "x", color=GREY, label="no speech")
        ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Time (s) · each point is a 3 s window")
    ax.set_ylabel("P(escalate)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e4e3df")
    fig.tight_layout()
    return fig


def analyse(audio, phone):
    if audio is None:
        return None, "Record or upload some speech first.", None
    sr, data = audio
    r = predictor.predict(data, sr=sr, phone=phone)
    if r["mode"] == "empty":
        return None, "No speech detected. Try speaking closer to the microphone.", None
    if r["mode"] == "clip":
        verdict = "⚠️ **ESCALATE**" if r["escalate"] else "✅ **OK**"
        md = (f"### {verdict}\n**Predicted emotion:** {r['emotion']}  \n"
              f"**P(escalate):** {r['p_escalate']:.2f} (threshold {r['threshold']:.2f})  \n"
              f"Clip length: {r['seconds']:.1f} s")
        return r["probabilities"], md, None
    speech = [x for x in r["windows"] if x["speech"]]
    if not speech:
        return None, "No speech detected in the recording.", None
    peak = max(speech, key=lambda x: x["p_escalate"])
    verdict = "⚠️ **ESCALATE**" if r["escalate"] else "✅ **OK**"
    md = (f"### Call verdict: {verdict}\n"
          f"**Most negative moment:** {peak['start']:.1f}–{peak['start'] + 3:.1f} s ({peak['emotion']}, "
          f"P(escalate) {peak['p_escalate']:.2f})  \n"
          f"**Peak 2-window score:** {r['peak_score']:.2f} (threshold {r['threshold']:.2f})  \n"
          f"Length: {r['seconds']:.1f} s · {len(speech)} speech windows")
    return peak["probabilities"], md, timeline_figure(r)


examples = [[str(EXAMPLE_DIR / f), phone] for f, phone in [
    ("single_angry_CRE_1001.wav", False), ("single_neutral_CRE_1001.wav", False),
    ("simulated_call.wav", True)] if (EXAMPLE_DIR / f).exists()]

with gr.Blocks(title="Call Escalation Detector") as demo:
    gr.Markdown(
        "# Call Escalation Detector\n"
        "Speech emotion recognition for customer-support calls. The model recognises six emotions and flags "
        "negative ones (angry, disgust, fear, sad) for **early escalation and QA review**. Clips up to 3 s get one "
        "decision; longer audio is scored in 3 s windows and gets a call-level verdict with a timeline.\n\n"
        "*CNN on log-mel spectrograms (1.2 M parameters, ONNX Runtime on CPU), trained on acted speech (CREMA-D, "
        "RAVDESS) with telephone-channel augmentation. Decision support only: not for automated decisions about "
        "people. Audio is processed in memory and not stored.*")
    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Speech (record or upload)")
            phone_in = gr.Checkbox(value=False, label="Simulate a telephone channel (8 kHz, 300–3400 Hz, μ-law)")
            run = gr.Button("Analyse", variant="primary")
            if examples:
                gr.Examples(examples, inputs=[audio_in, phone_in], label="Examples (unseen test speaker)")
        with gr.Column(scale=1):
            verdict_out = gr.Markdown()
            probs_out = gr.Label(num_top_classes=6, label="Emotion probabilities")
    timeline_out = gr.Plot(label="Escalation timeline (longer audio)")
    run.click(analyse, inputs=[audio_in, phone_in], outputs=[probs_out, verdict_out, timeline_out])

if __name__ == "__main__":
    demo.launch()
