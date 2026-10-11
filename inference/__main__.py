"""python -m inference FILE.wav [--phone] [--json]"""
import argparse
import json

from .predictor import EscalationPredictor


def main():
    ap = argparse.ArgumentParser(prog="python -m inference", description="Speech emotion / escalation prediction")
    ap.add_argument("file")
    ap.add_argument("--phone", action="store_true", help="simulate a telephone channel first")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    a = ap.parse_args()
    r = EscalationPredictor().predict(a.file, phone=a.phone)
    if a.json:
        print(json.dumps(r, indent=2))
    elif r["mode"] == "clip":
        top = sorted(r["probabilities"].items(), key=lambda kv: -kv[1])
        print("  ".join(f"{c} {v:.2f}" for c, v in top))
        print(f"Emotion: {r['emotion'].upper()} | P(escalate) {r['p_escalate']:.2f} | "
              f"{'ESCALATE' if r['escalate'] else 'OK'} (threshold {r['threshold']:.2f})")
    elif r["mode"] == "call":
        for w in r["windows"]:
            mark = "  no speech" if not w["speech"] else ("  <" if w["p_escalate"] >= r.get("threshold", 1) else "")
            print(f"{w['start']:6.1f}s  {(w['emotion'] if w['speech'] else '-'):<8} {w['p_escalate']:.2f}{mark}")
        print(f"Call verdict: {'ESCALATE' if r['escalate'] else 'OK'}")
    else:
        print("No speech detected.")


if __name__ == "__main__":
    main()
