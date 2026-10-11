"""ONNX Runtime predictor for single clips and whole calls."""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .audio import fix_length, load_audio, logmel_db, normalise_loudness, telephone, trim_silence

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MIN_SPEECH_DBFS = -65.0   # below this after trimming = no speech; quietest real dataset clips are about -63 dBFS


class EscalationPredictor:
    """Speech emotion + escalation predictor.

    >>> p = EscalationPredictor()
    >>> p.predict("clip.wav")                 # <= 3 s of speech: one decision
    >>> p.predict("call.wav", phone=True)     # longer audio: sliding windows + call verdict
    """

    def __init__(self, model_dir=DEFAULT_MODEL_DIR, threads=None):
        model_dir = Path(model_dir)
        self.cfg = json.loads((model_dir / "config.json").read_text())
        fe = np.load(model_dir / "frontend.npz")
        self.mel_fb, self.window = fe["mel_fb"], fe["window"]
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_dir / self.cfg["model"]), opts,
                                            providers=["CPUExecutionProvider"])
        self.classes = self.cfg["classes"]
        self.esc_idx = [self.classes.index(c) for c in self.cfg["escalate_classes"]]
        self.threshold = self.cfg["threshold"]
        self.sr, self.n = self.cfg["sample_rate"], self.cfg["clip_samples"]

    # ---------------------------------------------------------------- building blocks
    def _norm(self, y):
        return normalise_loudness(y, self.cfg["target_rms_dbfs"], self.cfg["peak_limit"])

    def _phone(self, clean, seg=slice(None)):
        c = self.cfg
        ph = telephone(clean, self.sr, c["phone_sr"], tuple(c["phone_band"]), c["mu"])
        ph = ph * np.sqrt(np.mean(clean[seg] ** 2)) / (np.sqrt(np.mean(ph[seg] ** 2)) + 1e-9)
        return np.clip(ph, -c["peak_limit"], c["peak_limit"])

    def score(self, clips):
        """clips: [N, clip_samples] float32 -> probabilities [N, 6]."""
        mels = np.stack([logmel_db(c, self.mel_fb, self.window, self.cfg["n_fft"], self.cfg["hop"],
                                   self.cfg["top_db"]) for c in clips])
        return self.session.run(None, {"logmel_db": mels})[0]

    def prepare_clip(self, y, phone=False):
        """Exactly the training pipeline: trim -> normalise -> fix length (-> telephone)."""
        y = self._norm(trim_silence(y, self.cfg["trim_margin_db"], tuple(self.cfg["trim_top_db_range"])))
        clean, start, length = fix_length(y, self.n)
        return self._phone(clean, slice(start, start + length)) if phone else clean

    def prepare_window(self, seg, phone=False):
        clean = self._norm(seg)
        return self._phone(clean) if phone else clean

    # ---------------------------------------------------------------- public API
    def predict(self, audio, sr=None, phone=False):
        y = load_audio(audio, sr, self.sr)
        trimmed = trim_silence(y, self.cfg["trim_margin_db"], tuple(self.cfg["trim_top_db_range"]))
        level = 20 * np.log10(np.sqrt(np.mean(trimmed ** 2)) + 1e-12) if len(trimmed) else -np.inf
        if level < MIN_SPEECH_DBFS:
            return {"mode": "empty", "seconds": len(y) / self.sr, "escalate": False}
        if len(trimmed) <= self.n:
            return self._clip_result(self.score([self.prepare_clip(y, phone)])[0], len(y))
        return self.predict_call(y, phone)

    def predict_call(self, y, phone=False):
        c = self.cfg
        hop = int(c["window_hop_seconds"] * self.sr)
        starts = list(range(0, max(len(y) - self.n, 0) + 1, hop))
        if starts[-1] + self.n < len(y):
            starts.append(len(y) - self.n)
        segs = [y[s:s + self.n] for s in starts]
        rms_db = np.array([20 * np.log10(np.sqrt(np.mean(s ** 2)) + 1e-9) for s in segs])
        speech = (rms_db >= rms_db.max() - c["vad_rel_db"]) & (rms_db >= c["vad_abs_dbfs"])
        probs = self.score([self.prepare_window(s, phone) for s in segs])
        p_esc = probs[:, self.esc_idx].sum(1)
        windows = [{"start": s / self.sr, "speech": bool(v), "emotion": self.classes[int(p.argmax())],
                    "p_escalate": float(pe), "probabilities": {c: float(x) for c, x in zip(self.classes, p)}}
                   for s, v, p, pe in zip(starts, speech, probs, p_esc)]
        idx = np.flatnonzero(speech)
        if len(idx) == 0:
            return {"mode": "call", "seconds": len(y) / self.sr, "escalate": False, "peak_score": None,
                    "windows": windows}
        sc = p_esc[idx]
        pair = (sc[:-1] + sc[1:]) / 2 if len(sc) > 1 else sc
        k = int(np.argmax(pair))
        return {"mode": "call", "seconds": len(y) / self.sr, "escalate": bool(pair[k] >= self.threshold),
                "peak_score": float(pair[k]), "peak_start": starts[idx[k]] / self.sr,
                "threshold": self.threshold, "windows": windows}

    def _clip_result(self, p, n_samples):
        p_esc = float(p[self.esc_idx].sum())
        return {"mode": "clip", "seconds": n_samples / self.sr, "emotion": self.classes[int(p.argmax())],
                "probabilities": {c: float(v) for c, v in zip(self.classes, p)}, "p_escalate": p_esc,
                "escalate": p_esc >= self.threshold, "threshold": self.threshold}
