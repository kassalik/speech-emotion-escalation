"""Project-wide paths, label definitions and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

RAVDESS_DIR = RAW / "ravdess"
CREMA_DIR = RAW / "crema-d"
CREMA_DEMOGRAPHICS = RAW / "crema_demographics.csv"
METADATA_CSV = PROCESSED / "metadata.csv"

SEED = 42

# Six emotion classes shared by both datasets
CLASSES = ["neutral", "happy", "sad", "angry", "fear", "disgust"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Business mapping: emotions that should trigger an escalation / QA review
ESCALATE_CLASSES = {"angry", "disgust", "fear", "sad"}

# RAVDESS emotion code -> shared class (calm merged into neutral, surprised dropped)
RAVDESS_EMOTION = {
    "01": "neutral", "02": "neutral", "03": "happy", "04": "sad",
    "05": "angry", "06": "fear", "07": "disgust", "08": None,
}
RAVDESS_INTENSITY = {"01": "normal", "02": "strong"}

# CREMA-D emotion code -> shared class
CREMA_EMOTION = {
    "NEU": "neutral", "HAP": "happy", "SAD": "sad",
    "ANG": "angry", "FEA": "fear", "DIS": "disgust",
}
CREMA_INTENSITY = {"LO": "low", "MD": "medium", "HI": "high", "XX": "unspecified"}

# Speaker-independent split proportions (train / val / test)
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)

# Output folders are created on import so a fresh checkout can run end to end
for _d in (PROCESSED, FIGURES, RESULTS / "models"):
    _d.mkdir(parents=True, exist_ok=True)

# Audio settings (used from Step 4 onwards)
SAMPLE_RATE = 16000
CLIP_SECONDS = 3.0
