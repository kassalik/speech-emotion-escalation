"""Step 3: build a unified metadata table with labels and a speaker-independent split."""
import random

import pandas as pd
import soundfile as sf

from config import (CLASS_TO_IDX, CREMA_DEMOGRAPHICS, CREMA_DIR, CREMA_EMOTION, CREMA_INTENSITY,
                    ESCALATE_CLASSES, METADATA_CSV, RAVDESS_DIR, RAVDESS_EMOTION, RAVDESS_INTENSITY,
                    ROOT, SEED, SPLIT_FRACTIONS)


def _audio_info(path):
    info = sf.info(path)
    return {"orig_sr": info.samplerate, "channels": info.channels, "duration": round(info.duration, 3)}


def load_ravdess():
    """Filename: 03-01-EE-II-SS-RR-AA.wav (EE emotion, II intensity, SS statement, AA actor; odd actor = male)."""
    rows = []
    for path in sorted(RAVDESS_DIR.glob("Actor_*/*.wav")):
        _, _, emo, inten, stmt, _, actor = path.stem.split("-")
        emotion = RAVDESS_EMOTION[emo]
        if emotion is None:  # surprised has no CREMA-D counterpart
            continue
        rows.append({
            "path": str(path.relative_to(ROOT)), "dataset": "RAVDESS",
            "speaker": f"RAV_{actor}", "sex": "Male" if int(actor) % 2 else "Female",
            "emotion": emotion, "orig_emotion": emo, "intensity": RAVDESS_INTENSITY[inten],
            "sentence": f"S{stmt}", **_audio_info(path),
        })
    return pd.DataFrame(rows)


def load_crema():
    """Filename: SSSS_SEN_EMO_LVL.wav; sex from VideoDemographics.csv."""
    sex = pd.read_csv(CREMA_DEMOGRAPHICS).set_index("ActorID")["Sex"].to_dict()
    rows = []
    for path in sorted(CREMA_DIR.glob("*.wav")):
        spk, sent, emo, lvl = path.stem.split("_")
        rows.append({
            "path": str(path.relative_to(ROOT)), "dataset": "CREMA-D",
            "speaker": f"CRE_{spk}", "sex": sex[int(spk)],
            "emotion": CREMA_EMOTION[emo], "orig_emotion": emo, "intensity": CREMA_INTENSITY[lvl],
            "sentence": sent, **_audio_info(path),
        })
    return pd.DataFrame(rows)


def speaker_split(df, fractions=SPLIT_FRACTIONS, seed=SEED):
    """Assign each speaker to exactly one split, stratified by dataset and sex."""
    rng = random.Random(seed)
    assignment = {}
    for _, group in df.groupby(["dataset", "sex"]):
        speakers = sorted(group["speaker"].unique())
        rng.shuffle(speakers)
        n = len(speakers)
        n_val = round(n * fractions[1])
        n_test = round(n * fractions[2])
        for i, s in enumerate(speakers):
            assignment[s] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return df["speaker"].map(assignment)


def build_metadata():
    df = pd.concat([load_ravdess(), load_crema()], ignore_index=True)
    df["label"] = df["emotion"].map(CLASS_TO_IDX)
    df["escalate"] = df["emotion"].isin(ESCALATE_CLASSES).astype(int)
    df["split"] = speaker_split(df)
    METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(METADATA_CSV, index=False)
    return df


if __name__ == "__main__":
    meta = build_metadata()
    print(f"Saved {len(meta)} rows to {METADATA_CSV}")
