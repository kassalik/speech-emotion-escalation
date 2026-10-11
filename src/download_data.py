"""Download and verify the two public datasets into data/raw/ (about 1.2 GB, any operating system).

  python src/download_data.py            # download what is missing, then verify
  python src/download_data.py --check    # verify only

RAVDESS (speech audio, Zenodo record 1188976) and CREMA-D (AudioWAV, GitHub / Git LFS).
Both are free for research use: RAVDESS CC BY-NC-SA 4.0, CREMA-D ODbL. Please cite the authors.
"""
import argparse
import hashlib
import json
import re
import shutil
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

from config import CREMA_DEMOGRAPHICS, CREMA_DIR, RAVDESS_DIR, RAW

RAVDESS_URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
RAVDESS_MD5 = "bc696df654c87fed845eb13823edef8a"
RAVDESS_FILES = 1440
CREMA_REPO = "CheyneyComputerScience/CREMA-D"
CREMA_MEDIA = f"https://media.githubusercontent.com/media/{CREMA_REPO}/master/AudioWAV/"
CREMA_DEMOGRAPHICS_URL = f"https://raw.githubusercontent.com/{CREMA_REPO}/master/VideoDemographics.csv"
CREMA_FILES = 7442
CREMA_NAME = re.compile(r"\d{4}_[A-Z]{3}_[A-Z]{3}_[A-Z]{2}\.wav")


def fetch(url, dest, retries=3):
    for attempt in range(retries):
        try:
            tmp = dest.with_name(dest.name + ".part")
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            tmp.replace(dest)
            return
        except OSError:
            if attempt == retries - 1:
                raise


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def get_ravdess():
    if len(list(RAVDESS_DIR.glob("Actor_*/*.wav"))) == RAVDESS_FILES:
        print("RAVDESS: already present")
        return
    RAW.mkdir(parents=True, exist_ok=True)
    zip_path = RAW / "Audio_Speech_Actors_01-24.zip"
    if not zip_path.exists() or md5(zip_path) != RAVDESS_MD5:
        print("RAVDESS: downloading about 200 MB from Zenodo ...")
        fetch(RAVDESS_URL, zip_path)
    if md5(zip_path) != RAVDESS_MD5:
        raise RuntimeError("RAVDESS zip checksum mismatch; delete it and try again")
    target = RAVDESS_DIR.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():  # refuse paths that would escape the target folder
            if not (target / member).resolve().is_relative_to(target):
                raise RuntimeError(f"unsafe path in zip: {member}")
        z.extractall(target)
    zip_path.unlink()
    print("RAVDESS: extracted")


def crema_file_list():
    api = f"https://api.github.com/repos/{CREMA_REPO}"
    root = json.load(urllib.request.urlopen(f"{api}/contents/", timeout=60))
    sha = next(e["sha"] for e in root if e["name"] == "AudioWAV")
    tree = json.load(urllib.request.urlopen(f"{api}/git/trees/{sha}", timeout=120))
    return sorted(e["path"] for e in tree["tree"] if e["type"] == "blob" and CREMA_NAME.fullmatch(e["path"]))


def get_crema(workers=16):
    CREMA_DIR.mkdir(parents=True, exist_ok=True)
    if not CREMA_DEMOGRAPHICS.exists():
        fetch(CREMA_DEMOGRAPHICS_URL, CREMA_DEMOGRAPHICS)
    if len(list(CREMA_DIR.glob("*.wav"))) == CREMA_FILES:
        print("CREMA-D: already present")
        return
    names = crema_file_list()
    todo = [n for n in names if not (CREMA_DIR / n).exists() or (CREMA_DIR / n).stat().st_size == 0]
    print(f"CREMA-D: downloading {len(todo):,} of {len(names):,} files (about 600 MB) ...")
    done = 0
    with ThreadPoolExecutor(workers) as pool:
        for _ in pool.map(lambda n: fetch(CREMA_MEDIA + n, CREMA_DIR / n), todo):
            done += 1
            if done % 1000 == 0:
                print(f"  {done:,}/{len(todo):,}", flush=True)


def verify():
    import soundfile as sf
    ok = True
    for name, files, expected in [("RAVDESS", sorted(RAVDESS_DIR.glob("Actor_*/*.wav")), RAVDESS_FILES),
                                  ("CREMA-D", sorted(CREMA_DIR.glob("*.wav")), CREMA_FILES)]:
        bad = [f.name for f in files if not _readable(sf, f)]
        status = "OK" if len(files) == expected and not bad else "PROBLEM"
        ok &= status == "OK"
        print(f"{name}: {len(files):,} / {expected:,} files, {len(bad)} unreadable -> {status}")
    print(f"CREMA-D demographics: {'OK' if CREMA_DEMOGRAPHICS.exists() else 'MISSING'}")
    return ok and CREMA_DEMOGRAPHICS.exists()


def _readable(sf, path):
    try:
        return sf.info(path).frames > 0
    except Exception:
        return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify existing files only")
    if not ap.parse_args().check:
        get_ravdess()
        get_crema()
    raise SystemExit(0 if verify() else 1)
