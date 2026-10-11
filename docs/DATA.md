# Datasets

The raw audio is not included in the submission. To reproduce, run `python src/download_data.py` from the project folder; it downloads both datasets into `data/raw/` and verifies them.

## RAVDESS (speech audio only)
- Source: https://zenodo.org/records/1188976 (`Audio_Speech_Actors_01-24.zip`, MD5 `bc696df654c87fed845eb13823edef8a`)
- Extract to `data/raw/ravdess/Actor_XX/*.wav`
- 1,440 clips, 24 actors (12 female, 12 male), 48 kHz
- Filename `03-01-EE-II-SS-RR-AA.wav`: EE = emotion (01 neutral, 02 calm, 03 happy, 04 sad, 05 angry, 06 fearful, 07 disgust, 08 surprised), AA = actor
- Licence: CC BY-NC-SA 4.0
- Citation: Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). PLoS ONE 13(5): e0196391.

## CREMA-D
- Source: https://github.com/CheyneyComputerScience/CREMA-D (`AudioWAV/`, Git LFS)
- Files saved flat to `data/raw/crema-d/*.wav`; file list in `data/raw/crema_files.txt`
- 7,442 clips, 91 actors, 16 kHz mono
- Filename `SSSS_SEN_EMO_LVL.wav`: SSSS = speaker, EMO = ANG / DIS / FEA / HAP / NEU / SAD
- Licence: Open Database License (ODbL)
- Citation: Cao, H., Cooper, D. G., Keutmann, M. K., Gur, R. C., Nenkova, A., & Verma, R. (2014). CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset. IEEE Transactions on Affective Computing 5(4): 377–390.

## Download commands used originally (Linux shell; `src/download_data.py` does the same on any OS)
```bash
curl -L -o Audio_Speech_Actors_01-24.zip "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
sed 's#^#https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/AudioWAV/#' crema_files.txt \
  | xargs -P 16 -n 20 curl -L --fail -sS --retry 3 --remote-name-all
```

## Example clips included in this repository
`results/demo/` and `results/audio_examples/` contain a few short clips from the two datasets (CREMA-D speaker 1001,
RAVDESS actor 01), either unmodified or processed by this project's pipeline, for demonstration only. They are
redistributed under the datasets' own terms: RAVDESS under CC BY-NC-SA 4.0 (attribution, non-commercial,
share-alike) and CREMA-D under the Open Database License / Database Contents License (attribution). Credit to the
dataset authors as cited above.
