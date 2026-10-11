# Model Card: CNN speech-emotion / escalation classifier

## Model details

| | |
|---|---|
| **Model** | 2-D convolutional neural network on log-mel spectrograms (`CNN` in `src/train_cnn.py`) |
| **Version** | `cnn_s42` (seed 42, fixed in advance as the release model); seeds 7 and 123 used to measure variance |
| **Size** | 1.18 M parameters, 4.6 MB (PyTorch state dict) |
| **Architecture** | 4 × [Conv3×3 → BatchNorm → ReLU] × 2 → MaxPool 2×2 → Dropout2d(0.1), channels 32 / 64 / 128 / 256; mean over frequency; mean + max pooling over time; Dropout(0.3); linear layer → 6 classes |
| **Input** | 3 s of 16 kHz mono audio → 64-band log-mel spectrogram (25 ms window, 10 ms hop, 64 × 301), per-band standardised with training statistics stored in the checkpoint |
| **Output** | Softmax over 6 emotions (neutral, happy, sad, angry, fear, disgust); P(escalate) = P(angry) + P(disgust) + P(fear) + P(sad) |
| **Decision threshold** | 0.416 on P(escalate), chosen on validation telephone audio as the highest threshold with ≥ 90% Escalate recall |
| **Author** | Muhammad Kassali K |
| **Licence** | CC BY-NC-SA 4.0 (the weights are derived from RAVDESS) |

## Release artefacts

| File | Description |
|---|---|
| `models/ser_cnn.onnx` | ONNX (opset 18) graph: log-mel dB `[N, 64, 301]` → probabilities `[N, 6]`, per-band standardisation built in; 4.7 MB; max difference vs PyTorch < 3 × 10⁻⁷ |
| `models/frontend.npz` | Mel filterbank and Hann window exported from torchaudio, used by the NumPy front-end |
| `models/config.json` | Class order, escalation classes, threshold 0.416, pre-processing and windowing constants |

Runtime: `inference/` (ONNX Runtime + NumPy/SciPy). On all 1,274 test clips it agrees with the PyTorch pipeline on
99.5% (clean) / 97.6% (telephone) of top-1 predictions and 99.8% / 99.5% of escalation decisions. Input quieter than
−65 dBFS after silence trimming is reported as "no speech" instead of being classified.

Latency (laptop AMD Ryzen 7 260, CPU): about 15 ms per 3 s window on one thread, about 6–8 ms on four threads, about
175 MB peak memory.

## Intended use

- **Primary use:** decision support for customer-support quality teams: ranking or flagging recorded calls that
  likely contain negative emotion, for **human review**.
- **Also suitable for:** research, teaching and demonstrations of speech emotion recognition and robustness to
  channel effects.
- **Users:** QA analysts, support supervisors, ML practitioners.

### Out of scope

- Automated decisions about individual customers (for example service denial, credit or fraud decisions).
- Use as a lie detector, a mental-health assessment, or any clinical or legal judgement.
- Covert monitoring of people without their knowledge and consent.
- Commercial use of the released weights (not permitted by the licence).

## Training data

| | |
|---|---|
| **Datasets** | CREMA-D (7,442 clips, 91 speakers, 16 kHz) and RAVDESS speech (1,248 of 1,440 clips, 24 speakers, 48 kHz) |
| **Labels** | Six shared emotions; RAVDESS *calm* merged into *neutral*, *surprised* dropped |
| **Split** | Speaker-independent 70 / 15 / 15, stratified by dataset and sex: train 6,142 clips (81 speakers), validation 1,274 (17), test 1,274 (17); no speaker in more than one split |
| **Pre-processing** | Resampled to 16 kHz mono, silence trimmed with a noise-floor-adaptive threshold, RMS-normalised to −23 dBFS, centre-cropped / zero-padded to 3 s |
| **Augmentation (training only)** | Simulated telephone channel (8 kHz, 300–3400 Hz band-pass, G.711 μ-law) on 50% of clips; white noise at 10–20 dB SNR (50%); random circular time shift (50%); SpecAugment (2 frequency masks ≤ 8 bands, 2 time masks ≤ 30 frames) |
| **Optimisation** | AdamW (lr 1e-3, weight decay 1e-2), one-cycle schedule, 60 epochs, batch 64, label smoothing 0.05, class-weighted cross-entropy, bfloat16 mixed precision; model selected on mean validation macro-F1 over clean and telephone audio |
| **Hardware** | NVIDIA RTX 5060 Laptop GPU (8 GB); about 6–8 minutes per training run |

## Evaluation

Held-out test set: 1,274 clips from 17 speakers not used for training or tuning.

### Emotion recognition

| | Clean audio | Telephone audio |
|---|---|---|
| Accuracy (release model) | 65.6% | 64.1% |
| Macro-F1 (release model) | 0.655 | 0.642 |
| Macro-F1 (mean ± std, 3 seeds) | 0.654 ± 0.010 | 0.642 ± 0.003 |

Per-class recall, release model, telephone audio: neutral 0.80 · angry 0.71 · happy 0.64 · sad 0.60 ·
disgust 0.59 · fear 0.52.

### Escalation decision (telephone audio, threshold 0.416)

| | Release model | Flag every call |
|---|---|---|
| Escalate recall | 0.895 | 1.00 |
| Precision | 0.853 | 0.67 |
| Alerts per 100 calls | 70.5 | 100 |
| False alerts per 100 calls | 10.4 | 32.8 |
| ROC AUC | 0.885 | 0.50 |

### Main error modes

- **angry → happy (14%)**: the business-critical miss; both are high-arousal, high-pitch emotions.
- **fear → sad (25%)** and **sad → neutral (14%)**: confusions among low-energy emotions.
- angry ↔ disgust confusions are frequent but do not change the escalation decision.

## Limitations

- **Acted, scripted speech.** Both datasets use actors reading fixed sentences. Real customer frustration is subtler,
  often mixed, and develops over a conversation, so real-world accuracy is expected to be lower until the model is
  validated and fine-tuned on real recordings.
- **Simulated channel.** Only a narrow-band G.711 channel was simulated; other codecs (Opus, AMR, G.729), packet
  loss, echo and environmental noise were not.
- **Clip-level calibration.** The threshold was tuned on 2–3 s clips. Applied to sliding windows over a whole call,
  it can raise false alarms; call-level thresholds need calibration on real calls.
- **Mixed speakers.** The model scores whatever voice it hears; agent and customer must be separated (diarisation)
  before scoring real calls.
- **Data quality.** One CREMA-D training clip (`1076_MTI_SAD_XX.wav`, labelled *sad*) is digital silence. It is
  1 of 6,142 training clips and the test set is unaffected, but it teaches the model that silence looks "sad",
  hence the −65 dBFS speech gate in the runtime. It should be excluded in future training runs.
- **Language and accent coverage.** The training speech is English; performance on other languages and accents has
  not been evaluated.

## Ethical considerations

- Voice is personal data. Deployments need a lawful basis and informed consent, PII redaction, access control and
  data-residency compliance; on-device (edge) processing reduces exposure.
- Emotion recognition can be biased across demographic groups. Per-group performance (for example by sex, age or
  accent) has not yet been audited and should be before any deployment.
- Outputs should support, not replace, human judgement; every alert should be explainable and logged.

## Citation of training data

- Cao, H., Cooper, D. G., Keutmann, M. K., Gur, R. C., Nenkova, A., & Verma, R. (2014). CREMA-D: Crowd-sourced
  Emotional Multimodal Actors Dataset. *IEEE Transactions on Affective Computing*, 5(4), 377–390.
- Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song
  (RAVDESS). *PLoS ONE*, 13(5), e0196391.
