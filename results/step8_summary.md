# Step 8: held-out test results (1,274 clips, 17 unseen speakers)

## 6-class emotion recognition (macro-F1; chance ≈ 0.17)
| Model | Clean | Phone | Drop |
|---|---|---|---|
| SVM (clean-trained) | 0.542 | 0.209 | −0.333 |
| Random Forest | 0.516 | 0.367 | −0.149 |
| SVM (multi-condition: clean + phone training) | 0.533 | 0.519 | −0.014 |
| **CNN (mean ± std, 3 seeds)** | **0.654 ± 0.010** | **0.642 ± 0.003** | −0.012 |
| CNN + BiLSTM | 0.628 | 0.606 | −0.022 |

Final model = CNN seed 42 (chosen in advance): clean acc 65.6% / F1 0.655; phone acc 64.1% / F1 0.642.
Per-class recall (phone): neutral 0.80, angry 0.71, happy 0.64, sad 0.60, disgust 0.59, fear 0.52.

## Escalation decision (phone test audio, threshold chosen on validation for ≥ 90% recall)
| Model | Threshold | Recall | Precision | Alerts / 100 calls | False alerts / 100 | Missed / 100 | AUC |
|---|---|---|---|---|---|---|---|
| Flag every call | — | 1.00 | 0.67 | 100 | 32.8 | 0 | 0.50 |
| SVM (clean-trained) | 0.295 | 0.88 | 0.70 | 83.7 | 24.8 | 8.3 | 0.645 |
| SVM (multi-condition) | 0.399 | 0.87 | 0.76 | 76.6 | 18.1 | 8.7 | 0.787 |
| **CNN (3 seeds)** | 0.42–0.46 | **0.90** | **0.85** | **71.0** | **10.5** | 6.8 | **0.889** |

- The validation-chosen threshold transferred to the test set: target recall 0.90, achieved 0.90 (3-seed mean).
- CNN cuts false alerts from 32.8 to 10.5 per 100 calls (−68%) versus flagging everything.
- Prioritised review (phone): if QA reviews only the top-ranked 10 / 20 / 30 / 50 calls per 100, the CNN's
  precision is 0.99 / 0.97 / 0.96 / 0.92 (clean-trained SVM: 0.83 / 0.83 / 0.82 / 0.76).

## Where the CNN's advantage comes from (phone macro-F1)
- Clean-trained SVM → multi-condition SVM: +0.31 (training on telephone audio)
- Multi-condition SVM → CNN: +0.12 (learned spectro-temporal features vs handcrafted statistics)

## Remaining errors (CNN, phone test)
- fear → sad 25%, sad → neutral 14%, fear → happy 15%
- angry → happy 14%: the business-critical miss (Escalate predicted as No-escalate)
- angry ↔ disgust confusions do not affect the escalation decision (both Escalate)

## Baseline feature importance (grouped permutation, clean validation, SVM)
MFCC std 0.135 > MFCC ΔΔ 0.111 > MFCC means 0.103 > MFCC Δ 0.092 ≫ spectral shape 0.030, voicing 0.028,
energy 0.017, pitch 0.016, ZCR 0.012. Correlated families can stand in for each other, so these are lower bounds.
