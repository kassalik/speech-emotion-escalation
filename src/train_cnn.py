"""Step 7: CNN (and CNN + BiLSTM) on log-mel spectrograms, trained on the GPU.

Training-time augmentation (on the fly, per example):
  - telephone channel: 50% of clips use the pre-computed telephone version (Step 4)
  - additive white noise at a random 10-20 dB SNR (50%)
  - random circular time shift (50%)
  - SpecAugment: 2 frequency masks (<= 8 bands) + 2 time masks (<= 30 frames)
Log-mels are computed on the GPU with the same transform as Step 5 and standardised with
per-band mean/std from the training split.  Model selection: best mean validation macro-F1
over clean and telephone audio (the deployment target is phone audio).

Usage: python train_cnn.py [cnn|crnn] [seed]
"""
import json
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import plotstyle
from baseline import MODELS_DIR, metrics_from_proba, plot_confusions
from config import CLASSES, FIGURES, METADATA_CSV, PROCESSED, RESULTS, SEED
from features import N_MELS, make_logmel

EPOCHS = 60
PATIENCE = 12
BATCH = 64
LR = 1e-3
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.05
P_PHONE, P_NOISE, P_SHIFT = 0.5, 0.5, 0.5
SNR_RANGE = (10.0, 20.0)


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.1))

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """4 conv blocks -> mean over frequency -> mean+max over time -> linear."""

    def __init__(self, n_classes=len(CLASSES), channels=(32, 64, 128, 256)):
        super().__init__()
        chans = (1,) + channels
        self.blocks = nn.Sequential(*[ConvBlock(a, b) for a, b in zip(chans[:-1], chans[1:])])
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * channels[-1], n_classes))

    def forward(self, x):                      # x: [B, 1, mels, frames]
        h = self.blocks(x).mean(dim=2)         # [B, C, T]
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], dim=1))


class CRNN(nn.Module):
    """3 conv blocks -> BiLSTM over time -> mean+max pooling -> linear."""

    def __init__(self, n_classes=len(CLASSES), channels=(32, 64, 128), hidden=128):
        super().__init__()
        chans = (1,) + channels
        self.blocks = nn.Sequential(*[ConvBlock(a, b) for a, b in zip(chans[:-1], chans[1:])])
        self.rnn = nn.LSTM(channels[-1] * (N_MELS // 8), hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(4 * hidden, n_classes))

    def forward(self, x):
        h = self.blocks(x)                     # [B, C, F', T']
        h = h.flatten(1, 2).transpose(1, 2)    # [B, T', C*F']
        h, _ = self.rnn(h)
        return self.head(torch.cat([h.mean(1), h.amax(1)], dim=1))


ARCHS = {"cnn": CNN, "crnn": CRNN}


def augment(clean, phone, gen):
    """Waveform-level augmentation on GPU. clean/phone: [B, T] float32."""
    B = clean.shape[0]
    use_phone = torch.rand(B, 1, device=clean.device, generator=gen) < P_PHONE
    x = torch.where(use_phone, phone, clean)
    noisy = torch.rand(B, device=x.device, generator=gen) < P_NOISE
    snr = SNR_RANGE[0] + (SNR_RANGE[1] - SNR_RANGE[0]) * torch.rand(B, device=x.device, generator=gen)
    sig_pow = x.pow(2).mean(dim=1).clamp_min(1e-8)
    noise = torch.randn(x.shape, device=x.device, generator=gen) * torch.sqrt(sig_pow / 10 ** (snr / 10)).unsqueeze(1)
    x = x + noise * noisy.unsqueeze(1)
    shift = torch.randint(0, x.shape[1], (B,), device=x.device, generator=gen)
    shift = torch.where(torch.rand(B, device=x.device, generator=gen) < P_SHIFT, shift, torch.zeros_like(shift))
    idx = (torch.arange(x.shape[1], device=x.device).unsqueeze(0) - shift.unsqueeze(1)) % x.shape[1]
    return torch.gather(x, 1, idx)


def spec_augment(m):
    fm = torchaudio.transforms.FrequencyMasking(8, iid_masks=True)
    tm = torchaudio.transforms.TimeMasking(30, iid_masks=True)
    for _ in range(2):
        m = fm(m)
        m = tm(m)
    return m


def predict_proba(model, logmel, mean, std, batch=256):
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for lo in range(0, len(logmel), batch):
            x = ((logmel[lo:lo + batch].float() - mean) / std).unsqueeze(1)
            out.append(F.softmax(model(x).float(), dim=1))
    return torch.cat(out).cpu().numpy()


def plot_history(hist, name):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ep = hist["epoch"]
    axes[0].plot(ep, hist["train_loss"], color=plotstyle.SERIES[0], lw=2, label="Train loss")
    axes[0].set_title("Training loss (with augmentation)", fontsize=10)
    axes[0].set_xlabel("Epoch")
    for col, label, color in [("val_f1_clean", "Val clean", plotstyle.SERIES[0]),
                              ("val_f1_phone", "Val phone", plotstyle.SERIES[1]),
                              ("val_f1_mean", "Mean (selection)", plotstyle.SERIES[2])]:
        axes[1].plot(ep, hist[col], color=color, lw=2, label=label)
    best = hist.loc[hist["val_f1_mean"].idxmax()]
    axes[1].axvline(best["epoch"], color=plotstyle.TEXT_2, ls="--", lw=1)
    axes[1].text(best["epoch"] + 0.5, 0.12, f"best epoch {int(best['epoch'])}", fontsize=9, color=plotstyle.TEXT_2)
    axes[1].set_ylim(0.1, max(0.7, hist[["val_f1_clean", "val_f1_phone"]].max().max() + 0.05))
    axes[1].set_title("Validation macro-F1", fontsize=10)
    axes[1].set_xlabel("Epoch")
    axes[1].legend(loc="upper left")
    fig.suptitle(f"{name.upper()} training curves", x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / f"09_training_curves_{name}.png")
    plt.close(fig)


def main(arch="cnn", seed=SEED):
    if not torch.cuda.is_available():
        raise RuntimeError("CNN training needs an NVIDIA GPU with CUDA (PyTorch CUDA 12.8 build, see requirements.txt).")
    plotstyle.apply()
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(seed)
    meta = pd.read_csv(METADATA_CSV)
    tr = np.where(meta["split"] == "train")[0]
    va = np.where(meta["split"] == "val")[0]
    y_tr = torch.tensor(meta["label"].values[tr], device=dev)
    y_va = meta["label"].values[va]

    # Training waveforms live on the GPU (float16, ~1.2 GB for clean + phone)
    wav_clean = torch.from_numpy(np.load(PROCESSED / "audio_clean.npy", mmap_mode="r")[tr]).to(dev)
    wav_phone = torch.from_numpy(np.load(PROCESSED / "audio_phone.npy", mmap_mode="r")[tr]).to(dev)
    val_mel = {v: torch.from_numpy(np.load(PROCESSED / f"logmel_{v}.npy", mmap_mode="r")[va]).to(dev)
               for v in ("clean", "phone")}

    # Per-band normalisation statistics from training log-mels (clean + phone)
    train_mel = np.concatenate([np.load(PROCESSED / f"logmel_{v}.npy", mmap_mode="r")[tr] for v in ("clean", "phone")])
    mean = torch.tensor(train_mel.astype(np.float32).mean(axis=(0, 2)), device=dev).view(1, -1, 1)
    std = torch.tensor(train_mel.astype(np.float32).std(axis=(0, 2)), device=dev).view(1, -1, 1)
    del train_mel

    to_logmel = make_logmel(dev)
    model = ARCHS[arch]().to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    counts = torch.bincount(y_tr, minlength=len(CLASSES)).float()
    criterion = nn.CrossEntropyLoss(weight=(counts.sum() / (len(CLASSES) * counts)), label_smoothing=LABEL_SMOOTHING)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps = EPOCHS * int(np.ceil(len(tr) / BATCH))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.15)

    name = f"{arch}_s{seed}"
    history, best_f1, best_epoch, t0 = [], -1.0, 0, time.time()
    print(f"{name}: {n_params / 1e6:.2f} M parameters, {len(tr)} train clips", flush=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(tr), device=dev, generator=gen)
        losses = []
        for lo in range(0, len(tr), BATCH):
            b = perm[lo:lo + BATCH]
            with torch.no_grad():
                wave = augment(wav_clean[b].float(), wav_phone[b].float(), gen)
                x = spec_augment(((to_logmel(wave) - mean) / std).unsqueeze(1))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = criterion(model(x).float(), y_tr[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        f1 = {v: metrics_from_proba(predict_proba(model, val_mel[v], mean, std), y_va)["macro_f1"]
              for v in ("clean", "phone")}
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_f1_clean": f1["clean"],
               "val_f1_phone": f1["phone"], "val_f1_mean": (f1["clean"] + f1["phone"]) / 2}
        history.append(row)
        if row["val_f1_mean"] > best_f1:
            best_f1, best_epoch = row["val_f1_mean"], epoch
            torch.save({"state_dict": model.state_dict(), "arch": arch, "mean": mean.cpu(), "std": std.cpu(),
                        "classes": CLASSES, "epoch": epoch}, MODELS_DIR / f"{name}.pt")
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:2d} loss {row['train_loss']:.3f} | val F1 clean {f1['clean']:.3f} "
                  f"phone {f1['phone']:.3f} | best {best_f1:.3f} @ {best_epoch} | {time.time() - t0:.0f}s", flush=True)
        if epoch - best_epoch >= PATIENCE:
            print(f"  early stop at epoch {epoch}", flush=True)
            break

    hist = pd.DataFrame(history)
    hist.to_csv(RESULTS / f"history_{name}.csv", index=False)
    plot_history(hist, name)

    ckpt = torch.load(MODELS_DIR / f"{name}.pt", map_location=dev)
    model.load_state_dict(ckpt["state_dict"])
    res = {"arch": arch, "seed": seed, "params": n_params, "best_epoch": best_epoch,
           "train_seconds": round(time.time() - t0, 1)}
    for v in ("clean", "phone"):
        res[v] = metrics_from_proba(predict_proba(model, val_mel[v], mean, std), y_va)
    plot_confusions(res, name)
    with open(RESULTS / f"metrics_step7_{name}.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"{name} best epoch {best_epoch}: val clean acc {res['clean']['accuracy']:.3f} F1 {res['clean']['macro_f1']:.3f}"
          f" | phone acc {res['phone']['accuracy']:.3f} F1 {res['phone']['macro_f1']:.3f}"
          f" | esc AUC clean {res['clean']['escalate']['roc_auc']:.3f} phone {res['phone']['escalate']['roc_auc']:.3f}"
          f" | {res['train_seconds']}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cnn", int(sys.argv[2]) if len(sys.argv) > 2 else SEED)
