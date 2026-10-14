"""Train the flight-call species classifier on BirdVox-ANAFCC.

Clips from the same site and night are kept together (all in train or all in
test) so the test score isn't inflated by near-duplicate calls.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, MODEL_DIR, OUTPUT_DIR
from pretrained import resample
from species_cnn import CODE_NAMES, CROP_N, CropFrontend, SpeciesNet, center_crop

ANAFCC = DATA_DIR / "birdvox_anafcc"
MIN_PER_CLASS = 150


def label_of(key: str, file_code: str) -> str | None:
    if file_code.startswith("0"):
        return "NOISE"
    parts = re.split(r"[_\-]", key)
    for p in reversed(parts):
        if p in CODE_NAMES:
            return p
        if p.upper() in CODE_NAMES and p.upper() != "NOISE":
            return p.upper()
    m = re.match(r"([A-Z]{4})", key)          # CLO-43SD keys start with the code
    if m and m.group(1) in CODE_NAMES:
        return m.group(1)
    return None


def group_of(key: str) -> str:
    m = re.match(r"([A-Za-z]+_\d{4}-\d{2}-\d{2})", key)       # site_date (Old Bird)
    if m:
        return m.group(1)
    m = re.match(r"(xc-\d+|\d+_n|additional-v2_\d+)", key)
    return m.group(1) if m else key[:10]


def load_dataset():
    cache = DATA_DIR / "features" / "anafcc_crops.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d["x"], d["y"], d["g"]
    xs, ys, gs = [], [], []
    for f in sorted(ANAFCC.glob("*_original.h5")):
        with h5py.File(f, "r") as h:
            if "waveforms" not in h:
                continue
            sr = int(h["sample_rate"][()])
            code = h["taxonomy_code"][()].decode()
            for k in h["waveforms"]:
                lab = label_of(k, code)
                if lab is None:
                    continue
                x = resample(h["waveforms"][k][()].astype(np.float32), sr, 24000)
                xs.append(center_crop(x))
                ys.append(lab)
                gs.append(group_of(k))
    x, y, g = np.stack(xs), np.array(ys), np.array(gs)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, x=x, y=y, g=g)
    return x, y, g


def main(epochs=30, batch=256, lr=1e-3):
    torch.manual_seed(0)
    x, y, g = load_dataset()
    counts = Counter(y)
    keep = np.array([counts[v] >= MIN_PER_CLASS for v in y])
    x, y, g = x[keep], y[keep], g[keep]
    classes = sorted(set(y))
    yi = np.array([classes.index(v) for v in y])
    print("classes:", {c: counts[c] for c in classes})

    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(x, yi, g))
    device = "cuda"
    fe = CropFrontend(device)
    xt = torch.from_numpy(x).to(device)
    yt = torch.from_numpy(yi).to(device)
    # Balance classes: sample each class roughly equally often (capped weight).
    cw = np.bincount(yi[tr], minlength=len(classes)).astype(float)
    sample_w = torch.tensor((1 / cw[yi[tr]]) ** 0.5, device=device)

    model = SpeciesNet(len(classes)).to(device)
    steps = max(1, len(tr) // batch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * steps)
    tr_t = torch.from_numpy(tr).to(device)
    noise_t = torch.from_numpy(tr[yi[tr] == classes.index("NOISE")]).to(device)
    for ep in range(epochs):
        model.train()
        tot = 0
        for _ in range(steps):
            idx = tr_t[torch.multinomial(sample_w, batch, replacement=True)]
            wav = xt[idx]
            # Augment: time shift, gain, add a random NOISE clip underneath.
            wav = torch.roll(wav, int(np.random.randint(-1200, 1200)), 1)
            wav = wav * torch.empty(batch, 1, device=device).uniform_(0.3, 2.0)
            noise_idx = noise_t[torch.randint(len(noise_t), (batch,), device=device)]
            wav = wav + 0.5 * torch.rand(batch, 1, device=device) * xt[noise_idx] * \
                (torch.rand(batch, 1, device=device) < 0.3)
            spec = fe(wav)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(spec), yt[idx], label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(fe(xt[te[i:i + 1024]])).argmax(1)
                              for i in range(0, len(te), 1024)]).cpu().numpy()
        acc = (pred == yi[te]).mean()
        bal = np.mean([(pred[yi[te] == c] == c).mean() for c in np.unique(yi[te])])
        print(f"epoch {ep + 1:2d} loss {tot / steps:.3f} test acc {acc:.3f} balanced acc {bal:.3f}", flush=True)

    MODEL_DIR.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": classes},
               MODEL_DIR / "species_cnn.pt")
    names = [CODE_NAMES[c] for c in classes]
    rep = classification_report(yi[te], pred, labels=range(len(classes)), target_names=names,
                                output_dict=True, zero_division=0)
    out = OUTPUT_DIR / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rep).T.round(3).to_csv(out / "species_cnn_test_report.csv")
    (out / "species_cnn_summary.json").write_text(json.dumps(
        dict(test_accuracy=float(acc), balanced_accuracy=float(bal), n_train=int(len(tr)),
             n_test=int(len(te)), classes={c: int(counts[c]) for c in classes}), indent=2))
    print(classification_report(yi[te], pred, labels=range(len(classes)), target_names=names,
                                zero_division=0))


if __name__ == "__main__":
    main()
