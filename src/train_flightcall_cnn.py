"""Train our flight-call detector on BirdVox-full-night.

BirdVox-full-night (Lostanlen et al. 2018, CC-BY-4.0): six ~10-hour fall
migration nights recorded in upstate New York, with every flight call marked
by experts (~35,000 calls). Split by recording unit so the test night is never
seen in training:
    train: units 01, 02, 03, 05   validation: unit 07   test: unit 10
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, MODEL_DIR, OUTPUT_DIR
from flightcall_cnn import FRAME_S, SR, FlightCallNet, SNRFrontend, frame_probs, pick_events

BV = DATA_DIR / "birdvox_full_night"
FEAT = DATA_DIR / "features"
SPLITS = {"train": ["01", "02", "03", "05"], "val": ["07"], "test": ["10"]}
POS_HALF_S, IGNORE_HALF_S = 0.05, 0.12
CROP = 256
MATCH_TOL_S = 0.15


def unit_features(unit: str, fe: SNRFrontend) -> np.ndarray:
    """SNR map for a whole night, cached as float16 .npy."""
    FEAT.mkdir(parents=True, exist_ok=True)
    path = FEAT / f"birdvox_unit{unit}.npy"
    if path.exists():
        return np.load(path, mmap_mode="r")
    f = sf.SoundFile(BV / f"BirdVox-full-night_flac-audio_unit{unit}.flac")
    assert f.samplerate == SR, f.samplerate
    chunk, ctx = 300 * SR, int(fe.context_s * SR)
    parts, pos = [], 0
    while pos < f.frames:
        a = max(0, pos - ctx)
        f.seek(a)
        x = f.read(min(f.frames, pos + chunk + ctx) - a, dtype="float32")
        x = x.mean(1) if x.ndim > 1 else x
        m = fe(x).cpu().numpy()
        lo = int(round((pos - a) / 256))
        n = int(round(min(chunk, f.frames - pos) / 256))
        parts.append(m[:, lo:lo + n])
        pos += chunk
    feats = np.concatenate(parts, axis=1)
    np.save(path, feats)
    print(f"unit{unit}: {feats.shape[1] * FRAME_S / 3600:.1f} h of features", flush=True)
    return feats


def annotations(unit: str) -> np.ndarray:
    df = pd.read_csv(BV / f"BirdVox-full-night_csv-annotations_unit{unit}.csv")
    return np.sort(df.iloc[:, 0].astype(float).values)


def frame_targets(n_frames: int, calls_s: np.ndarray):
    """Per frame: 1 = call, 0 = no call, and a weight (0 = ignore near call edges)."""
    t = np.arange(n_frames) * FRAME_S
    y = np.zeros(n_frames, np.float32)
    w = np.ones(n_frames, np.float32)
    for c in calls_s:
        a, b = np.searchsorted(t, [c - IGNORE_HALF_S, c + IGNORE_HALF_S])
        w[a:b] = 0
    for c in calls_s:
        a, b = np.searchsorted(t, [c - POS_HALF_S, c + POS_HALF_S])
        y[a:b] = 1
        w[a:b] = 1
    return y, w


class CropSampler:
    def __init__(self, units, device):
        self.data = []
        for feats, calls in units:
            y, w = frame_targets(feats.shape[1], calls)
            self.data.append((torch.from_numpy(np.ascontiguousarray(feats)).to(device),
                              torch.from_numpy(y).to(device), torch.from_numpy(w).to(device),
                              (calls / FRAME_S).astype(int)))
        self.rng = np.random.default_rng(0)

    def _one(self):
        f, y, w, calls = self.data[self.rng.integers(len(self.data))]
        n = f.shape[1]
        if self.rng.random() < 0.5 and len(calls):
            c = calls[self.rng.integers(len(calls))]
            s = int(np.clip(c - self.rng.integers(16, CROP - 16), 0, n - CROP))
        else:
            s = int(self.rng.integers(0, n - CROP))
        return f[:, s:s + CROP].float(), y[s:s + CROP], w[s:s + CROP]

    def batch(self, size):
        xs, ys, ws = zip(*[self._one() for _ in range(size)])
        x, y, w = torch.stack(xs), torch.stack(ys), torch.stack(ws)
        # Augmentation: louder/quieter, small pitch shift, overlay a second crop.
        x = (x * torch.empty(size, 1, 1, device=x.device).uniform_(0.6, 1.3)).clamp(0, 1)
        for i in range(size):
            k = int(self.rng.integers(-3, 4))
            if k:
                x[i] = torch.roll(x[i], k, 0)
                (x[i, :k] if k > 0 else x[i, k:]).zero_()
        mix = torch.from_numpy(self.rng.random(size) < 0.3).to(x.device)
        if mix.any():
            x2, y2, w2 = (t[torch.randperm(size, device=x.device)] for t in (x, y, w))
            ym = torch.maximum(y, y2)
            wm = torch.maximum(torch.minimum(w, w2), ym)
            x = torch.where(mix[:, None, None], torch.maximum(x, x2), x)
            y = torch.where(mix[:, None], ym, y)
            w = torch.where(mix[:, None], wm, w)
        return x, y, w


def match_events(pred_peaks: np.ndarray, calls: np.ndarray, tol=MATCH_TOL_S):
    """Greedy one-to-one matching; returns number of true positives."""
    used = np.zeros(len(calls), bool)
    tp = 0
    for p in np.sort(pred_peaks):
        a, b = np.searchsorted(calls, [p - tol, p + tol])
        free = [j for j in range(a, b) if not used[j]]
        if free:
            j = min(free, key=lambda j: abs(calls[j] - p))
            used[j] = True
            tp += 1
    return tp


def event_metrics(probs, calls, thresholds=np.round(np.arange(0.1, 0.96, 0.05), 2)):
    rows = []
    for th in thresholds:
        ev = pick_events(probs, th)
        peaks = np.array([e[2] for e in ev])
        tp = match_events(peaks, calls)
        p = tp / max(1, len(peaks))
        r = tp / max(1, len(calls))
        rows.append(dict(threshold=float(th), n_pred=len(peaks), tp=tp, precision=p, recall=r,
                         f1=2 * p * r / max(1e-9, p + r)))
    return pd.DataFrame(rows)


def evaluate_night(model, feats, calls, device):
    probs = frame_probs(model, torch.from_numpy(np.ascontiguousarray(feats)).to(device))
    y, w = frame_targets(len(probs), calls)
    ap = average_precision_score(y[w > 0], probs[w > 0])
    return probs, ap


def main(epochs=20, steps=1500, batch=64, lr=2e-3):
    device = "cuda"
    torch.manual_seed(0)
    fe = SNRFrontend(device)
    units = {u: (unit_features(u, fe), annotations(u)) for s in SPLITS.values() for u in s}
    sampler = CropSampler([units[u] for u in SPLITS["train"]], device)
    model = FlightCallNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * steps)
    scaler = torch.amp.GradScaler()
    MODEL_DIR.mkdir(exist_ok=True)
    best, log = -1, []
    for ep in range(epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for _ in range(steps):
            x, y, w = sampler.batch(batch)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
            loss = (F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none") * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        model.eval()
        _, ap = evaluate_night(model, *units["07"], device)
        log.append(dict(epoch=ep + 1, loss=tot / steps, val_frame_ap=ap))
        print(f"epoch {ep + 1:2d} loss {tot / steps:.4f} val AP {ap:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if ap > best:
            best = ap
            torch.save({"state_dict": model.state_dict(), "hparams": {},
                        "val_frame_ap": ap, "epoch": ep + 1}, MODEL_DIR / "flightcall_cnn.pt")

    # Final: pick the threshold on validation night, report on the untouched test night.
    from flightcall_cnn import load_model
    model = load_model(MODEL_DIR / "flightcall_cnn.pt")
    out = OUTPUT_DIR / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for split, u in [("val", "07"), ("test", "10")]:
        probs, ap = evaluate_night(model, *units[u], device)
        m = event_metrics(probs, units[u][1])
        m.to_csv(out / f"cnn_birdvox_{split}_unit{u}_events.csv", index=False)
        res[split] = dict(unit=u, frame_ap=ap, n_calls=len(units[u][1]), events=m)
    th = float(res["val"]["events"].sort_values("f1").iloc[-1]["threshold"])
    test_row = res["test"]["events"].set_index("threshold").loc[th]
    summary = dict(chosen_threshold=th, val_frame_ap=res["val"]["frame_ap"],
                   test_frame_ap=res["test"]["frame_ap"], test_calls=res["test"]["n_calls"],
                   test_precision=test_row.precision, test_recall=test_row.recall,
                   test_f1=test_row.f1, training_log=log)
    (out / "cnn_birdvox_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "training_log"}, indent=2))


if __name__ == "__main__":
    main()
