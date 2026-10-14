"""Experiment 1: does denoising help the pretrained models on OUR recordings?

Positives: the 34 hand-cut clips (time spans the team marked as birds).
Background: 400 random 2-s spans from the same recordings (mostly no bird,
though a few may contain unnoticed calls, which makes the numbers slightly
pessimistic).

For every span and every model we take the highest "bird" score of any window
overlapping the span, once on raw audio and once on denoised audio.
Reported: ROC-AUC (chance of ranking a real call above a random span) and
recall when the threshold is set so 5% / 1% of random spans would be flagged.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from audio_io import find_manual_clips, find_recordings
from config import OUTPUT_DIR, RECORDINGS_DIR
from denoise import Denoiser
from pretrained import BirdNET, Nighthawk, Perch, resample


def main():
    recs = {r.stem: r for r in find_recordings(RECORDINGS_DIR)}
    clips = find_manual_clips(RECORDINGS_DIR)
    rng = np.random.default_rng(42)
    spans = [(c.source_stem, c.start_s, c.end_s, 1) for c in clips]
    stems = sorted({c.source_stem for c in clips})
    for _ in range(400):
        r = recs[rng.choice(stems)]
        a = rng.uniform(10, r.duration - 12)
        spans.append((r.stem, a, a + 2, 0))

    dn = Denoiser()
    bn, pc, nh = BirdNET(), Perch(), Nighthawk()
    bn_bird = np.array([c not in bn.noise_classes for c in bn.common])
    bn_expected = bn.expected_species() & bn_bird
    pc_bird = np.array([e != "no_ebird_code" for e in pc.ebird])
    sig = lambda z: 1 / (1 + np.exp(-z.astype(np.float32)))

    rows = []
    for i, (stem, a, b, y) in enumerate(spans):
        r = recs[stem]
        pad = 6.0 + dn.context_s
        a0 = max(0.0, a - pad)
        raw = r.read(a0, (b - a0) + pad)
        n_ctx = int(dn.context_s * r.sample_rate)
        versions = {"raw": raw, "denoised": dn(raw)}
        for name, x in versions.items():
            x = x[n_ctx:len(x) - n_ctx]
            t0 = a0 + dn.context_s
            row = dict(stem=stem, start=a, end=b, label=y, audio=name)
            st, lg = bn.logits(x, hop_s=0.5)
            ov = (t0 + st < b) & (t0 + st + 3 > a)
            p = sig(lg[ov])
            row["birdnet"] = p[:, bn_bird].max()
            row["birdnet_local"] = p[:, bn_expected].max()
            st, lg, _ = pc.run(resample(x, r.sample_rate, 32000), hop_s=1.0)
            ov = (t0 + st < b) & (t0 + st + 5 > a)
            l = lg[ov].astype(np.float32)
            row["perch_logit"] = l[:, pc_bird].max()
            sm = np.exp(l - l.max(1, keepdims=True)); sm /= sm.sum(1, keepdims=True)
            row["perch_softmax"] = sm[:, pc_bird].sum(1).max()
            st, outs = nh.logits(resample(x, r.sample_rate, 22050))
            ov = (t0 + st < b) & (t0 + st + 1 > a)
            row["nighthawk"] = max(sig(o[ov]).max() for o in outs)
            rows.append(row)
        if i % 50 == 0:
            print(f"{i}/{len(spans)}", flush=True)

    df = pd.DataFrame(rows)
    out = OUTPUT_DIR / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "exp1_scores.csv", index=False)

    models = ["birdnet", "birdnet_local", "perch_logit", "perch_softmax", "nighthawk"]
    summary = []
    for audio in ["raw", "denoised"]:
        d = df[df.audio == audio]
        for m in models:
            pos, neg = d[d.label == 1][m], d[d.label == 0][m]
            summary.append(dict(
                audio=audio, model=m, auc=roc_auc_score(d.label, d[m]),
                recall_at_5pct=(pos > np.quantile(neg, 0.95)).mean(),
                recall_at_1pct=(pos > np.quantile(neg, 0.99)).mean()))
    s = pd.DataFrame(summary)
    s.to_csv(out / "exp1_summary.csv", index=False)
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
