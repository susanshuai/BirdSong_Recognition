"""Draw spectrograms of random detections on one sheet for a quick visual check.

  python src/review_sheet.py --n 48 --seed 1          # random detections
  python src/review_sheet.py --label "Savannah Sparrow"
Saves outputs/review/review_<seed>.png and a matching CSV of what was drawn.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import spectrogram

import config as C
from audio_io import find_recordings
from denoise import Denoiser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--label", default=None)
    ap.add_argument("--period", default=None, help="night / twilight / day")
    args = ap.parse_args()

    df = pd.read_csv(C.OUTPUT_DIR / "detections.csv")
    if args.label:
        df = df[df.best_label.str.contains(args.label, regex=False)]
    if args.period:
        df = df[df.period == args.period]
    df = df.sample(min(args.n, len(df)), random_state=args.seed).sort_values(["recording", "start"])
    recs = {r.stem: r for r in find_recordings(C.RECORDINGS_DIR)}
    dn = Denoiser()
    cols = 6
    rows = int(np.ceil(len(df) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.4))
    for k, (ax, (_, r)) in enumerate(zip(np.ravel(axs), df.iterrows())):
        rec = recs[r.recording]
        mid = (r.start + r.end) / 2
        a = max(0.0, mid - 1.5 - dn.context_s)
        x = dn(rec.read(a, 3.0 + 2 * dn.context_s))
        n0 = int((mid - 1.5 - a) * rec.sample_rate)
        y = x[n0:n0 + 3 * rec.sample_rate]
        f, t, S = spectrogram(y, rec.sample_rate, nperseg=1024, noverlap=768)
        L = 10 * np.log10(S + 1e-14)
        ax.pcolormesh(t - 1.5, f / 1000, L, shading="auto", cmap="magma",
                      vmin=L.max() - 50, vmax=L.max())
        ax.axvspan(r.start - mid, r.end - mid, color="c", alpha=0.12)
        ax.set_ylim(0, 12)
        ax.set_title(f"#{k} {str(r.best_label)[:28]}\np={r.bird_prob:.2f} {r.period} {r.clock_time[11:16]}",
                     fontsize=7)
        ax.tick_params(labelsize=5)
    for ax in np.ravel(axs)[len(df):]:
        ax.axis("off")
    out = C.OUTPUT_DIR / "review"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.seed}" + (f"_{args.label}" if args.label else "") + (f"_{args.period}" if args.period else "")
    plt.tight_layout()
    plt.savefig(out / f"review_{tag}.png", dpi=70)
    df.assign(sheet_index=range(len(df))).to_csv(out / f"review_{tag}.csv", index=False)
    print(out / f"review_{tag}.png")


if __name__ == "__main__":
    main()
