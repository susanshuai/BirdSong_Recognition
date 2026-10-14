"""Experiment 2: our flight-call detector on the team's hand-cut clips.

Uses exactly the same 34 positive spans and 400 random background spans as
experiment 1, so the numbers sit side by side with BirdNET / Perch / Nighthawk.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from audio_io import find_recordings
from config import MODEL_DIR, OUTPUT_DIR, RECORDINGS_DIR
from flightcall_cnn import FRAME_S, SNRFrontend, frame_probs, load_model
from pretrained import resample


def main():
    out = OUTPUT_DIR / "experiments"
    spans = pd.read_csv(out / "exp1_scores.csv")
    spans = spans[spans.audio == "raw"].reset_index(drop=True)
    recs = {r.stem: r for r in find_recordings(RECORDINGS_DIR)}
    model, fe = load_model(MODEL_DIR / "flightcall_cnn.pt"), SNRFrontend()
    scores = []
    for _, s in spans.iterrows():
        r = recs[s.stem]
        a0 = max(0.0, s.start - fe.context_s)
        x = resample(r.read(a0, (s.end - a0) + fe.context_s), r.sample_rate, 24000)
        p = frame_probs(model, fe(x))
        i0, i1 = int((s.start - a0) / FRAME_S), int(np.ceil((s.end - a0) / FRAME_S))
        scores.append(float(p[i0:i1].max()))
    spans["our_cnn"] = scores
    rows = []
    for m in ["birdnet", "birdnet_local", "perch_logit", "perch_softmax", "nighthawk", "our_cnn"]:
        pos, neg = spans[spans.label == 1][m], spans[spans.label == 0][m]
        rows.append(dict(model=m, auc=roc_auc_score(spans.label, spans[m]),
                         recall_at_5pct=(pos > np.quantile(neg, 0.95)).mean(),
                         recall_at_1pct=(pos > np.quantile(neg, 0.99)).mean()))
    res = pd.DataFrame(rows)
    res.to_csv(out / "exp2_summary_raw_audio.csv", index=False)
    spans.to_csv(out / "exp2_scores.csv", index=False)
    print(res.round(3).to_string(index=False))
    print(spans[spans.label == 1][["stem", "start", "end", "nighthawk", "our_cnn"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
