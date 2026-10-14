"""Train the "referee" that combines all models, and grade every model.

Fit night:  BirdVox unit 07 (the CNN only used it to pick its best epoch)
Test night: BirdVox unit 10 (never used for any training or tuning)

Report card metrics, all on the test night:
  recall    = share of the experts' marked calls that were found
  precision = share of the model's detections that really contain a call
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import cross_val_predict

from config import CACHE_DIR, MODEL_DIR, OUTPUT_DIR
from detect import FEATURES, candidates, feature_matrix, load_cache, sig
from flightcall_cnn import pick_events
from train_flightcall_cnn import annotations

TOL = 0.1   # a span "contains" a call if the call center is within 0.1 s of it


def label_spans(starts, ends, calls):
    """For each span: does it contain a call? And which calls are covered."""
    y = np.zeros(len(starts), bool)
    covered = np.zeros(len(calls), bool)
    for i, (a, b) in enumerate(zip(starts, ends)):
        lo, hi = np.searchsorted(calls, [a - TOL, b + TOL])
        if hi > lo:
            y[i] = True
            covered[lo:hi] = True
    return y, covered


def pr_at(starts, ends, keep, calls):
    y, cov = label_spans(starts[keep], ends[keep], calls)
    p = y.mean() if len(y) else 0.0
    r = cov.mean()
    return dict(n_detections=int(keep.sum()), precision=p, recall=r,
                f1=2 * p * r / (p + r) if p + r > 0 else 0.0,
                mean_span_s=float((ends[keep] - starts[keep]).mean()) if keep.any() else 0.0)


def model_report(c, calls):
    rows = []
    fs = float(c["cnn_frame_s"])
    for th in [0.3, 0.5, 0.7, 0.9]:
        ev = pick_events(c["cnn"].astype(np.float32), th)
        s = np.array([e[0] for e in ev]); e_ = np.array([e[1] for e in ev])
        rows.append(dict(model="Our CNN detector", setting=f"p>={th}", **pr_at(s, e_, np.ones(len(s), bool), calls)))
    nh_all = np.max([sig(c[f"nh{k}"]).max(1) for k in range(4)], axis=0)
    for th in [0.5, 0.8, 0.95]:
        rows.append(dict(model="Nighthawk (1-s windows)", setting=f"p>={th}",
                         **pr_at(c["nh_t"], c["nh_t"] + 1, nh_all >= th, calls)))
    for th in [0.1, 0.3, 0.5, 0.7]:
        rows.append(dict(model="BirdNET (3-s windows, local species)", setting=f"p>={th}",
                         **pr_at(c["bn_t"], c["bn_t"] + 3, c["bn_local"].astype(float) >= th, calls)))
    for th in [0.3, 0.5, 0.8]:
        rows.append(dict(model="Perch v2 (5-s windows)", setting=f"bird share>={th}",
                         **pr_at(c["pc_t"], c["pc_t"] + 5, c["pc_bird_mass"].astype(float) >= th, calls)))
    return rows


def nighthawk_official(c, calls):
    from pretrained import Nighthawk
    nh = Nighthawk()
    det = nh.detections([c[f"nh{k}"].astype(np.float32) for k in range(4)])
    s, e = det.start_sec.to_numpy(float), det.end_sec.to_numpy(float)
    return dict(model="Nighthawk (official settings)", setting="threshold 80 + its post-processing",
                **pr_at(s, e, np.ones(len(s), bool), calls))


def main():
    fit_c, test_c = (load_cache(CACHE_DIR / "birdvox" / f"birdvox_unit{u}.npz") for u in ("07", "10"))
    fit_calls, test_calls = annotations("07"), annotations("10")
    fit, test = candidates(fit_c), candidates(test_c)
    fit["y"], _ = label_spans(fit.start.values, fit.end.values, fit_calls)
    test["y"], _ = label_spans(test.start.values, test.end.values, test_calls)
    Xf, Xt = feature_matrix(fit), feature_matrix(test)
    print(f"candidates: fit {len(fit)} ({fit.y.mean():.0%} contain a call), "
          f"test {len(test)} ({test.y.mean():.0%})")
    _, cov = label_spans(test.start.values, test.end.values, test_calls)
    print(f"candidate recall on test night (ceiling for the referee): {cov.mean():.3f}")

    models = {"logistic": LogisticRegression(C=1.0, max_iter=2000),
              "boosted_trees": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                                              max_leaf_nodes=15)}
    scores = {}
    for name, m in models.items():
        cv = cross_val_predict(m, Xf, fit.y, cv=5, method="predict_proba")[:, 1]
        m.fit(Xf, fit.y)
        pt = m.predict_proba(Xt)[:, 1]
        scores[name] = dict(cv=cv, test=pt, auc=roc_auc_score(test.y, pt),
                            ap=average_precision_score(test.y, pt))
        print(f"{name}: test AUC {scores[name]['auc']:.3f}  AP {scores[name]['ap']:.3f}")
    # Prefer the simpler model unless trees are clearly better (they transfer worse).
    best = "boosted_trees" if scores["boosted_trees"]["ap"] > scores["logistic"]["ap"] + 0.03 else "logistic"
    m = models[best]
    ths = np.round(np.arange(0.1, 0.91, 0.05), 2)
    f1s = []
    for th in ths:
        keep = scores[best]["cv"] >= th
        r = pr_at(fit.start.values, fit.end.values, keep, fit_calls)
        f1s.append(r["f1"])
    th = float(ths[int(np.argmax(f1s))])

    rows = model_report(test_c, test_calls)
    rows.append(nighthawk_official(test_c, test_calls))
    for t in sorted({0.3, th, 0.7}):
        keep = scores[best]["test"] >= t
        rows.append(dict(model=f"Combined referee ({best})", setting=f"p>={t}" + (" (chosen)" if t == th else ""),
                         **pr_at(test.start.values, test.end.values, keep, test_calls)))
    report = pd.DataFrame(rows)
    out = OUTPUT_DIR / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    report.to_csv(out / "report_card_birdvox_test.csv", index=False)
    print(report.round(3).to_string(index=False))

    coefs = dict(zip(FEATURES, m.coef_[0].round(3).tolist())) if best == "logistic" else None
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / "stacker.pkl", "wb") as fh:
        pickle.dump(dict(model=m, kind=best, threshold=th, features=FEATURES), fh)
    (out / "stacker_summary.json").write_text(json.dumps(dict(
        kind=best, threshold=th, coefficients=coefs, n_test_calls=int(len(test_calls)),
        **{f"{k}_test_auc": v["auc"] for k, v in scores.items()},
        **{f"{k}_test_ap": v["ap"] for k, v in scores.items()}), indent=2))
    print("referee:", best, "threshold", th, coefs)


if __name__ == "__main__":
    main()
