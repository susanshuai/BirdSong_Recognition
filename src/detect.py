"""Stage 2: combine the models' scores into bird events.

1. Candidates: any stretch where at least one model is mildly interested
   (low thresholds, so few real calls are missed).
2. Evidence: for each candidate, every model's strongest score inside it.
3. Referee ("stacker"): a logistic regression trained on a fully labelled
   BirdVox night turns the evidence into one probability that the candidate
   holds a bird call. BirdNET's confident species detections are always kept,
   because BirdVox only labels night flight calls, not songs.
4. Species: each model's best guess is kept in its own column, and one
   "best_label" is chosen (night: Nighthawk > our species model > BirdNET;
   day: BirdNET > Perch > others).
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

# Candidate thresholds (deliberately low: high recall).
CAND = dict(cnn=0.3, nh=0.5, bn=0.3, pc=0.5)
MERGE_GAP_S = 0.3
BIRDNET_KEEP = 0.7          # BirdNET local-species score that is always trusted
FEATURES = ["cnn_max", "cnn_mean_top", "nh_max", "nh_order", "bn_bird", "bn_local",
            "pc_mass", "pc_logit", "bn_noise", "log_dur"]


def load_cache(path) -> dict:
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def _spans_from_mask(t, on, width):
    """Windows (start times t, fixed width) that are 'on' -> merged spans."""
    spans = []
    for s in t[on]:
        if spans and s <= spans[-1][1] + MERGE_GAP_S:
            spans[-1][1] = max(spans[-1][1], s + width)
        else:
            spans.append([s, s + width])
    return spans


def _merge(spans):
    spans = sorted(spans)
    out = []
    for a, b, src in spans:
        if out and a <= out[-1][1] + MERGE_GAP_S:
            out[-1][1] = max(out[-1][1], b)
            out[-1][2] |= src
        else:
            out.append([a, b, set(src)])
    return out


def sig(z):
    return 1 / (1 + np.exp(-np.asarray(z, np.float32)))


def candidates(c: dict) -> pd.DataFrame:
    """Candidate spans + evidence features for one cached recording."""
    from flightcall_cnn import pick_events
    fs = float(c["cnn_frame_s"])
    cnn = c["cnn"].astype(np.float32)
    spans = [(s, e, {"cnn"}) for s, e, _, _ in pick_events(cnn, CAND["cnn"], min_frames=2)]
    nh_all = np.max([sig(c[f"nh{k}"]).max(1) for k in range(4)], axis=0)
    spans += [(a, b, {"nh"}) for a, b in _spans_from_mask(c["nh_t"], nh_all >= CAND["nh"], 1.0)]
    bn = c["bn_local"].astype(np.float32)
    spans += [(a, b, {"bn"}) for a, b in _spans_from_mask(c["bn_t"], bn >= CAND["bn"], 3.0)]
    pc = c["pc_bird_mass"].astype(np.float32)
    spans += [(a, b, {"pc"}) for a, b in _spans_from_mask(c["pc_t"], pc >= CAND["pc"], 5.0)]
    merged = _merge(spans)

    rows = []
    nh_order = sig(c["nh0"]).max(1)
    for a, b, src in merged:
        b = min(b, float(c["duration"]))
        fa, fb = int(a / fs), max(int(a / fs) + 1, int(np.ceil(b / fs)))
        seg = cnn[fa:fb]
        nh_m = (c["nh_t"] < b) & (c["nh_t"] + 1.0 > a)
        bn_m = (c["bn_t"] < b) & (c["bn_t"] + 3.0 > a)
        pc_m = (c["pc_t"] < b) & (c["pc_t"] + 5.0 > a)
        mx = lambda arr, m: float(arr[m].max()) if m.any() else 0.0
        rows.append(dict(
            start=a, end=b, sources="+".join(sorted(src)),
            cnn_max=float(seg.max()) if len(seg) else 0.0,
            cnn_mean_top=float(np.sort(seg)[-5:].mean()) if len(seg) else 0.0,
            cnn_peak_s=(fa + int(np.argmax(seg))) * fs if len(seg) else (a + b) / 2,
            nh_max=mx(nh_all, nh_m), nh_order=mx(nh_order, nh_m),
            bn_bird=mx(c["bn_bird"].astype(np.float32), bn_m), bn_local=mx(bn, bn_m),
            bn_noise=mx(c["bn_noise"].astype(np.float32).max(1), bn_m),
            pc_mass=mx(pc, pc_m), pc_logit=mx(c["pc_bird_logit"].astype(np.float32), pc_m),
            log_dur=float(np.log(max(b - a, 0.05)))))
    return pd.DataFrame(rows)


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    eps = 1e-4
    X = []
    for f in FEATURES:
        v = df[f].to_numpy(np.float64)
        if f in ("log_dur", "pc_logit"):
            X.append(v)
        else:                        # probabilities -> log-odds, the natural scale for LR
            v = np.clip(v, eps, 1 - eps)
            X.append(np.log(v / (1 - v)))
    return np.stack(X, 1)


# ------------------------------------------------------------- labelling
@lru_cache(maxsize=1)
def _taxonomy():
    from pretrained import BirdNET, Perch
    bn, pc = BirdNET(threads=1), Perch.__new__(Perch)
    import birdnet_analyzer.config as bcfg
    assets = os.path.join(bcfg.PERCH_V2_MODEL_PATH, "assets")
    pc_sci = pd.read_csv(os.path.join(assets, "labels.csv")).iloc[:, 0].tolist()
    pc_ebird = pd.read_csv(os.path.join(assets, "perch_v2_ebird_classes.csv")).iloc[:, 0].tolist()
    sci2common = dict(zip(bn.sci, bn.common))
    bn_local = bn.expected_species()
    return bn.common, bn.sci, bn_local, pc_sci, pc_ebird, sci2common


def birdnet_label(c, a, b):
    common, sci, local, *_ = _taxonomy()
    m = (c["bn_t"] < b) & (c["bn_t"] + 3.0 > a)
    best, score = None, 0.0
    for idx_row, val_row in zip(c["bn_idx"][m], c["bn_val"][m]):
        for i, v in zip(idx_row, val_row):
            if local[i] and float(v) > score:
                best, score = common[i], float(v)
                break
    return best, score


def perch_label(c, a, b):
    *_, pc_sci, pc_ebird, sci2common = _taxonomy()
    m = (c["pc_t"] < b) & (c["pc_t"] + 5.0 > a)
    best, score, top_any = None, -1e9, None
    for idx_row, val_row in zip(c["pc_idx"][m], c["pc_val"][m]):
        if top_any is None:
            top_any = pc_sci[idx_row[0]]
        for i, v in zip(idx_row, val_row):
            if pc_ebird[i] != "no_ebird_code" and float(v) > score:
                best, score = sci2common.get(pc_sci[i], pc_sci[i]), float(v)
                break
    return best, score, top_any


def nighthawk_labels(det: pd.DataFrame, a, b):
    """Most specific Nighthawk taxon overlapping [a, b] from its own detection table."""
    if det is None or det.empty:
        return None, 0.0
    m = det[(det.start_sec < b) & (det.end_sec > a)]
    best, prob, rank = None, 0.0, -1
    for _, r in m.iterrows():
        for k, lvl in enumerate(["order", "family", "group", "species"]):
            if lvl in r and isinstance(r[lvl], str) and r[lvl]:
                p = float(r.get(f"prob_{lvl}", 0) or 0)
                if (k, p) > (rank, prob):
                    best, prob, rank = (r[lvl], lvl), p, k
    if best is None:
        return None, 0.0
    return _nh_readable(*best), prob


@lru_cache(maxsize=1)
def _nh_names():
    from nighthawk import detector
    tax = detector._TAXONOMY_DIR_PATH
    t = pd.read_csv(tax / "ebird_taxonomy.csv").set_index("code")["name"].to_dict()
    g = pd.read_csv(tax / "groups_ebird_codes.csv")
    groups = {k: " / ".join(t.get(c, c) for c in d.ebird_code) for k, d in g.groupby("group")}
    return t, groups


def _nh_readable(name: str, level: str) -> str:
    species, groups = _nh_names()
    if level == "species":
        return species.get(name, name)
    if level == "group":
        return f"{name} group: {groups.get(name, '?')} (species unknown)"
    return f"{name} ({level}, species unknown)"
