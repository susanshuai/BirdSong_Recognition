"""Stage 3: decide which stretches are birds, label them, and cut clips.

For every recording with cached scores (from scan.py):
  * candidates + referee probability (detect.py / models/stacker.pkl)
  * species guesses from Nighthawk, our species model, BirdNET and Perch
  * merge nearby detections into clips, cut them from the DENOISED audio, and
    save as <recording>-<start s>-<end s>_auto.WAV next to the original file
  * write one table of all detections (outputs/detections.csv) plus a copy
    per night folder (auto_detections.csv) so clips can be looked up by name

Original recordings are never modified.
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from datetime import timezone
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import soundfile as sf

import config as C
from audio_io import find_recordings
from detect import (BIRDNET_KEEP, birdnet_label, candidates, feature_matrix, load_cache,
                    nighthawk_labels, perch_label)


def sun_elevation(dt_local) -> float:
    from astral import Observer
    from astral.sun import elevation
    aware = dt_local.replace(tzinfo=ZoneInfo(C.TIMEZONE))
    return float(elevation(Observer(C.LATITUDE, C.LONGITUDE), aware.astimezone(timezone.utc)))


def period(elev: float) -> str:
    """night = sun more than 12 deg below the horizon (birds rarely sing yet);
    twilight = dawn/dusk (dawn chorus starts here); day = sun up."""
    return "night" if elev < -12 else ("twilight" if elev < 0 else "day")


def split_long(ev: pd.DataFrame) -> pd.DataFrame:
    """Cut detections longer than a clip may be (continuous daytime song) into pieces."""
    max_len = C.CLIP_MAX_S - 2 * C.CLIP_PAD_S - 1
    rows = []
    for _, r in ev.iterrows():
        n = max(1, math.ceil((r.end - r.start) / max_len))
        edges = np.linspace(r.start, r.end, n + 1)
        for a, b in zip(edges[:-1], edges[1:]):
            piece = r.copy()
            piece["start"], piece["end"] = a, b
            rows.append(piece)
    return pd.DataFrame(rows).reset_index(drop=True)


def group_into_clips(ev: pd.DataFrame, duration: float) -> list[tuple[int, int, list[int]]]:
    """Merge detections closer than CLIP_MERGE_GAP_S, never beyond CLIP_MAX_S.
    Clip edges are whole seconds (like the hand-cut clips) inside the recording."""
    clips = []
    for i, r in ev.sort_values("start").iterrows():
        a, b = max(0.0, r.start - C.CLIP_PAD_S), min(duration, r.end + C.CLIP_PAD_S)
        if clips and a <= clips[-1][1] + C.CLIP_MERGE_GAP_S and b - clips[-1][0] <= C.CLIP_MAX_S:
            clips[-1][1] = max(clips[-1][1], b)
            clips[-1][2].append(i)
        else:
            clips.append([a, b, [i]])
    last = int(math.floor(duration))
    return [(int(math.floor(a)), min(int(math.ceil(b)), last), idx) for a, b, idx in clips]


def remove_old_clips(rec):
    """Delete this tool's earlier clips for one recording (only *_auto.WAV files)."""
    for p in rec.path.parent.glob(f"{rec.stem}-*_auto.WAV"):
        if p.name.endswith("_auto.WAV") and p.name.startswith(rec.stem + "-"):
            p.unlink()


class Labeler:
    def __init__(self):
        from denoise import Denoiser
        from pretrained import Nighthawk
        from species_cnn import CropFrontend, load_model
        self.nh = Nighthawk()
        self.sp_model, self.sp_classes = load_model(C.MODEL_DIR / "species_cnn.pt")
        self.sp_fe = CropFrontend()
        self.denoiser = Denoiser()
        with open(C.MODEL_DIR / "stacker.pkl", "rb") as fh:
            self.stacker = pickle.load(fh)

    def species_cnn(self, rec, peaks_s):
        from pretrained import resample
        from species_cnn import CODE_NAMES, CROP_S, classify
        if not len(peaks_s):
            return [], []
        crops = []
        for p in peaks_s:
            x = rec.read(max(0.0, p - CROP_S / 2), CROP_S)
            x = resample(x, rec.sample_rate, 24000)
            crops.append(np.pad(x, (0, max(0, int(CROP_S * 24000) - len(x))))[: int(CROP_S * 24000)])
        prob = classify(self.sp_model, self.sp_fe, np.stack(crops))
        k = prob.argmax(1)
        return [CODE_NAMES[self.sp_classes[i]] for i in k], prob[np.arange(len(k)), k].tolist()

    def events(self, rec, cache) -> pd.DataFrame:
        cand = candidates(cache)
        if cand.empty:
            return cand
        st = self.stacker
        cand["bird_prob"] = st["model"].predict_proba(feature_matrix(cand))[:, 1]
        keep = (cand.bird_prob >= st["threshold"]) | (cand.bn_local >= BIRDNET_KEEP)
        ev = cand[keep].reset_index(drop=True)
        if ev.empty:
            return ev
        nh_det = self.nh.detections([cache[f"nh{k}"].astype(np.float32) for k in range(4)])
        lab = {k: [] for k in ["nighthawk", "nighthawk_prob", "birdnet", "birdnet_score",
                               "perch", "perch_logit", "perch_top_any"]}
        for _, r in ev.iterrows():
            n, p = nighthawk_labels(nh_det, r.start, r.end)
            lab["nighthawk"].append(n); lab["nighthawk_prob"].append(p)
            n, p = birdnet_label(cache, r.start, r.end)
            lab["birdnet"].append(n); lab["birdnet_score"].append(p)
            n, p, top = perch_label(cache, r.start, r.end)
            lab["perch"].append(n); lab["perch_logit"].append(p); lab["perch_top_any"].append(top)
        for k, v in lab.items():
            ev[k] = v
        ev["our_species_model"], ev["our_species_prob"] = self.species_cnn(rec, ev.cnn_peak_s.values)
        ev["sun_elevation"] = [sun_elevation(rec.time_at(t)) for t in ev.start]
        ev["period"] = ev.sun_elevation.map(period)
        best = [best_label(r) for _, r in ev.iterrows()]
        ev["best_label"] = [b[0] for b in best]
        ev["label_support"] = [b[1] for b in best]
        ev["models_agreeing"] = [b[2] for b in best]
        return ev


NOT_SPECIES = ("Not a bird", "Zeep", "Short double")
FLIGHT_CALL_WEIGHT = {"night": 1.0, "twilight": 0.5, "day": 0.3}
MIN_SUPPORT = 0.5


def label_votes(r) -> dict:
    """Each model votes for its species, weighted by its confidence.

    Flight-call specialists (Nighthawk, our species model) count fully in the
    dark but less at dawn/dusk and in daylight, when most sounds are songs.
    """
    votes = {}
    w_fc = FLIGHT_CALL_WEIGHT[r.period]

    def add(name, w):
        if isinstance(name, str) and name and w > 0:
            votes[name] = votes.get(name, 0.0) + float(w)

    if isinstance(r.nighthawk, str) and "species unknown" not in r.nighthawk:
        add(r.nighthawk, w_fc * r.nighthawk_prob)
    if not str(r.our_species_model).startswith(NOT_SPECIES):
        add(r.our_species_model, w_fc * 0.8 * r.our_species_prob)
    add(r.birdnet, min(1.0, 2 * r.birdnet_score))
    if isinstance(r.perch_logit, float) and np.isfinite(r.perch_logit):
        add(r.perch, 1 / (1 + np.exp(-1.5 * (r.perch_logit - 10))))
    return votes


def best_label(r) -> tuple[str, float, int]:
    """(label, support, number of models agreeing)."""
    votes = label_votes(r)
    if votes:
        name, support = max(votes.items(), key=lambda kv: kv[1])
        if support >= MIN_SUPPORT:
            n = sum(1 for m in (r.nighthawk, r.our_species_model, r.birdnet, r.perch) if m == name)
            return name, support, n
    if isinstance(r.nighthawk, str) and r.nighthawk:
        return r.nighthawk, r.nighthawk_prob, 1
    return "Unknown bird", 0.0, 0


def write_clip(rec, denoiser, a, b, path: Path):
    """Denoise [a, b] (with context, each channel) and save 16-bit, peak-normalized."""
    ctx = denoiser.context_s
    a0 = max(0.0, a - ctx)
    raw = rec.read(a0, (b - a0) + ctx, mono=False)
    lo = int(round((a - a0) * rec.sample_rate))
    n = int(round((b - a) * rec.sample_rate))
    chans = [denoiser(raw[:, ch])[lo:lo + n] for ch in range(raw.shape[1])]
    y = np.stack(chans, 1)
    peak = np.abs(y).max()
    if peak > 0:
        y *= 10 ** (C.CLIP_PEAK_DBFS / 20) / peak
    sf.write(path, y if y.shape[1] > 1 else y[:, 0], rec.sample_rate, subtype="PCM_16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-clips", action="store_true", help="only write the detection table")
    ap.add_argument("--only", nargs="*", help="recording names to process (default: all)")
    args = ap.parse_args()

    lab = Labeler()
    all_rows = []
    for rec in find_recordings(C.RECORDINGS_DIR):
        if args.only and rec.stem not in args.only:
            continue
        cpath = C.CACHE_DIR / rec.night / f"{rec.stem}.npz"
        if not cpath.exists():
            print(f"skip {rec.stem}: not scanned yet")
            continue
        if not args.no_clips:
            remove_old_clips(rec)
        ev = lab.events(rec, load_cache(cpath))
        if ev.empty:
            print(f"{rec.stem}: no detections")
            continue
        ev = split_long(ev)
        ev.insert(0, "recording", rec.stem)
        ev.insert(1, "night", rec.night)
        ev["clock_time"] = [rec.time_at(t).strftime("%Y-%m-%d %H:%M:%S") for t in ev.start]
        ev["clip_file"] = ""
        for a, b, idx in group_into_clips(ev, rec.duration):
            name = C.CLIP_NAME.format(stem=rec.stem, start=a, end=b)
            ev.loc[idx, "clip_file"] = name
            if not args.no_clips:
                write_clip(rec, lab.denoiser, a, b, rec.path.parent / name)
        print(f"{rec.stem}: {len(ev)} detections in {ev['clip_file'].nunique()} clips", flush=True)
        all_rows.append(ev)

    df = pd.concat(all_rows, ignore_index=True)
    cols = ["night", "recording", "clip_file", "clock_time", "period", "start", "end", "best_label",
            "label_support", "models_agreeing", "bird_prob", "sources", "nighthawk", "nighthawk_prob", "our_species_model",
            "our_species_prob", "birdnet", "birdnet_score", "perch", "perch_logit", "perch_top_any",
            "cnn_max", "nh_max", "bn_local", "pc_mass", "bn_noise", "sun_elevation"]
    df = df[cols].round(3)
    C.OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(C.OUTPUT_DIR / "detections.csv", index=False)
    if not args.no_clips:
        for night, d in df.groupby("night"):
            d.to_csv(C.RECORDINGS_DIR / night / "auto_detections.csv", index=False)
    print(f"total: {len(df)} detections, {df['clip_file'].nunique()} clips")


if __name__ == "__main__":
    main()
