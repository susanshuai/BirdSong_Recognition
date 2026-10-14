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
    return "night" if elev < -6 else ("twilight" if elev < 0 else "day")


def group_into_clips(ev: pd.DataFrame, duration: float) -> list[tuple[int, int, list[int]]]:
    """Merge detections closer than CLIP_MERGE_GAP_S; split bouts longer than CLIP_MAX_S."""
    clips = []
    for i, r in ev.sort_values("start").iterrows():
        a, b = max(0.0, r.start - C.CLIP_PAD_S), min(duration, r.end + C.CLIP_PAD_S)
        if clips and a <= clips[-1][1] + C.CLIP_MERGE_GAP_S and b - clips[-1][0] <= C.CLIP_MAX_S:
            clips[-1][1] = max(clips[-1][1], b)
            clips[-1][2].append(i)
        else:
            clips.append([a, b, [i]])
    return [(int(math.floor(a)), int(math.ceil(b)), idx) for a, b, idx in clips]


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
        ev["best_label"] = [best_label(r) for _, r in ev.iterrows()]
        return ev


def best_label(r) -> str:
    """One readable label. Night: flight-call models first. Day: song models first."""
    sp_ok = r.our_species_prob >= 0.6 and not str(r.our_species_model).startswith(("Not a bird", "Zeep", "Short"))
    nh_species = r.nighthawk and "species unknown" not in r.nighthawk
    if r.period != "day":
        if nh_species and r.nighthawk_prob >= 0.5:
            return r.nighthawk
        if sp_ok:
            return r.our_species_model
        if r.birdnet and r.birdnet_score >= 0.5:
            return r.birdnet
        if r.nighthawk:
            return r.nighthawk
    else:
        if r.birdnet and r.birdnet_score >= 0.5:
            return r.birdnet
        if r.perch and r.perch_logit >= 5:
            return r.perch
        if nh_species:
            return r.nighthawk
        if sp_ok:
            return r.our_species_model
    return "Unknown bird"


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
        ev = lab.events(rec, load_cache(cpath))
        if ev.empty:
            print(f"{rec.stem}: no detections")
            continue
        ev.insert(0, "recording", rec.stem)
        ev.insert(1, "night", rec.night)
        ev["clock_time"] = [rec.time_at(t).strftime("%Y-%m-%d %H:%M:%S") for t in ev.start]
        ev["clip"] = ""
        for a, b, idx in group_into_clips(ev, rec.duration):
            name = C.CLIP_NAME.format(stem=rec.stem, start=a, end=b)
            ev.loc[idx, "clip"] = name
            if not args.no_clips:
                write_clip(rec, lab.denoiser, a, b, rec.path.parent / name)
        print(f"{rec.stem}: {len(ev)} detections in {ev.clip.nunique()} clips", flush=True)
        all_rows.append(ev)

    df = pd.concat(all_rows, ignore_index=True)
    cols = ["night", "recording", "clip", "clock_time", "period", "start", "end", "best_label",
            "bird_prob", "sources", "nighthawk", "nighthawk_prob", "our_species_model",
            "our_species_prob", "birdnet", "birdnet_score", "perch", "perch_logit", "perch_top_any",
            "cnn_max", "nh_max", "bn_local", "pc_mass", "bn_noise", "sun_elevation"]
    df = df[cols].round(3)
    C.OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(C.OUTPUT_DIR / "detections.csv", index=False)
    for night, d in df.groupby("night"):
        d.to_csv(C.RECORDINGS_DIR / night / "auto_detections.csv", index=False)
    print(f"total: {len(df)} detections, {df.clip.nunique()} clips")


if __name__ == "__main__":
    main()
