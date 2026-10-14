"""Summaries of the detection table for the team.

Writes to outputs/:
  summary_by_night.csv   hours recorded, detections, clips, calls per night-hour
  species_counts.csv     detections per best label, split by night/twilight/day
  manual_clip_check.csv  for every hand-cut clip: did the pipeline find it, and as what?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import config as C
from audio_io import find_manual_clips, find_recordings


def main():
    df = pd.read_csv(C.OUTPUT_DIR / "detections.csv")
    recs = find_recordings(C.RECORDINGS_DIR)
    hours = pd.Series({n: sum(r.duration for r in recs if r.night == n) / 3600
                       for n in {r.night for r in recs}}, name="hours_recorded")

    by_night = df.groupby("night").agg(detections=("start", "size"), clips=("clip", "nunique"),
                                       night_detections=("period", lambda s: (s == "night").sum()))
    by_night = by_night.join(hours, how="outer").fillna(0)
    by_night["first_start"] = [min((r.start_time for r in recs if r.night == n)).strftime("%m-%d %H:%M")
                               for n in by_night.index]
    by_night.round(2).to_csv(C.OUTPUT_DIR / "summary_by_night.csv")

    sp = df.pivot_table(index="best_label", columns="period", values="start", aggfunc="size",
                        fill_value=0)
    sp["total"] = sp.sum(1)
    sp.sort_values("total", ascending=False).to_csv(C.OUTPUT_DIR / "species_counts.csv")

    rows = []
    for c in find_manual_clips(C.RECORDINGS_DIR):
        hit = df[(df.recording == c.source_stem) & (df.start < c.end_s + 0.5) & (df.end > c.start_s - 0.5)]
        rows.append(dict(clip=c.path.name, team_label=c.label or "", found=not hit.empty,
                         auto_clip="; ".join(sorted(set(hit.clip))),
                         auto_labels="; ".join(sorted(set(hit.best_label.astype(str))))))
    man = pd.DataFrame(rows)
    man.to_csv(C.OUTPUT_DIR / "manual_clip_check.csv", index=False)

    print(by_night.round(1).to_string())
    print(f"\nhand-cut clips found: {man.found.sum()} of {len(man)}")
    print(sp.sort_values("total", ascending=False).head(25).to_string())


if __name__ == "__main__":
    main()
