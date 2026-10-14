"""Build the kids' team page (one self-contained HTML file, audio embedded).

  python src/build_team_page.py
Writes outputs/team_page/night_sky_listener.html. The page template lives in
src/team_page_template.html; this script cuts the sound samples, draws their
spectrograms and fills in the numbers.
"""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, spectrogram

import config as C
from audio_io import find_recordings
from denoise import Denoiser
from export_clips import sun_elevation
from pretrained import resample

OUT_SR = 32000          # keeps everything below 16 kHz; flight calls are 2-10 kHz
CLIP_S = 2.4
SPEC_FMAX = 12000

# Gallery: confident night detections, checked by eye on their spectrograms.
GALLERY = [
    dict(id="gcth", stem="100626-193652", win=(6772.6, 6774.4), name="Gray-cheeked Thrush",
         team=True, agree="Nighthawk: 99.99% sure",
         fact="Nests near the Arctic tundra in Canada and Alaska and spends the winter in South America. Its night call slides down: “vee-er”."),
    dict(id="wtsp", stem="100726-045604", win=(1432.6, 1434.6), name="White-throated Sparrow",
         team=True, agree="all 4 computers agree",
         fact="Its song sounds like “Oh sweet Canada, Canada, Canada”. Many spend the winter in New Jersey backyards. Its flight call is a thin, rising “seet”."),
    dict(id="savs", stem="100726-012217", win=(4702.0, 4703.8), name="Savannah Sparrow",
         agree="all 4 computers agree",
         fact="A small striped sparrow of grassy fields. It was the bird our computers heard most often in the dark."),
    dict(id="swth", stem="101026-003853", win=(2426.0, 2428.0), name="Swainson's Thrush",
         agree="3 computers agree", band=(1500, 10000),
         fact="Its flight call is a soft “heep” that sounds a lot like a spring peeper frog. It flies all the way to South America."),
    dict(id="chsp", stem="100626-193652", win=(10917.2, 10919.0), name="Chipping Sparrow",
         agree="3 computers agree",
         fact="Wears a rusty-red cap. Its flight call is a very high, thin “tsiit”."),
    dict(id="oven", stem="100726-014940", win=(1314.0, 1317.0), name="Ovenbird",
         agree="3 computers agree",
         fact="A warbler that builds a domed nest on the forest floor, shaped like an old-fashioned outdoor oven. Its song is “teacher, teacher, teacher!”"),
    dict(id="amre", stem="100526-213546", win=(6030.6, 6032.4), name="American Redstart",
         agree="2 computers agree",
         fact="A warbler that flashes the orange patches on its tail and wings to scare bugs out of the leaves, then catches them."),
]
# Before/after demo: the same redstart call, raw and cleaned.
BEFORE_AFTER = dict(stem="100526-213546", win=(6030.6, 6032.4))
# Bird-or-bug quiz.
QUIZ = [
    dict(id="q1", stem="100626-190929", win=(9191.6, 9194.5), answer="bug", clip_s=3.0,
         why="It's an insect, probably a katydid. Look at the picture: the same little chirp repeats again and again, evenly spaced like a ticking clock. Birds don't call like a metronome. This sound fooled two of our computers, so it's a great one to remember."),
    dict(id="q2", stem="100626-224316", win=(6994.5, 6997.5), answer="bird", clip_s=2.4,
         why="It's a bird! All 4 computers said Savannah Sparrow. One short, thin call that sweeps through the high notes is what a night flight call looks like."),
    dict(id="q3", stem="100326-221200", win=(3721.5, 3726.5), answer="other", clip_s=5.0,
         why="This is one of the clips the team cut by hand on Oct 3. None of the computers think it's a bird. The picture shows a wide smear from low to high notes that lasts about 3 seconds: that is how a passing car, an airplane or a gust of wind looks. Listen again together and decide. You are the judges!"),
]


def b64(data: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def find_peak(x, sr, band):
    sos = butter(4, band, "bandpass", fs=sr, output="sos")
    e = np.convolve(sosfiltfilt(sos, x) ** 2, np.ones(int(0.05 * sr)), "same")
    return int(np.argmax(e))


def cut(rec, dn, win, clip_s, denoise=True, band=(2000, 10000)):
    """Clip of clip_s seconds centered on the loudest call moment inside win."""
    ctx = dn.context_s
    a0 = max(0.0, win[0] - clip_s - ctx)
    raw = rec.read(a0, (win[1] - a0) + clip_s + ctx)
    x = dn(raw) if denoise else raw
    lo, hi = int((win[0] - a0) * rec.sample_rate), int((win[1] - a0) * rec.sample_rate)
    pk = lo + find_peak(dn(raw)[lo:hi], rec.sample_rate, band)   # same center for raw & clean
    n = int(clip_s * rec.sample_rate)
    s = max(0, pk - n // 2)
    y = x[s:s + n]
    t0 = a0 + s / rec.sample_rate
    return resample(y, rec.sample_rate, OUT_SR), t0


def wav_uri(y):
    y = y / (np.abs(y).max() + 1e-9) * 10 ** (-3 / 20)
    buf = io.BytesIO()
    sf.write(buf, y, OUT_SR, format="WAV", subtype="PCM_16")
    return b64(buf.getvalue(), "audio/wav")


def spec_uri(y, vmin_db=55):
    """Spectrogram image that fills the whole picture (axes drawn in HTML), so a
    playhead at x% is exactly at x% of the clip."""
    f, t, S = spectrogram(y, OUT_SR, nperseg=512, noverlap=448)
    L = 10 * np.log10(S + 1e-14)
    keep = f <= SPEC_FMAX
    fig = plt.figure(figsize=(6, 2.4), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.pcolormesh(t, f[keep], L[keep], shading="auto", cmap="magma",
                  vmin=L.max() - vmin_db, vmax=L.max())
    ax.set_xlim(0, len(y) / OUT_SR)
    ax.set_ylim(0, SPEC_FMAX)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return b64(buf.getvalue(), "image/png")


def friendly_time(dt) -> str:
    return f"{dt:%b} {dt.day}, {dt.hour % 12 or 12}:{dt:%M} {'AM' if dt.hour < 12 else 'PM'}"


def night_label(night: str) -> str:
    return {"2026_10_01": "Oct 1→2", "2026_10_03": "Oct 3", "2026_10_05": "Oct 5→6",
            "2026_10_06": "Oct 6→7 (recorder 1)", "2026_10_07": "Oct 6→7 (recorder 2)",
            "2026_10_09": "Oct 9→10"}.get(night, night)


def chart_data(recs):
    det = pd.read_csv(C.OUTPUT_DIR / "detections.csv")
    dark = {}
    for r in recs:        # hours with the sun more than 12 degrees below the horizon
        ts = np.arange(0, r.duration, 60.0)
        dark[r.night] = dark.get(r.night, 0) + sum(sun_elevation(r.time_at(t)) < -12 for t in ts) / 60
    per_night = []
    for night, h in sorted(dark.items()):
        n = int(((det.night == night) & (det.period == "night")).sum())
        label = night_label(night) + (" (only 2 h recorded)" if h < 3 else "")
        per_night.append(dict(label=label, value=round(n / h, 1) if h else 0,
                              detail=f"{n} calls in {h:.1f} dark hours"))
    sp = det[det.period == "night"].best_label.value_counts()
    sp = sp[~sp.index.str.contains("species unknown|Unknown")].head(8)
    species = [dict(label=k, value=int(v), detail=f"{v} times in the dark") for k, v in sp.items()]
    man = pd.read_csv(C.OUTPUT_DIR / "manual_clip_check.csv")
    totals = dict(hours=round(sum(r.duration for r in recs) / 3600, 1), recordings=len(recs),
                  clips=int(det.clip_file.nunique()), detections=int(len(det)),
                  night_calls=int((det.period == "night").sum()),
                  team_found=int(man.found.sum()), team_total=int(len(man)),
                  dark_hours=round(sum(dark.values()), 1))
    return per_night, species, totals


def main():
    recs = find_recordings(C.RECORDINGS_DIR)
    by = {r.stem: r for r in recs}
    dn = Denoiser()

    gallery = []
    for g in GALLERY:
        r = by[g["stem"]]
        y, t0 = cut(r, dn, g["win"], CLIP_S, band=g.get("band", (2000, 10000)))
        gallery.append(dict(id=g["id"], name=g["name"], fact=g["fact"], agree=g["agree"],
                            team=g.get("team", False), dur=CLIP_S,
                            when=friendly_time(r.time_at(t0 + CLIP_S / 2)),
                            file=f"{g['stem']} at {t0:.0f} s", audio=wav_uri(y), img=spec_uri(y)))

    r = by[BEFORE_AFTER["stem"]]
    raw, _ = cut(r, dn, BEFORE_AFTER["win"], CLIP_S, denoise=False)
    clean, _ = cut(r, dn, BEFORE_AFTER["win"], CLIP_S, denoise=True)
    before_after = dict(raw_audio=wav_uri(raw), raw_img=spec_uri(raw, 70),
                        clean_audio=wav_uri(clean), clean_img=spec_uri(clean, 70), dur=CLIP_S)

    quiz = []
    for q in QUIZ:
        r = by[q["stem"]]
        y, _ = cut(r, dn, q["win"], q["clip_s"])
        quiz.append(dict(id=q["id"], answer=q["answer"], why=q["why"], dur=q["clip_s"],
                         audio=wav_uri(y), img=spec_uri(y)))

    per_night, species, totals = chart_data(recs)
    data = dict(gallery=gallery, before_after=before_after, quiz=quiz, per_night=per_night,
                species=species, totals=totals,
                detectors=[dict(label="BirdNET", value=21, detail="caught 21 of every 100 team calls"),
                           dict(label="Perch v2 (Google)", value=44, detail="caught 44 of every 100 team calls"),
                           dict(label="Nighthawk", value=47, detail="caught 47 of every 100 team calls"),
                           dict(label="Our own detector", value=71, detail="caught 71 of every 100 team calls", ours=True)])
    template = (Path(__file__).parent / "team_page_template.html").read_text(encoding="utf-8")
    html = template.replace("__PAGE_DATA__", json.dumps(data))
    for k, v in totals.items():
        html = html.replace(f"__{k.upper()}__", f"{v:,}" if isinstance(v, int) else str(v))
    out = C.OUTPUT_DIR / "team_page"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "night_sky_listener.html"
    path.write_text(html, encoding="utf-8")
    print(path, f"{path.stat().st_size / 1e6:.1f} MB", totals)
    print(json.dumps(per_night, indent=1), json.dumps(species, indent=1))


if __name__ == "__main__":
    main()
