"""Stage 1: run every model over every recording and save the scores.

For each recording (processed in 10-minute chunks):
  1. denoise (GPU)                                 -> denoise.py
  2. Nighthawk, 1-s windows every 0.2 s (CPU)      -> pretrained.py
  3. BirdNET, 3-s windows every 1.5 s (CPU)
  4. Perch v2, 5-s windows every 2.5 s (CPU)
Scores go to outputs/cache/<night>/<recording>.npz, so later steps (combining
models, cutting clips) can be re-run in seconds without repeating this.
Our own flight-call detector (GPU, seconds per file) is run separately with
--cnn and saved as <recording>.cnn.npy, so it can be retrained cheaply.

Usage:
  python src/scan.py                      # all recordings in E:/sensor_recording
  python src/scan.py --birdvox 07 10      # BirdVox evaluation nights
  python src/scan.py --workers 2          # split files over 2 processes
  python src/scan.py --cnn                # (re)compute our detector's scores
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import soundfile as sf

from config import CACHE_DIR, DATA_DIR, MODEL_DIR, RECORDINGS_DIR

CHUNK_S = 600.0          # multiple of every model's hop, so windows line up
TAIL_S = 5.0             # extra audio so windows near a chunk end are complete
TOP_K = 10
PRETRAINED_INPUT = "raw"   # experiment 1: denoising lowered BirdNET & Nighthawk accuracy


class FlacSource:
    """BirdVox-full-night night, with the same read() interface as Recording."""

    def __init__(self, unit: str):
        self.path = DATA_DIR / "birdvox_full_night" / f"BirdVox-full-night_flac-audio_unit{unit}.flac"
        self.stem, self.night = f"birdvox_unit{unit}", "birdvox"
        info = sf.info(self.path)
        self.sample_rate, self.frames = info.samplerate, info.frames
        self.duration = self.frames / self.sample_rate

    def read(self, start_s=0.0, dur_s=None, mono=True):
        a = int(round(start_s * self.sample_rate))
        n = self.frames - a if dur_s is None else int(round(dur_s * self.sample_rate))
        x, _ = sf.read(self.path, start=a, frames=min(n, self.frames - a), dtype="float32")
        return x.mean(1) if (mono and x.ndim > 1) else x


def topk(scores: np.ndarray, k=TOP_K):
    idx = np.argpartition(-scores, k, axis=1)[:, :k]
    val = np.take_along_axis(scores, idx, 1)
    order = np.argsort(-val, axis=1)
    return np.take_along_axis(idx, order, 1).astype(np.int16), np.take_along_axis(val, order, 1)


class Scanner:
    def __init__(self, threads: int):
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        from pretrained import BirdNET, Nighthawk, Perch
        self.bn = BirdNET(threads=threads)
        self.pc = Perch()
        self.nh = Nighthawk()
        self.bn_bird = np.array([c not in self.bn.noise_classes for c in self.bn.common])
        self.bn_noise = np.flatnonzero(~self.bn_bird)
        self.bn_local = self.bn.expected_species() & self.bn_bird
        self.pc_bird = np.array([e != "no_ebird_code" for e in self.pc.ebird])
        self._denoisers = {}

    def denoiser(self, sr):
        from denoise import Denoiser
        if sr not in self._denoisers:
            self._denoisers[sr] = Denoiser(sample_rate=sr, n_fft=1024 * sr // 48000,
                                           hop=256 * sr // 48000)
        return self._denoisers[sr]

    def scan(self, src, out_path: Path):
        from pretrained import resample
        sr = src.sample_rate
        dn = self.denoiser(sr)
        ctx = dn.context_s
        sig = lambda z: 1 / (1 + np.exp(-z.astype(np.float32)))
        acc = {k: [] for k in ["bn_t", "bn_idx", "bn_val", "bn_bird", "bn_local", "bn_noise",
                               "pc_t", "pc_idx", "pc_val", "pc_bird_logit", "pc_bird_mass", "pc_emb",
                               "nh_t", "nh0", "nh1", "nh2", "nh3"]}
        t_start = time.time()
        start = 0.0
        while start < src.duration:
            n_main = min(CHUNK_S, src.duration - start)
            a = max(0.0, start - ctx)
            raw = src.read(a, (start - a) + n_main + TAIL_S + ctx)
            lo = int(round((start - a) * sr))
            # (1) denoise; pretrained models get the denoised or raw version.
            x = dn(raw) if PRETRAINED_INPUT == "denoised" else raw
            x = x[lo:lo + int(round((n_main + TAIL_S) * sr))]
            # (2) Nighthawk
            st, outs = self.nh.logits(resample(x, sr, 22050))
            keep = st < n_main - 1e-6
            acc["nh_t"].append(start + st[keep])
            for k in range(4):
                acc[f"nh{k}"].append(outs[k][keep].astype(np.float16))
            # (3) BirdNET
            st, lg = self.bn.logits(resample(x, sr, 48000), hop_s=1.5)
            keep = st < n_main - 1e-6
            pr = sig(lg[keep])
            acc["bn_t"].append(start + st[keep])
            i, v = topk(pr)
            acc["bn_idx"].append(i); acc["bn_val"].append(v.astype(np.float16))
            acc["bn_bird"].append(pr[:, self.bn_bird].max(1).astype(np.float16))
            acc["bn_local"].append(pr[:, self.bn_local].max(1).astype(np.float16))
            acc["bn_noise"].append(pr[:, self.bn_noise].astype(np.float16))
            # (4) Perch v2
            st, lg, emb = self.pc.run(resample(x, sr, 32000), hop_s=2.5)
            keep = st < n_main - 1e-6
            lg = lg[keep].astype(np.float32)
            sm = np.exp(lg - lg.max(1, keepdims=True)); sm /= sm.sum(1, keepdims=True)
            acc["pc_t"].append(start + st[keep])
            i, v = topk(lg)
            acc["pc_idx"].append(i); acc["pc_val"].append(v.astype(np.float16))
            acc["pc_bird_logit"].append(lg[:, self.pc_bird].max(1).astype(np.float16))
            acc["pc_bird_mass"].append(sm[:, self.pc_bird].sum(1).astype(np.float16))
            acc["pc_emb"].append(emb[keep])
            start += CHUNK_S
        out = {k: np.concatenate(v) for k, v in acc.items()}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp.npz")
        np.savez(tmp, duration=src.duration, pretrained_input=PRETRAINED_INPUT, **out)
        os.replace(tmp, out_path)
        el = time.time() - t_start
        print(f"{src.stem}: {src.duration / 3600:.2f} h in {el / 60:.1f} min "
              f"({src.duration / el:.0f}x real time)", flush=True)


def scan_cnn(src, out_path: Path, model, fe):
    """Our detector's per-frame call probability for a whole recording."""
    from flightcall_cnn import FRAME_S, frame_probs
    from pretrained import resample
    sr, parts, start = src.sample_rate, [], 0.0
    while start < src.duration:
        n_main = min(CHUNK_S, src.duration - start)
        a = max(0.0, start - fe.context_s)
        raw = src.read(a, (start - a) + n_main + fe.context_s)
        p = frame_probs(model, fe(resample(raw, sr, 24000)))
        f0 = int(round((start - a) / FRAME_S))
        parts.append(p[f0:f0 + int(round(n_main / FRAME_S))].astype(np.float16))
        start += CHUNK_S
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, np.concatenate(parts))


def sources(args):
    if args.birdvox:
        return [FlacSource(u) for u in args.birdvox]
    from audio_io import find_recordings
    return find_recordings(RECORDINGS_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--birdvox", nargs="*", help="BirdVox unit numbers instead of our recordings")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--worker-id", type=int, default=None)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cnn", action="store_true", help="only run our detector (GPU)")
    args = ap.parse_args()

    if args.workers > 1 and args.worker_id is None:
        import subprocess
        cmd = [sys.executable, "-u", __file__, "--workers", str(args.workers),
               "--threads", str(max(4, os.cpu_count() // args.workers))]
        if args.birdvox:
            cmd += ["--birdvox", *args.birdvox]
        if args.force:
            cmd.append("--force")
        procs = [subprocess.Popen(cmd + ["--worker-id", str(i)]) for i in range(args.workers)]
        sys.exit(max(p.wait() for p in procs))

    todo = sources(args)
    if args.cnn:
        from flightcall_cnn import SNRFrontend, load_model
        model, fe = load_model(MODEL_DIR / "flightcall_cnn.pt"), SNRFrontend("cuda")
        for src in todo:
            t = time.time()
            scan_cnn(src, CACHE_DIR / src.night / f"{src.stem}.cnn.npy", model, fe)
            print(f"{src.stem}: detector done in {time.time() - t:.0f}s", flush=True)
        return
    if args.worker_id is not None:
        todo = todo[args.worker_id::args.workers]
    scanner = None
    for src in todo:
        out = CACHE_DIR / src.night / f"{src.stem}.npz"
        if out.exists() and not args.force:
            continue
        scanner = scanner or Scanner(args.threads)
        scanner.scan(src, out)
    print("SCAN_DONE", flush=True)


if __name__ == "__main__":
    main()
