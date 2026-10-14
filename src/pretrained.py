"""Wrappers around the three pretrained bird-sound models.

* BirdNET V2.4 (Cornell / Chemnitz): 6,522 classes, mostly bird species plus
  a few noise classes (Engine, Dog, Human...). Looks at 3-second windows.
* Perch v2 (Google DeepMind): ~15,000 classes including insects and frogs.
  Looks at 5-second windows. Also gives a 1,536-number "embedding" (a summary
  of the sound) that we reuse to train our own classifier.
* Nighthawk (Van Doren et al. 2024): built specifically for nocturnal flight
  calls of the Americas. 1-second windows, outputs order/family/group/species.

Each wrapper takes a long mono signal at the model's own sample rate and
returns one row of scores per window.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
from scipy.signal import resample_poly


def frame(x: np.ndarray, win: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    """Cut x into overlapping windows (zero-padding the end). Returns (starts, frames)."""
    n = max(1, int(np.ceil(max(0, len(x) - win) / hop)) + 1)
    need = (n - 1) * hop + win
    if len(x) < need:
        x = np.pad(x, (0, need - len(x)))
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return hop * np.arange(n), x[idx]


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    g = np.gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


# --------------------------------------------------------------------- BirdNET
class BirdNET:
    sr, win_s = 48000, 3.0

    def __init__(self, threads: int = 16, batch: int = 64):
        import birdnet_analyzer.config as bcfg
        from tensorflow import lite as tflite
        self.it = tflite.Interpreter(model_path=bcfg.BIRDNET_MODEL_PATH, num_threads=threads)
        self.inp = self.it.get_input_details()[0]["index"]
        self.out = self.it.get_output_details()[0]["index"]
        self.batch = batch
        self._shape = None
        with open(bcfg.BIRDNET_LABELS_FILE, encoding="utf-8") as fh:
            raw = [l.strip() for l in fh if l.strip()]
        self.sci = [l.split("_")[0] for l in raw]
        self.common = [l.split("_")[1] if "_" in l else l for l in raw]
        # The non-bird classes; everything else is a bird (or a few frogs/insects/mammals).
        self.noise_classes = {"Dog", "Engine", "Environmental", "Fireworks", "Gun",
                              "Noise", "Siren", "Human vocal", "Human non-vocal",
                              "Human whistle", "Power tools"}
        self._mdata_path = bcfg.MDATA_MODEL_PATH

    def _invoke(self, xb: np.ndarray) -> np.ndarray:
        if self._shape != xb.shape:
            self.it.resize_tensor_input(self.inp, list(xb.shape))
            self.it.allocate_tensors()
            self._shape = xb.shape
        self.it.set_tensor(self.inp, xb)
        self.it.invoke()
        return self.it.get_tensor(self.out).copy()

    def logits(self, x: np.ndarray, hop_s: float = 1.5):
        starts, fr = frame(x, int(self.win_s * self.sr), int(hop_s * self.sr))
        out = []
        for i in range(0, len(fr), self.batch):
            xb = np.ascontiguousarray(fr[i:i + self.batch], dtype=np.float32)
            if len(xb) < self.batch:   # keep one tensor shape -> no re-allocation
                xb = np.concatenate([xb, np.zeros((self.batch - len(xb), xb.shape[1]), np.float32)])
                out.append(self._invoke(xb)[:len(fr) - i])
            else:
                out.append(self._invoke(xb))
        return starts / self.sr, np.concatenate(out).astype(np.float16)

    def expected_species(self, weeks=(37, 38, 39), threshold: float = 0.03) -> np.ndarray:
        """Boolean mask: species BirdNET's range model expects here in early October."""
        from tensorflow import lite as tflite
        from config import LATITUDE, LONGITUDE
        it = tflite.Interpreter(model_path=self._mdata_path)
        it.allocate_tensors()
        i, o = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
        best = 0
        for w in weeks:
            it.set_tensor(i, np.array([[LATITUDE, LONGITUDE, w]], np.float32))
            it.invoke()
            best = np.maximum(best, it.get_tensor(o)[0])
        return best >= threshold


# ----------------------------------------------------------------------- Perch
class Perch:
    sr, win_s = 32000, 5.0

    def __init__(self, batch: int = 32):
        import tensorflow as tf
        import birdnet_analyzer.config as bcfg
        self.fn = tf.saved_model.load(bcfg.PERCH_V2_MODEL_PATH).signatures["serving_default"]
        self.batch = batch
        assets = os.path.join(bcfg.PERCH_V2_MODEL_PATH, "assets")
        self.sci = pd.read_csv(os.path.join(assets, "labels.csv")).iloc[:, 0].tolist()
        self.ebird = pd.read_csv(os.path.join(assets, "perch_v2_ebird_classes.csv")).iloc[:, 0].tolist()

    def run(self, x: np.ndarray, hop_s: float = 2.5):
        starts, fr = frame(x, int(self.win_s * self.sr), int(hop_s * self.sr))
        logits, emb = [], []
        for i in range(0, len(fr), self.batch):
            r = self.fn(inputs=np.ascontiguousarray(fr[i:i + self.batch], dtype=np.float32))
            logits.append(r["label"].numpy().astype(np.float16))
            emb.append(r["embedding"].numpy().astype(np.float16))
        return starts / self.sr, np.concatenate(logits), np.concatenate(emb)


# ------------------------------------------------------------------- Nighthawk
class Nighthawk:
    """Nighthawk with batched inference.

    The published package feeds the network one 1-s window at a time (~36/s).
    The saved model contains the spectrogram step (`preprocess`) and the
    network (`model`) separately; calling them in batches gives identical
    numbers (checked: max difference < 1e-5) about 10x faster. Nighthawk's own
    post-processing (calibration, thresholds, taxonomy merging) is reused as is.
    """
    sr, win_s, hop_s = 22050, 1.0, 0.2

    def __init__(self, batch: int = 512):
        import tensorflow as tf
        from nighthawk import detector
        self.tf = tf
        self.detector = detector
        self.m = tf.saved_model.load(str(detector._MODEL_DIR_PATH))
        # A fixed signature avoids re-compiling the graph for every new batch size.
        self._pre = tf.function(
            lambda b: tf.map_fn(self.m.preprocess, b, parallel_iterations=64,
                                fn_output_signature=tf.float32),
            input_signature=[tf.TensorSpec([None, int(self.sr * self.win_s)], tf.float32)])
        self.batch = batch

    def logits(self, x: np.ndarray):
        """Returns (starts_s, [order, family, group, species] logit arrays)."""
        win, hop = int(self.win_s * self.sr), int(round(self.hop_s * self.sr))
        n = (len(x) - win) // hop + 1          # Nighthawk drops the incomplete tail
        starts, fr = frame(x[: (n - 1) * hop + win], win, hop)
        outs = [[], [], [], []]
        for i in range(0, len(fr), self.batch):
            spec = self._pre(np.ascontiguousarray(fr[i:i + self.batch], dtype=np.float32))
            for k, o in enumerate(self.m.model(spec[..., None], training=False)):
                outs[k].append(o.numpy())
        return starts / self.sr, [np.concatenate(o) for o in outs]

    def detections(self, preds: list[np.ndarray], name: str = "audio",
                   threshold: float = 80, drop_uncertain: bool = True) -> pd.DataFrame:
        """Nighthawk's standard detection table from cached logits."""
        from nighthawk import run_reconstructed_model as rrm
        d = self.detector
        p = d._get_configuration_file_paths()

        def runner(model, file_path, input_dur, hop_dur, target_sr):
            return [a.copy() for a in preds], [], len(preds[0])

        df, _ = rrm.run_model_on_file(
            None, name, self.sr, self.win_s, self.hop_s, p.species, p.groups,
            p.families, p.orders, p.ebird_taxonomy, p.group_ebird_codes,
            p.calibrators, p.config, threshold=threshold / 100, quiet=True,
            model_runner=runner, postprocess_drop_singles_by_tax_level=drop_uncertain,
            postprocess_merge_overlaps=True, postprocess_retain_only_overlaps=drop_uncertain,
            mask_output_ap_threshold=d.DEFAULT_AP_MASK_THRESHOLD,
            test_set_performance_dir=p.test_set_performance)
        return df
