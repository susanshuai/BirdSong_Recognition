# Night Sky Listener: automatic detection of nocturnal bird flight calls

**Project documentation: what was built, why, and how well it works**

| | |
|---|---|
| Project folder | `C:\Users\susan\CodingProjects\BirdSong_Recognition` |
| Recordings | `E:\sensor_recording` (6 night folders, 30 full recordings, 67.6 h) |
| Work done | 2026-10-14 |
| Team | Tantalum 73 (FIRST LEGO League, BIOGLOW season), Innovation Project "Night Sky Listener" |
| Status | Full run complete: 3,079 detections in 2,235 clips; models trained and evaluated |

---

## Contents

1. [Summary](#1-summary)
2. [The task](#2-the-task)
3. [The recordings (data inventory)](#3-the-recordings-data-inventory)
4. [Computer setup](#4-computer-setup)
5. [Outside resources](#5-outside-resources)
6. [The pipeline, part by part](#6-the-pipeline-part-by-part)
7. [Experiments and evaluation](#7-experiments-and-evaluation)
8. [Results on the team's recordings](#8-results-on-the-teams-recordings)
9. [Problems found along the way and how they were fixed](#9-problems-found-along-the-way-and-how-they-were-fixed)
10. [Limitations](#10-limitations)
11. [How to re-run and how to add new nights](#11-how-to-re-run-and-how-to-add-new-nights)
12. [File inventory](#12-file-inventory)
13. [Suggested next steps](#13-suggested-next-steps)
14. [Credits, licenses and references](#14-credits-licenses-and-references)

---

## 1. Summary

Thirty long night recordings (67.6 hours) were processed automatically. The
pipeline:

1. **Denoises** the audio on the GPU (removes low road/wind rumble and steady
   cricket/katydid bands while keeping short bird calls).
2. Runs **four detectors** over every second of audio:
   - our own **flight-call detector** (a neural network trained here on 35,000
     expert-marked night calls from BirdVox-full-night),
   - **Nighthawk** (published night-flight-call model),
   - **BirdNET V2.4** (Cornell / TU Chemnitz),
   - **Perch v2** (Google DeepMind).
3. Combines their scores with a small **referee** model (logistic regression)
   trained on a fully expert-labelled night.
4. Names the bird by a **weighted vote** of four opinions: Nighthawk, our own
   **species classifier** (trained here on 80,000 labelled flight calls from
   BirdVox-ANAFCC), BirdNET and Perch.
5. Cuts every detection out as a denoised, loudness-normalized clip named
   `<recording>-<start s>-<end s>_auto.WAV` next to the original recording, and
   writes a table (`auto_detections.csv`) per night folder.

Key numbers:

| Measure | Result |
|---|---|
| Expert-labelled test night (BirdVox unit 10, never used for training) | **96.7%** of calls found; **87%** of detections real |
| Team's own 34 hand-cut clips | **26** found by the final pipeline |
| Our detector vs. pretrained models on the team's clips (AUC) | **0.91** (Nighthawk 0.87, Perch 0.85, BirdNET 0.80) |
| Visual check of 36 random night detections | ~27 clear calls, 2 insects, ~7 too faint to judge (≈75-85% real) |
| Species classifier on held-out sites | 92.9% accuracy (89.9% balanced) |
| Output | 3,079 detections, 2,235 clips (1.6 GB), 802 detections in full darkness |

A kid-friendly summary page with embedded sound samples is at
`outputs\team_page\night_sky_listener.html` (see [§6.11](#611-the-team-page)).

---

## 2. The task

Request (2026-10-14): process the night recordings in `E:\sensor_recording`
(files named `xxxxxx_xxxxxx.WAV`), ignore other files (hand-cut clips), denoise
the audio (cars, crickets), build several models (machine learning / deep
learning, with BirdNET-Analyzer as a reference) to detect and distinguish bird
calls, and save each bird segment as `xxxxxx_xxxxxx_yyyy_yyyy_auto.WAV`
where `yyyy` are the start and end seconds in the original file, following the
naming of the hand-cut clips.

Context from the team's Innovation Project Plan (Sept 26, 2026): the project is
a low-cost "Night Sky Listener" that records nocturnal flight calls, names the
species with AI, and combines that with BirdCast radar to send lights-out
alerts. The plan asks for an "AI report card" (how often the AI is right) and
for kids to check doubtful calls by eye and ear.

**Naming decision.** The real recordings use dashes (`100626-193652.WAV`) and
the hand-cut clips continue with dashes (`100626-193652-6773-6774.wav`). Auto
clips keep exactly that prefix and add `_auto`:
`100626-193652-6772-6775_auto.WAV`. They sort right next to hand-cut clips of
the same moment. The pattern is one setting (`CLIP_NAME` in `src/config.py`).

---

## 3. The recordings (data inventory)

### 3.1 Files

Full recordings match `MMDDYY-HHMMSS.WAV` (the start time on the recorder's
clock). All are 48 kHz, 32-bit float, Zoom recorder (`bext`/`zTRK`/`MicR` chunks).

| Folder | Recordings | Channels | Hours | First start | Notes |
|---|---|---|---|---|---|
| 2026_10_01 | 6 | stereo | 8.6 | Oct 1 22:32 | 1.55-h files |
| 2026_10_03 | 2 | stereo | 2.0 | Oct 3 22:12 | only ~2 h recorded; 2 hand-cut clips |
| 2026_10_05 | 5 | stereo (morning file) / mono | 12.1 | Oct 5 08:10 | starts with a daytime file; 11 hand-cut "_band" clips |
| 2026_10_06 | 5 | mono | 12.8 | Oct 6 19:36 | 21 hand-cut clips (some named by species) |
| 2026_10_07 | 5 | mono | 13.2 | Oct 6 19:09 | **same night as 2026_10_06**, so probably a second station |
| 2026_10_09 | 7 | mono | 18.9 | Oct 9 15:19 | includes an afternoon file and the next morning |
| **Total** | **30** | | **67.6** | | 34 hand-cut clips |

### 3.2 Ten recordings have broken headers

When the recorder stops abruptly it leaves the RIFF and `data` size fields at
0. Many programs (including the `soundfile` library, and possibly Audacity)
then treat the file as empty. The audio itself is intact. Affected files:

- `2026_10_07`: `100626-221553`, `100726-012217`, `100726-042841`, `100726-073505`
- `2026_10_09`: `100926-182605`, `100926-213229`, `101026-003853`, `101026-034517`, `101026-065141`, `101026-095805`

`src/audio_io.py` parses the header itself and takes the audio length from
the file size, so these files are processed normally. The originals were not
modified. (If the team wants to open them in Audacity, header-repaired copies
can be written on request.)

### 3.3 Hand-cut clips

34 clips, three naming styles, all parsed by `find_manual_clips()`:
`100626-193652-3227-3229.wav`, `100526-081044_band-675-684.wav`,
`100726-014940-2256-2262-White-throated Sparrow.wav`. Five carry a species
name (3× White-throated Sparrow, White-breasted Nuthatch, Northern Cardinal,
Canada Goose). One name looks like a typo: `100726-045604-5749-5793.wav` is
4.2 s long, so probably 5749-5753.

### 3.4 What the noise looks like

Spectrograms of random stretches showed:

- steady horizontal bands at about 2.5, 3.5-4, 5-7 and 9-10 kHz all night
  (crickets and katydids), in the same range as flight calls;
- a constant rumble below ~1 kHz (traffic, wind);
- the team's night calls as tiny marks of 50-150 ms: high "tseeps" at
  6-9 kHz (sparrows, warblers) and lower ~3 kHz notes (thrushes).

---

## 4. Computer setup

| Item | Value |
|---|---|
| Machine | Windows 11, 32 logical CPU threads, 96 GB RAM, NVIDIA RTX 4090 (24 GB) |
| Package manager | `uv` 0.12.20 |
| Main environment `.venv` | Python 3.11.9; birdnet-analyzer 2.4.0; TensorFlow 2.21; PyTorch 2.14.1+cu126; librosa, soundfile, scikit-learn, pandas, matplotlib, astral, h5py |
| Nighthawk environment `.venv-nighthawk` | Python 3.10; nighthawk 0.3.1; TensorFlow 2.12.1 (used only to verify the official command line) |
| Version control | local git repository in the project folder, no remote yet |

TensorFlow has no GPU support on native Windows, so BirdNET, Perch and
Nighthawk run on the CPU; the denoiser and our own two models run on the GPU.
Nighthawk is also installed (without dependencies) in the main environment,
where its saved model loads fine under TensorFlow 2.21.

---

## 5. Outside resources

### 5.1 Pretrained models (used as published)

| Model | Source | Input | Output |
|---|---|---|---|
| BirdNET V2.4 (FP32 TFLite) | birdnet-analyzer 2.4.0 | 3-s windows, 48 kHz | 6,522 classes (birds + Dog, Engine, Human, Siren, etc.) |
| BirdNET range model (MData V2) | birdnet-analyzer | latitude, longitude, week | which species are expected; used with 40.35 N, -74.62 W, weeks 37-39 |
| Perch v2 (CPU SavedModel) | Kaggle, via birdnet-analyzer's downloader | 5-s windows, 32 kHz | 14,795 classes (birds, insects, frogs, other sounds) + 1,536-number embedding |
| Nighthawk 0.3.1 | PyPI `nighthawk` | 1-s windows, 22.05 kHz | 4 orders, 19 families, 17 call groups, 130 species codes |

### 5.2 Training data (downloaded to `data/`, git-ignored)

| Dataset | What it is | Size | License |
|---|---|---|---|
| BirdVox-full-night (Zenodo 1205569) | 6 complete fall-migration nights (2015, Ithaca NY area), 24 kHz FLAC, with the time and pitch of every flight call marked by experts (~35,000 calls) | 5.6 GB | CC-BY-4.0 |
| BirdVox-ANAFCC (Zenodo 3666782) | ~85,000 short flight-call clips (Old Bird, CLO-43SD, Macaulay, xeno-canto, extra), 22.05 kHz, labelled by taxonomy code; species also appear in the clip names | 2.3 GB | CC-BY-4.0 |

Zenodo throttles each connection to ~150 KB/s, so `data/download.py`
downloads every file in parallel 32 MB byte ranges (48 threads, ~7 MB/s,
with retry and back-off for occasional HTTP 503 replies).

---

## 6. The pipeline, part by part

```
E:\sensor_recording\*.WAV
   │  audio_io.py   (safe reading, broken headers)
   ▼
scan.py ──► raw audio ──► Nighthawk, BirdNET, Perch (CPU) ─┐
   │                                                       ├─► outputs/cache/<night>/<rec>.npz
   └─────► 24 kHz audio ─► SNR front-end ─► our detector ──┘    outputs/cache/<night>/<rec>.cnn.npy
                                                    │
detect.py   candidates + evidence  ◄────────────────┘
   │        referee (models/stacker.pkl)
   ▼
export_clips.py   species vote (+ our species model), sun position, clip cutting
   │              denoise.py for the clip audio
   ▼
E:\sensor_recording\<night>\<rec>-<start>-<end>_auto.WAV + auto_detections.csv
outputs/detections.csv, summarize.py tables, review_sheet.py, build_team_page.py
```

### 6.1 Reading audio (`src/audio_io.py`)

- `find_recordings()` matches only `^\d{6}-\d{6}\.WAV$`, so hand-cut clips
  and `_auto` clips are never treated as recordings.
- `open_recording()` walks the RIFF chunks to the `data` chunk and computes the
  frame count from the file size (handles the zero-size headers). Audio is read
  through a read-only NumPy memory map, so 2 GB files never load in full.
- The start time comes from the file name; `time_at(seconds)` gives clock time.

### 6.2 Denoising (`src/denoise.py`)

Idea: a flight call lasts ~0.1 s; crickets and road noise stay at one pitch for
many seconds.

| Step | Setting |
|---|---|
| STFT | 1,024-point Hann window, hop 256 (48 kHz): 21 ms × 47 Hz |
| Noise floor per frequency bin | median of 0.5-s blocks, then running median over 21 blocks (~10 s) |
| Gate | sound 4 dB above the floor starts to pass, fully passes at 10 dB; 3×3 smoothing before, 3×5 smoothing of the mask after (less "musical noise") |
| Floor | everything else turned down by 30 dB |
| High-pass | 1 kHz with a 300 Hz ramp |
| Context | each chunk is processed with 6 s of extra audio on both sides, so chunk borders don't matter |
| Speed | ~875× real time on the RTX 4090 (a 3.1-h file in ~13 s) |

Visual checks before/after showed calls fully preserved, rumble and hiss
removed, steady cricket bands mostly removed. A strongly pulsing cricket band
(3.5-4 kHz, night of Oct 5) is only partly removed.

### 6.3 Pretrained model wrappers (`src/pretrained.py`)

- **BirdNET**: 3-s windows every 1.5 s, batched 64 at a time; sigmoid scores.
  "Local" = species the range model expects near Princeton in early October
  (weeks 37-39, threshold 0.03).
- **Perch v2**: 5-s windows every 2.5 s at 32 kHz; logits, softmax "bird share"
  (sum over classes that have an eBird code) and embeddings.
- **Nighthawk, batched**. The published package feeds one 1-s window at a time
  (~36 windows/s on this PC; ~11 h for all recordings). Its saved model
  contains a `preprocess` function (waveform → 128×171 spectrogram) and the
  network as separate parts; calling them in batches of 512 gives the same
  numbers (max difference < 1e-5) at ~350 windows/s. Checked against the
  official `nighthawk` command on a 3-minute excerpt: identical detections and
  probabilities, 2.9 s instead of 40 s. Nighthawk's own post-processing
  (calibration, threshold 80, taxonomy merging, dropping isolated windows) is
  reused unchanged through its `model_runner` hook. A fixed `input_signature`
  avoids TensorFlow re-compiling for every batch size.

### 6.4 Our flight-call detector (`src/flightcall_cnn.py`, `src/train_flightcall_cnn.py`)

**Input ("SNR map").** Audio is resampled to 24 kHz; STFT 512/256
(21 ms window, 10.7 ms hop); 64 mel bands from 1 to 11 kHz; for each band, dB
above its running ~10-s median background; clipped to 0-40 dB and scaled to
0-1. This is the denoiser's idea built into the model input, and it makes the
model independent of recorder gain.

**Network.** Four convolution blocks (2× 3×3 conv + batch norm + ReLU each;
32-64-128-128 channels), pooling only along pitch to keep time detail; a
bidirectional GRU (128 units each way); a linear layer giving one
call-probability per 10.7-ms frame. About 0.6 M parameters.

**Labels.** BirdVox marks each call's center time. Frames within ±50 ms are
"call"; frames 50-120 ms away are ignored in the loss (call length unknown);
everything else is "no call".

**Training.** Nights split by recording unit: train 01, 02, 03, 05; validation
07; test 10. Random 2.7-s crops (half centered on a call), batch 64,
AdamW (lr 2e-3, one-cycle schedule, weight decay 1e-3), mixed precision,
20 epochs × 1,500 steps (~70 s per epoch, ~23 min in total). Augmentation:
loudness ×0.6-1.3, pitch shift ±3 mel bands, overlaying a second crop
(maximum of the two maps) with probability 0.3. The best epoch by validation
frame-level average precision was kept (epoch 8, AP 0.929; later epochs
over-fitted slightly).

**Turning frames into events.** `pick_events()`: frames above a threshold,
gaps of ≤ 5 frames merged, events shorter than 3 frames dropped; peak time
and peak probability recorded.

**Test night (unit 10, 6,839 calls).** Frame-level AP 0.918. One-to-one
matching of event peaks to expert call centers within ±0.15 s, threshold 0.85
(chosen on the validation night): precision 0.825, recall 0.696, F1 0.755.
Most misses are very faint calls (many annotated calls are not visible on a
spectrogram) and calls inside flocks merging into one event.

### 6.5 Our species classifier (`src/species_cnn.py`, `src/train_species_cnn.py`)

- **Data**: BirdVox-ANAFCC. Labels come from the clip name (species code such as
  `WTSP`, `SVSP`, `YRWA`) or from the file's taxonomy code (`0.x.x` = not a bird).
  Classes with at least 150 clips were kept: 28 classes, 80,451 clips
  (25 species + "Zeep-type warbler call" + "short double-banded upsweep
  (sparrow/bunting)" + "Not a bird"). The ambiguous code `PALM` was left out.
- **Input**: a 0.6-s window centered on the loudest 50 ms in 1-11 kHz, same SNR
  map as the detector, but with the background taken from the window itself.
- **Network**: four convolution blocks (32-64-128-256 channels), global max +
  mean pooling, dropout 0.3, linear layer to 28 classes.
- **Training**: split by recording site and date (`GroupShuffleSplit`, 20%
  test) so near-identical calls from one night never appear on both sides;
  65,886 train / 14,565 test clips. Class-balanced sampling (weight
  ∝ 1/√count), label smoothing 0.05, AdamW lr 1e-3, 30 epochs, bfloat16.
  Augmentation: time shift, loudness ×0.3-2, mixing in a "Not a bird" clip.
- **Result** on held-out sites: accuracy 92.9%, balanced accuracy 89.9%.
  F1 per species, e.g. White-throated Sparrow 0.982, Savannah Sparrow 0.980,
  Black-throated Blue Warbler 0.976, American Redstart 0.956, Palm Warbler
  0.951, Chipping Sparrow 0.941, Yellow-rumped Warbler 0.832, "Not a bird"
  0.881. Weak only where training clips are few: Black-throated Green Warbler
  0.53 (5 test clips), Northern Waterthrush 0.62, Nashville Warbler 0.68,
  Tennessee Warbler 0.68. Full table: `outputs/experiments/species_cnn_test_report.csv`.
- It knows no thrushes (ANAFCC has almost none); thrush names come from
  Nighthawk and Perch.

### 6.6 Scanning (`src/scan.py`)

Processes each recording in 10-minute chunks (a multiple of every model's
hop, so windows line up) plus 5 s of tail. Pretrained models get **raw**
audio (decided by experiment 1, §7.1). Saved per recording:

- `<rec>.npz`: Nighthawk logits at 0.2-s steps (4 levels), BirdNET top-10
  classes and scores + bird / local-bird / noise-class scores at 1.5-s steps,
  Perch top-10 + bird logit + bird share + 1,536-d embedding at 2.5-s steps.
- `<rec>.cnn.npy`: our detector's frame probabilities (10.7-ms steps), made by
  `scan.py --cnn` on the GPU (~25 s per 3-h recording), so the detector can be
  retrained without repeating the slow CPU scan.

Run time: 30 recordings in ~2.9 h with two worker processes (11-14× real time
each); BirdVox nights 07 and 10 at ~8× real time (78.5 and 57.9 min). Cache size ~1 GB.

### 6.7 Candidates and the referee (`src/detect.py`, `src/train_stacker.py`)

**Candidates** (deliberately low thresholds, high recall): our detector's
events at p ≥ 0.3; Nighthawk windows with any-level probability ≥ 0.8; BirdNET
windows with a local-species score ≥ 0.3. Overlapping or nearly touching
(≤ 0.3 s) spans are merged. Perch does **not** start candidates (see §9).

**Evidence per candidate** (10 features): our detector's max and top-5 mean,
Nighthawk max (any level) and order-level max, BirdNET bird max, local-species
max and noise-class max, Perch bird share and bird logit, log of the duration.

**Referee**: logistic regression on log-odds features, fitted on BirdVox night 07
(4,583 candidates, 75% contain a call), tested on night 10 (3,677 candidates,
68% contain a call; the candidates cover 99.1% of the test night's calls).
Test AUC 0.936, AP 0.969 (gradient-boosted trees: 0.928 / 0.963, so the simpler
model was kept). The decision threshold (0.65) was chosen by 5-fold
cross-validated F1 on night 07. The referee leans on our detector
(top-5 mean, +0.83) and Nighthawk's order level (+0.95); BirdNET and Perch
add little for night calls.

**Kept detections**: referee probability ≥ 0.65, **or** BirdNET local-species
score ≥ 0.7 (BirdVox labels only flight calls, so the referee can't learn to
trust daytime songs).

### 6.8 Species labels (`src/export_clips.py`)

Each detection keeps four opinions:

| Column | Source |
|---|---|
| `nighthawk`, `nighthawk_prob` | most specific taxon from Nighthawk's own detection table (species code translated to English name, or "ZEEP group: … (species unknown)", or a family/order) |
| `our_species_model`, `our_species_prob` | species classifier on a 0.6-s window around our detector's peak |
| `birdnet`, `birdnet_score` | BirdNET's best locally expected species in the span |
| `perch`, `perch_logit` | Perch's best species among those BirdNET's range model expects here |

`best_label` is a weighted vote. Night-call specialists (Nighthawk, our species
model ×0.8) count ×1.0 when the sun is more than 12° below the horizon, ×0.5 at
dawn/dusk twilight and ×0.3 in daylight. BirdNET counts min(1, 2×score); Perch
counts sigmoid(1.5×(logit − 10)). Votes for the same name add up. Support ≥ 0.5
gives the name; otherwise Nighthawk's group/family, otherwise "Unknown bird".
`label_support` and `models_agreeing` are stored so the team can start with the
surest clips. `period` (night / twilight / day) uses the sun's elevation at
Princeton (astral library).

### 6.9 Clips (`src/export_clips.py`)

- 0.5 s padding on both sides; detections less than 1 s apart share a clip; a
  clip never grows beyond 30 s, and single detections longer than that
  (continuous daytime song) are split.
- Clip edges are whole seconds inside the recording (start rounded down, end
  rounded up but never past the last whole second).
- Audio is the **denoised** signal (each channel separately, stereo kept),
  16-bit PCM, peak-normalized to −3 dBFS so faint calls are audible.
- Before writing a recording's clips, that recording's earlier `*_auto.WAV`
  files are deleted (only files matching `<rec>-*_auto.WAV`), so re-runs leave
  no stale clips. Originals and hand-cut clips are never touched.
- `auto_detections.csv` in each night folder and `outputs/detections.csv`
  list every detection: night, recording, clip file, clock time, period,
  start/end seconds, best label, label support, models agreeing, referee
  probability, which models triggered it, each model's guess and score.

Export of all 30 recordings takes ~3 minutes.

### 6.10 Summaries and review tools

- `src/summarize.py` → `outputs/summary_by_night.csv`, `species_counts.csv`,
  `manual_clip_check.csv` (was each hand-cut clip found, and as what).
- `src/review_sheet.py` → a sheet of spectrograms of random detections
  (`--period night`, `--label "Savannah Sparrow"`, `--n`, `--seed`) for quick
  visual checks; saved in `outputs/review/`.

### 6.11 The team page

`src/build_team_page.py` (+ `src/team_page_template.html`) builds
`outputs\team_page\night_sky_listener.html`, a single 5.8 MB file with all
audio and images embedded (works offline). It explains the steps in words for
8-11-year-olds, plays a before/after denoising example, has a gallery of seven
confident night calls (Gray-cheeked Thrush, White-throated Sparrow, Savannah
Sparrow, Swainson's Thrush, Chipping Sparrow, Ovenbird, American Redstart; each
checked by eye) with a moving playhead and a half-speed "Slow-mo" button, a
"Bird or bug?" quiz (katydid false alarm, real Savannah Sparrow, the team's
Oct 3 mystery clip), the report card, charts, a mission per team member and a
word bank. It contains the kids' first names, so it should stay inside the team.

---

## 7. Experiments and evaluation

### 7.1 Experiment 1: does denoising help the pretrained models?

Positives: the 34 hand-cut clips. Background: 400 random 2-s spans from the
same recordings (a few may hold unnoticed calls, which makes the numbers
slightly pessimistic). Score = highest score of any window overlapping the
span. "Recall at 5% / 1%" = share of clips found when the threshold flags 5% /
1% of random spans. Script: `src/experiments/exp1_denoise_effect.py`.

| Model | Audio | AUC | Recall @5% | Recall @1% |
|---|---|---|---|---|
| BirdNET (all birds) | raw | **0.800** | 0.21 | 0.09 |
| | denoised | 0.724 | 0.12 | 0.09 |
| BirdNET (local species) | raw | 0.713 | 0.24 | 0.09 |
| | denoised | 0.751 | 0.15 | 0.03 |
| Perch (bird logit) | raw | 0.854 | 0.38 | 0.03 |
| | denoised | 0.870 | 0.38 | 0.03 |
| Perch (bird share) | raw | 0.820 | 0.44 | 0.03 |
| | denoised | 0.821 | 0.56 | 0.03 |
| Nighthawk | raw | **0.865** | 0.47 | 0.32 |
| | denoised | 0.814 | 0.47 | 0.35 |

**Decision**: pretrained models get raw audio; denoising is used inside our
own detector's input and for the clips people listen to.

### 7.2 Experiment 2: our detector on the team's clips

Same spans as experiment 1 (raw audio). Script:
`src/experiments/exp2_our_models_on_manual.py`.

| Model | AUC | Recall @5% | Recall @1% |
|---|---|---|---|
| BirdNET | 0.800 | 0.21 | 0.09 |
| Perch (bird share) | 0.820 | 0.44 | 0.03 |
| Nighthawk | 0.865 | 0.47 | 0.32 |
| **Our detector** | **0.906** | **0.71** | **0.47** |

Our detector scores ~1.0 on essentially every night flight call the team cut.
Its low scores are on daytime song (Oct 5, 08:10 file), the Canada Goose honk
(low pitch, outside its training), and clips with nothing visible.

### 7.3 Report card on the expert-labelled test night (BirdVox unit 10)

"Found" = share of expert-marked calls inside a detection (±0.1 s);
"real" = share of detections containing at least one call. 6,839 calls in 8.2 h.
Full table: `outputs/experiments/report_card_birdvox_test.csv`.

| Detector | Setting | Detections | Real (precision) | Found (recall) | F1 |
|---|---|---|---|---|---|
| Our detector | p ≥ 0.3 | 6,572 | 0.730 | 0.974 | 0.835 |
| | p ≥ 0.7 | 6,047 | 0.795 | 0.955 | 0.868 |
| | p ≥ 0.9 | 5,599 | 0.847 | 0.928 | 0.886 |
| Nighthawk (1-s windows) | p ≥ 0.8 | 22,302 | 0.864 | 0.980 | 0.918 |
| Nighthawk (official settings) | threshold 80 + post-processing | 5,193 | 0.876 | 0.945 | 0.909 |
| BirdNET (local species) | p ≥ 0.1 | 1,035 | 0.868 | 0.217 | 0.347 |
| | p ≥ 0.5 | 228 | 0.969 | 0.048 | 0.092 |
| Perch v2 | bird share ≥ 0.8 | 2,248 | 0.931 | 0.590 | 0.722 |
| **Referee (used)** | **p ≥ 0.65** | **2,683** | **0.874** | **0.967** | **0.918** |

Notes:
- Overlapping 1-s windows each count as a detection, which flatters the
  window-level rows; the official Nighthawk row and the referee row are the
  fair comparison.
- BirdVox nights are dense with calls, so "real" is easier to achieve than on
  a quiet suburban night.
- Nighthawk may have been trained partly on recordings from the same area; this
  could not be confirmed (paper's data section not reachable). On the team's own
  recordings (§7.2) our detector ranks above Nighthawk.

### 7.4 Final pipeline on the team's hand-cut clips

26 of 34 found (`outputs/manual_clip_check.csv`). The 8 not found:

| Clip | Likely reason |
|---|---|
| `100326-221200-3720-3732` | spectrogram shows a broad 0-10 kHz "whoosh" for ~3 s: car, plane or wind gust, probably not a bird |
| `100326-221200-4305-4311` | mostly regular cricket pulses; one tiny 7 kHz blip at 4306.5 s too faint to judge |
| `100526-081044_band-2014-2023` | morning song; BirdNET 0.49, below its keep threshold |
| `100626-193652-5738-5743` | nothing visible on the spectrogram |
| `100726-014940-6853-6859` | faint; all models scored low |
| `100726-045604-5743-5747` (White-breasted Nuthatch) | faint dawn song |
| `100726-045604-5806-5808` (Canada Goose) | low honk, outside the flight-call pitch range |
| `100726-045604-5835-5838` | nothing visible |

Labels on the team's named clips: the two night White-throated Sparrow clips are
labelled correctly (one only after switching to the vote, §9). The two dawn
clips at ~06:31 (5734 s "White-throated Sparrow", 5757 s "Northern Cardinal")
get a neighbour's label (Northern Cardinal / Blue Jay): several birds sing at
once in the dawn chorus and the label names the strongest one in the window.

### 7.5 Visual precision check on the team's recordings

36 random night detections drawn as spectrograms (`outputs/review/review_7_night.png`):
~27 clear flight calls, 2 clear insect trains (katydid: identical chirps at a
perfectly regular interval), ~7 too faint to judge, so ≈ 75-85% are real
calls. Both insect cases scored ~1.0 in our detector and in Nighthawk.

---

## 8. Results on the team's recordings

### 8.1 Per night

| Night folder | Hours | Hours of darkness* | Detections | Clips | In darkness | Per dark hour |
|---|---|---|---|---|---|---|
| 2026_10_01 (Oct 1→2) | 8.6 | 7.5 | 235 | 183 | 111 | 14.9 |
| 2026_10_03 (Oct 3) | 2.0 | 2.0 | 210 | 185 | 210 | 105.0 |
| 2026_10_05 (Oct 5→6) | 12.1 | 8.4 | 772 | 593 | 214 | 25.3 |
| 2026_10_06 (Oct 6→7, recorder 1) | 12.8 | 10.5 | 578 | 390 | 94 | 9.0 |
| 2026_10_07 (Oct 6→7, recorder 2) | 13.2 | 10.6 | 797 | 460 | 127 | 12.0 |
| 2026_10_09 (Oct 9→10) | 18.9 | 10.7 | 487 | 424 | 46 | 4.3 |
| **Total** | **67.6** | **49.6** | **3,079** | **2,235** | **802** | |

*Sun more than 12° below the horizon. Oct 3 covers only 22:12-00:10, so its
rate is not comparable with full nights. Detections by period: 802 night, 411
twilight, 1,866 day (day files: Oct 5 morning, Oct 9 afternoon, Oct 10 morning,
and the dawn ends of other nights).

### 8.2 Most common labels in full darkness

Savannah Sparrow 166 · ZEEP warbler call group 161 · "Unknown bird" 68 ·
Passeriformes (order only) 56 · Parulidae (warbler family only) 52 · Palm
Warbler 42 · Swainson's Thrush 39 · White-throated Sparrow 25 · American
Redstart 21 · Common Yellowthroat 15 · DBUP call group 15 · Ovenbird 13 ·
Northern Parula 12 · Rose-breasted Grosbeak 12 · Cape May Warbler 10.
In full darkness, 247 of 802 labels are backed by two or more models
(106 by two, 109 by three, 32 by all four).

These fit early-October migration in central New Jersey, but each label is a
machine guess. Full counts: `outputs/species_counts.csv`.

### 8.3 Output files

| Where | What |
|---|---|
| `E:\sensor_recording\<night>\*_auto.WAV` | 2,235 clips (1.6 GB), 48 kHz 16-bit, original channel count |
| `E:\sensor_recording\<night>\auto_detections.csv` | detections for that night |
| `outputs\detections.csv` | all 3,079 detections |
| `outputs\summary_by_night.csv`, `species_counts.csv`, `manual_clip_check.csv`, `summary.txt` | summaries |
| `outputs\experiments\` | experiment and report-card tables, model summaries |
| `outputs\review\` | spectrogram review sheet |
| `outputs\team_page\night_sky_listener.html` | kids' page |
| `models\flightcall_cnn.pt`, `species_cnn.pt`, `stacker.pkl` | trained models (4.3 MB, 4.8 MB, 1 KB) |

---

## 9. Problems found along the way and how they were fixed

| Problem | Fix |
|---|---|
| 10 WAVs report 0 samples (broken header) | own RIFF parser; length from file size |
| Official Nighthawk too slow (one window per call, ~11 h total) | batched `preprocess` + network; verified identical to the CLI |
| Nighthawk wrapper re-compiled TensorFlow graphs for each new batch size | fixed `input_signature` on the `tf.function` |
| Zenodo download at ~150 KB/s (10+ h) | parallel byte-range downloader (~7 MB/s), with back-off for HTTP 503 |
| Species model stuck at ~10% accuracy | short clips were zero-padded, so the background estimate was ~0 and every pixel saturated; background now taken only from real (non-silent) frames, and bfloat16 instead of float16 |
| Denoising lowered BirdNET/Nighthawk accuracy (experiment 1) | pretrained models get raw audio |
| Detections chained into minutes-long spans | Perch's "bird share" is ≥ 0.5 on 92% of background windows; Perch no longer starts candidates; Nighthawk candidate threshold raised from 0.5 to 0.8 |
| European species (European Robin, Song Thrush, Hawfinch) in labels | Perch has no location filter; its species now must be on BirdNET's local list |
| Fixed-priority labels mislabelled dawn songs and missed sparrows | weighted vote with time-of-day weights; "night" redefined as sun < −12° |
| Clips up to 70 s; end-of-file clips shorter than their names | long detections split to ≤ 30 s clips; clip end capped at the recording's last whole second |
| Re-runs could leave stale clips | export deletes that recording's earlier `_auto.WAV` files first |
| pandas `.clip` column name clashed with `DataFrame.clip()`; nested f-string quotes invalid in Python 3.11 | column renamed `clip_file`; quotes fixed |

---

## 10. Limitations

- **Katydids** with regular chirp trains still fool the detectors (§7.5).
- **Dawn chorus**: labels name the strongest bird in the window.
- **Faint calls**: about 30% of expert-marked calls are missed under strict
  one-to-one matching on the test night, mostly very faint ones.
- **Low sounds** (geese, owls, herons) are outside our detector's 1-11 kHz
  input; they are found only when BirdNET is confident.
- **Species**: our species model knows 25 northeastern sparrows/warblers and no
  thrushes; Nighthawk often answers only at group or family level. Treat every
  name as a candidate for human checking.
- **Domain shift**: our models were trained on BirdVox (upstate New York, 2015,
  different microphones). They do well on the team's clips, but the precision
  estimate on the team's data rests on 36 checked detections.
- **Counts are detections, not birds**: one bird can call several times; a
  flock gives many calls.
- The comparison of the two Oct 6→7 recorders assumes both were working
  equally; recorder placement and gain differ.

---

## 11. How to re-run and how to add new nights

Setup (once):

```bash
cd C:\Users\susan\CodingProjects\BirdSong_Recognition
uv venv --python 3.11 .venv
VIRTUAL_ENV=.venv uv pip install "birdnet_analyzer[all]" soundfile librosa matplotlib astral pandas scikit-learn h5py
VIRTUAL_ENV=.venv uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
VIRTUAL_ENV=.venv uv pip install --no-deps nighthawk
python data/download.py          # only needed to retrain
```

New recordings: copy the night's folder into `E:\sensor_recording`
(e.g. `2026_10_12`), then:

```bash
.venv/Scripts/python src/scan.py --workers 2    # ~2.5 min per recorded hour (CPU)
.venv/Scripts/python src/scan.py --cnn          # our detector (GPU, seconds)
.venv/Scripts/python src/export_clips.py        # clips + tables (all nights, ~3 min)
.venv/Scripts/python src/summarize.py
.venv/Scripts/python src/build_team_page.py     # refresh the kids' page
```

Already-scanned recordings are skipped. To retrain: `train_flightcall_cnn.py`
(~25 min), `train_species_cnn.py` (~5 min), then `scan.py --cnn --force`,
`scan.py --birdvox 07 10` (once), `train_stacker.py`, `export_clips.py`.

Settings worth knowing (`src/config.py`, `src/detect.py`, `src/export_clips.py`):
station latitude/longitude, high-pass frequency, clip naming/padding/merge
gap/maximum length/loudness, candidate thresholds, BirdNET keep threshold,
vote weights.

---

## 12. File inventory

```
BirdSong_Recognition/
├── README.md                     short overview + results
├── docs/PROJECT_DOCUMENTATION.md this document
├── data/download.py              dataset downloader (data itself git-ignored)
├── models/                       flightcall_cnn.pt, species_cnn.pt, stacker.pkl
├── outputs/
│   ├── detections.csv, summary_by_night.csv, species_counts.csv,
│   │   manual_clip_check.csv, summary.txt
│   ├── experiments/              exp1/exp2 tables, report card, model summaries
│   ├── review/                   spectrogram review sheet
│   ├── team_page/night_sky_listener.html
│   └── cache/                    per-recording model scores (git-ignored, ~1 GB)
└── src/
    ├── config.py                 settings
    ├── audio_io.py               recordings, hand-cut clips, safe WAV reading
    ├── denoise.py                GPU spectral gate + high-pass
    ├── pretrained.py             BirdNET, Perch v2, batched Nighthawk
    ├── flightcall_cnn.py         SNR front-end + detector network + event picking
    ├── train_flightcall_cnn.py   detector training/evaluation on BirdVox
    ├── species_cnn.py            species classifier
    ├── train_species_cnn.py      species training on BirdVox-ANAFCC
    ├── scan.py                   stage 1: run all models, cache scores
    ├── detect.py                 candidates, evidence, label helpers
    ├── train_stacker.py          referee + report card
    ├── export_clips.py           stage 3: events, labels, clips, tables
    ├── summarize.py              summaries
    ├── review_sheet.py           spectrogram sheets
    ├── build_team_page.py        kids' page builder
    ├── team_page_template.html   kids' page template
    └── experiments/              exp1_denoise_effect.py, exp2_our_models_on_manual.py
```

Git history (local): initial pipeline → models and experiments → candidate
rules and summaries → vote labels, clip fixes, results → model weights →
documentation and team page.

---

## 13. Suggested next steps

1. **Kids' review** (the plan's "AI report card"): check ~100 clips sorted by
   `bird_prob` and `models_agreeing`, mark ✅ / 🤔 / ❌, and compute the real
   precision on the team's data.
2. **Feed the review back**: the checked clips are new labelled examples from the
   team's own microphones. A short fine-tuning of our detector and species model
   on them should reduce katydid false alarms.
3. **Katydid filter**: flag detections that repeat at a near-constant interval.
4. **Compare with BirdCast**: log Mercer County radar numbers per night and
   compare with "calls per dark hour" (§8.1).
5. **Two-station comparison** (Oct 6→7): confirm which folder is which site
   (dark vs. bright) before drawing conclusions.
6. **eBird upload** with the Nocturnal Flight Call protocol for confirmed calls.
7. **Back-up**: push the repository to a private GitHub repository.

---

## 14. Credits, licenses and references

- BirdVox-full-night: V. Lostanlen, J. Salamon, A. Farnsworth, S. Kelling,
  J. P. Bello (2018). Zenodo record 1205569. CC-BY-4.0.
- BirdVox-ANAFCC: A. Cramer, V. Lostanlen, A. Farnsworth, J. Salamon,
  J. P. Bello (2020). Zenodo record 3666782. CC-BY-4.0.
- BirdNET-Analyzer and BirdNET V2.4: S. Kahl et al., Cornell Lab of
  Ornithology and Chemnitz University of Technology.
  https://github.com/birdnet-team/BirdNET-Analyzer
- Perch v2: Google DeepMind (Kaggle model
  `google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu`).
- Nighthawk: B. M. Van Doren et al. (2024), *Methods in Ecology and Evolution*
  15(2). https://github.com/bmvandoren/Nighthawk
- Sun position: astral. Deep learning: PyTorch, TensorFlow.
