# Night Sky Listener: automatic bird-call finder

Finds nocturnal flight calls (and other bird sounds) in the Tantalum 73 night
recordings under `E:\sensor_recording`, names the likely species, and saves each
detection as a short clean clip next to the original file:

```
E:\sensor_recording\2026_10_06\100626-193652-6773-6774.wav          <- cut by hand
E:\sensor_recording\2026_10_06\100626-193652-6772-6775_auto.WAV     <- cut by this tool
E:\sensor_recording\2026_10_06\auto_detections.csv                  <- what is in each clip
```

`<recording>-<start s>-<end s>_auto.WAV`: start/end are whole seconds from the
start of the original recording, same as the hand-cut clips. The pattern is set
in one place (`CLIP_NAME` in `src/config.py`) if the team prefers another.

## How it works (plain-language version)

```
 long WAV ──► denoise ──► 4 detectors ──► referee ──► species guess ──► _auto.WAV clips
                          (each scores                (4 opinions,      + detections table
                           every moment)               one best label)
```

1. **Denoise** (`src/denoise.py`). A flight call lasts about a tenth of a second;
   crickets, katydids and road noise sit at the same pitch for many seconds. For
   each pitch we track the usual background level over the surrounding ~10 s and
   turn down anything that isn't clearly above it. A high-pass filter removes
   the rumble below 1 kHz.
2. **Detectors**, each scoring the whole night:

   | Model | Who made it | What it's good at |
   |---|---|---|
   | **Our flight-call detector** (`flightcall_cnn.py`) | trained here on BirdVox-full-night | short night flight calls, precise timing (every 10 ms) |
   | Nighthawk | Van Doren et al. 2024 | night flight calls, 82 species / groups |
   | BirdNET V2.4 | Cornell Lab + TU Chemnitz | songs and daytime calls, 6,500 species |
   | Perch v2 | Google DeepMind | songs; also knows insects and frogs |

3. **Referee** (`train_stacker.py`). A logistic-regression model learns from a
   fully labelled BirdVox night how much to trust each detector, and turns
   their scores into one probability that a stretch holds a bird call.
   BirdNET's confident detections of locally expected species are always kept,
   because BirdVox only labels flight calls, not songs.
4. **Species**. Four opinions are kept side by side in the table: Nighthawk,
   **our species classifier** (`species_cnn.py`, trained on BirdVox-ANAFCC,
   ~25 sparrows and warblers), BirdNET and Perch. `best_label` picks one: at
   night the flight-call specialists come first; in daytime, BirdNET.
5. **Clips** (`export_clips.py`). Detections less than 1 s apart are merged,
   0.5 s of padding is added, and the stretch is cut from the **denoised**
   audio, peak-normalized to -3 dBFS so faint calls are audible.

## Results

_Filled in after the full run; see "Results" below._

## Things we learned

* **Denoising helps people, not the pretrained AIs.** On the team's hand-cut
  clips, BirdNET and Nighthawk scored *worse* on denoised audio (they learned
  from ordinary noisy recordings), so they get the raw audio. Denoising is used
  inside our own detector and for the clips people listen to.
  (`outputs/experiments/exp1_summary.csv`)
* **Ten recordings have a broken WAV header** (the recorder stopped abruptly,
  so the header says "0 samples"). Many programs open them as empty. The audio
  is intact; `src/audio_io.py` reads it by file size. Affected:
  `2026_10_07/100626-221553 … 100726-073505` and `2026_10_09/100926-182605 … 101026-095805`.
* One hand-cut clip name looks like a typo: `100726-045604-5749-5793.wav` is
  4.2 s long, so it is probably 5749-5753.

## Re-running

```bash
# one-time setup (two environments: BirdNET needs Python 3.11+, Nighthawk 3.10)
uv venv --python 3.11 .venv && VIRTUAL_ENV=.venv uv pip install "birdnet_analyzer[all]" soundfile librosa matplotlib astral pandas scikit-learn h5py
VIRTUAL_ENV=.venv uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
VIRTUAL_ENV=.venv uv pip install --no-deps nighthawk
python data/download.py                      # BirdVox training data (~8 GB, CC-BY-4.0)

# train our two models (GPU, ~25 min + ~10 min)
.venv/Scripts/python src/train_flightcall_cnn.py
.venv/Scripts/python src/train_species_cnn.py

# score everything (CPU, ~3 h for 68 h of audio), then our detector (GPU, ~10 min)
.venv/Scripts/python src/scan.py --workers 2
.venv/Scripts/python src/scan.py --birdvox 07 10
.venv/Scripts/python src/scan.py --cnn && .venv/Scripts/python src/scan.py --cnn --birdvox 07 10

.venv/Scripts/python src/train_stacker.py    # referee + report card
.venv/Scripts/python src/export_clips.py     # clips + detections.csv
.venv/Scripts/python src/summarize.py        # per-night / per-species tables
.venv/Scripts/python src/review_sheet.py     # spectrogram sheet of random detections
```

New recordings: drop them into a new night folder under `E:\sensor_recording`
and re-run `scan.py`, `scan.py --cnn`, `export_clips.py`, `summarize.py`;
already-scanned files are skipped.

## Credits and licenses

* BirdVox-full-night: Lostanlen et al. 2018, Zenodo 1205569, CC-BY-4.0.
* BirdVox-ANAFCC: Cramer et al. 2020, Zenodo 3666782, CC-BY-4.0.
* BirdNET-Analyzer (Kahl et al.), Perch v2 (Google DeepMind), Nighthawk
  (Van Doren et al. 2024): used as published, through their own packages.
