"""Project-wide settings. Change values here rather than inside the scripts."""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = Path("E:/sensor_recording")
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
CACHE_DIR = OUTPUT_DIR / "cache"          # per-recording model scores (large, git-ignored)
MODEL_DIR = PROJECT_DIR / "models"        # weights of the models we train ourselves

# Station location (Princeton / Plainsboro, NJ). Used to judge which species are
# plausible here in early October and to compute sunset/sunrise.
LATITUDE, LONGITUDE = 40.35, -74.62
TIMEZONE = "America/New_York"

# Denoising (see denoise.py)
HIGHPASS_HZ = 1000.0

# Output clip naming. Example: 100626-193652-3227-3229_auto.WAV
# The prefix matches the hand-cut clips (recording name - start s - end s),
# with "_auto" so automatic clips are easy to tell apart from manual ones.
CLIP_NAME = "{stem}-{start}-{end}_auto.WAV"
CLIP_PAD_S = 0.5            # silence-free context kept before/after a detection
CLIP_MERGE_GAP_S = 1.0      # detections closer than this go into one clip
CLIP_MAX_S = 30.0           # long bouts (e.g. dawn chorus) are split
CLIP_PEAK_DBFS = -3.0       # clips are loudness-normalized so faint calls are audible
