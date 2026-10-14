"""Find recordings and read them safely.

The Zoom recorder writes 32-bit float WAVs. When it stops abruptly (battery,
power-off) it leaves the size fields in the header at 0, so many programs
(including the `soundfile` library) think those files are empty. We parse the
header ourselves and take the length of the audio from the file size instead.
Original files are only ever opened read-only.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# Full recordings look like 100626-193652.WAV (MMDDYY-HHMMSS, recording start).
# Hand-cut clips have extra parts in the name and are ignored.
RECORDING_RE = re.compile(r"^(\d{6})-(\d{6})\.WAV$", re.IGNORECASE)
# Hand-cut clips: 100626-193652-3227-3229.wav, 100526-081044_band-675-684.wav,
# 100726-014940-2256-2262-White-throated Sparrow.wav
MANUAL_CLIP_RE = re.compile(
    r"^(\d{6}-\d{6})[-_](?:band-)?(\d+)-(\d+)(?:-(.+))?\.wav$", re.IGNORECASE)

_FORMAT_DTYPES = {(3, 32): np.float32, (1, 16): np.int16, (1, 32): np.int32}


@dataclass
class Recording:
    path: Path
    sample_rate: int
    channels: int
    frames: int
    start_time: datetime        # local clock time from the file name
    _data_offset: int
    _dtype: type

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate

    @property
    def night(self) -> str:
        """Folder name, e.g. 2026_10_06 (one folder per recording session)."""
        return self.path.parent.name

    def time_at(self, seconds: float) -> datetime:
        return self.start_time + timedelta(seconds=seconds)

    def memmap(self) -> np.ndarray:
        """All samples as a read-only (frames, channels) array backed by the file."""
        return np.memmap(self.path, dtype=self._dtype, mode="r",
                         offset=self._data_offset,
                         shape=(self.frames, self.channels))

    def read(self, start_s: float = 0.0, dur_s: float | None = None,
             mono: bool = True) -> np.ndarray:
        """Read a stretch of audio as float32 (mono by averaging channels)."""
        a = int(max(0, round(start_s * self.sample_rate)))
        b = self.frames if dur_s is None else min(
            self.frames, a + int(round(dur_s * self.sample_rate)))
        x = np.asarray(self.memmap()[a:b], dtype=np.float32)
        if self._dtype is np.int16:
            x /= 32768.0
        elif self._dtype is np.int32:
            x /= 2147483648.0
        return x.mean(axis=1) if mono else x


def open_recording(path: str | Path) -> Recording:
    path = Path(path)
    with open(path, "rb") as fh:
        head = fh.read(1 << 20)
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a WAV file")
    pos, fmt, data_offset = 12, None, None
    while pos + 8 <= len(head):
        cid = head[pos:pos + 4]
        size = struct.unpack("<I", head[pos + 4:pos + 8])[0]
        if cid == b"fmt ":
            fmt = list(struct.unpack("<HHIIHH", head[pos + 8:pos + 24]))
            if fmt[0] == 0xFFFE:   # WAVE_FORMAT_EXTENSIBLE: real tag is in the subformat GUID
                fmt[0] = struct.unpack("<H", head[pos + 32:pos + 34])[0]
        elif cid == b"data":
            data_offset = pos + 8
            break
        pos += 8 + size + (size & 1)
    if fmt is None or data_offset is None:
        raise ValueError(f"{path}: could not find fmt/data chunks")
    fmt_tag, channels, sr, _, block_align, bits = fmt
    dtype = _FORMAT_DTYPES[(fmt_tag, bits)]
    frames = (path.stat().st_size - data_offset) // block_align   # ignore header size
    start = None
    m = RECORDING_RE.match(path.name)
    if m:
        start = datetime.strptime(m.group(1) + m.group(2), "%m%d%y%H%M%S")
    return Recording(path, sr, channels, frames, start, data_offset, dtype)


def find_recordings(root: str | Path) -> list[Recording]:
    """All full-length recordings under root (hand-cut clips are skipped)."""
    files = [p for p in Path(root).rglob("*") if RECORDING_RE.match(p.name)]
    recs = [open_recording(p) for p in files]
    return sorted(recs, key=lambda r: (r.night, r.start_time))


@dataclass
class ManualClip:
    path: Path
    source_stem: str
    start_s: int
    end_s: int
    label: str | None
    band_filtered: bool


def find_manual_clips(root: str | Path) -> list[ManualClip]:
    clips = []
    for p in Path(root).rglob("*.wav"):
        if RECORDING_RE.match(p.name) or p.stem.endswith("_auto"):
            continue
        m = MANUAL_CLIP_RE.match(p.name)
        if m:
            clips.append(ManualClip(p, m.group(1), int(m.group(2)), int(m.group(3)),
                                    m.group(4), "_band-" in p.name))
    return sorted(clips, key=lambda c: (c.source_stem, c.start_s))
