"""Remove steady background noise (crickets, katydids, traffic rumble).

Idea in plain words: a flight call lasts a tenth of a second, while a cricket
chorus or road hum stays at the same pitch for many seconds. For every pitch
(frequency bin) we estimate the "usual" loudness over the surrounding ~10 s
using a median, which a short bird call can't move. Sound that rises clearly
above that usual level is kept; everything else is turned down by ~30 dB.
A high-pass filter also removes the low rumble (< 1 kHz) from cars and wind.

Runs on the GPU with PyTorch; a full 3-hour file takes a few seconds.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class Denoiser:
    def __init__(self, sample_rate: int = 48000, n_fft: int = 1024, hop: int = 256,
                 highpass_hz: float = 1000.0, noise_window_s: float = 10.0,
                 block_s: float = 0.5, gate_low_db: float = 4.0,
                 gate_high_db: float = 10.0, floor_db: float = -30.0,
                 device: str | None = None):
        self.sr, self.n_fft, self.hop = sample_rate, n_fft, hop
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.window = torch.hann_window(n_fft, device=self.device)
        freqs = torch.fft.rfftfreq(n_fft, 1 / sample_rate).to(self.device)
        # Smooth 300 Hz ramp so the high-pass doesn't ring.
        self.hp = ((freqs - (highpass_hz - 300)) / 300).clamp(0, 1)[:, None]
        self.block = max(1, int(round(block_s * sample_rate / hop)))
        self.n_blocks = max(3, int(round(noise_window_s / block_s)) | 1)   # odd
        self.lo, self.hi = gate_low_db, gate_high_db
        self.floor = 10 ** (floor_db / 20)
        # Context (seconds) read on each side of a chunk so the noise estimate is
        # the same no matter where chunk boundaries fall.
        self.context_s = noise_window_s / 2 + 1.0

    @torch.no_grad()
    def _noise_floor(self, power: torch.Tensor) -> torch.Tensor:
        """Per-bin running median of power over ~noise_window_s (F x T)."""
        n_freq, n_frames = power.shape
        nb = -(-n_frames // self.block)
        pad = nb * self.block - n_frames
        p = F.pad(power, (0, pad), mode="replicate") if pad else power
        block_med = p.reshape(n_freq, nb, self.block).median(dim=2).values
        k = self.n_blocks
        bm = F.pad(block_med[None], (k // 2, k // 2), mode="replicate")[0]
        running = bm.unfold(1, k, 1).median(dim=2).values               # F x nb
        floor = F.interpolate(running[None], size=nb * self.block, mode="linear",
                              align_corners=False)[0, :, :n_frames]
        return floor.clamp_min(1e-20)

    @torch.no_grad()
    def gain_mask(self, spec: torch.Tensor) -> torch.Tensor:
        power = spec.abs() ** 2
        # Light smoothing before comparing to the floor reduces "musical noise".
        smooth = F.avg_pool2d(power[None, None], (3, 3), 1, (1, 1),
                              count_include_pad=False)[0, 0]
        snr_db = 10 * torch.log10(smooth / self._noise_floor(power))
        g = ((snr_db - self.lo) / (self.hi - self.lo)).clamp(0, 1)
        g = F.avg_pool2d(g[None, None], (3, 5), 1, (1, 2),
                         count_include_pad=False)[0, 0]
        return (self.floor + (1 - self.floor) * g) * self.hp

    @torch.no_grad()
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Denoise a mono float32 signal; output has the same length."""
        t = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32,
                            device=self.device)
        spec = torch.stft(t, self.n_fft, self.hop, window=self.window,
                          return_complex=True)
        y = torch.istft(spec * self.gain_mask(spec), self.n_fft, self.hop,
                        window=self.window, length=len(t))
        return y.cpu().numpy()

    def process_recording(self, rec, chunk_s: float = 300.0):
        """Yield (start_s, denoised mono chunk) over a whole Recording."""
        ctx = self.context_s
        start = 0.0
        while start < rec.duration:
            a = max(0.0, start - ctx)
            seg = rec.read(a, (start - a) + chunk_s + ctx)
            out = self(seg)
            lo = int(round((start - a) * self.sr))
            hi = lo + int(round(min(chunk_s, rec.duration - start) * self.sr))
            yield start, out[lo:hi]
            start += chunk_s
