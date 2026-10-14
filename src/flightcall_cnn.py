"""Our own flight-call detector: a small convolutional + recurrent network.

Input ("SNR map"): for every 10.7 ms frame and each of 64 pitch bands between
1 and 11 kHz, how many dB the sound is above that band's usual background over
the surrounding ~10 s. Crickets and road noise sit near 0 dB; a call pops out
at 10-40 dB. Because it is relative, it doesn't matter how loud a recorder is.

Output: probability, for every frame, that a flight call is happening.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 24000            # all audio is resampled to 24 kHz for this model
N_FFT, HOP = 512, 256
FRAME_S = HOP / SR    # 10.67 ms
N_MELS, FMIN, FMAX = 64, 1000.0, 11000.0
SNR_MAX_DB = 40.0


def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX) -> torch.Tensor:
    import librosa
    return torch.tensor(librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
                                            fmin=fmin, fmax=fmax), dtype=torch.float32)


class SNRFrontend:
    """Waveform (24 kHz) -> (64 x frames) map of dB above local background."""

    def __init__(self, device="cuda", noise_window_s=10.0, block_s=0.5):
        self.device = device
        self.window = torch.hann_window(N_FFT, device=device)
        self.mel = mel_filterbank().to(device)
        self.block = int(round(block_s / FRAME_S))
        self.n_blocks = int(round(noise_window_s / block_s)) | 1
        self.context_s = noise_window_s / 2 + 1.0

    @torch.no_grad()
    def __call__(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        spec = torch.stft(t, N_FFT, HOP, window=self.window, return_complex=True)
        mel = self.mel @ (spec.abs() ** 2) + 1e-12            # 64 x T
        n_mel, n = mel.shape
        nb = -(-n // self.block)
        p = F.pad(mel, (0, nb * self.block - n), mode="replicate")
        bm = p.reshape(n_mel, nb, self.block).median(dim=2).values
        k = self.n_blocks
        bm = F.pad(bm[None], (k // 2, k // 2), mode="replicate")[0]
        floor = bm.unfold(1, k, 1).median(dim=2).values
        floor = F.interpolate(floor[None], size=nb * self.block, mode="linear",
                              align_corners=False)[0, :, :n]
        snr = 10 * torch.log10(mel / floor.clamp_min(1e-12))
        return (snr.clamp(0, SNR_MAX_DB) / SNR_MAX_DB).half()

    def frames_to_s(self, i):
        return np.asarray(i) * FRAME_S


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, fpool):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.fpool = fpool

    def forward(self, x):
        x = F.relu(self.b1(self.c1(x)))
        x = F.relu(self.b2(self.c2(x)))
        return F.max_pool2d(x, (self.fpool, 1))     # pool pitch only, keep time detail


class FlightCallNet(nn.Module):
    """(batch, 64 mel, T) -> (batch, T) logits."""

    def __init__(self, width=32, rnn=128, dropout=0.2):
        super().__init__()
        w = width
        self.blocks = nn.Sequential(ConvBlock(1, w, 2), ConvBlock(w, 2 * w, 2),
                                    ConvBlock(2 * w, 4 * w, 2), ConvBlock(4 * w, 4 * w, 2))
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(4 * w * (N_MELS // 16), rnn, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * rnn, 1)

    def forward(self, x):
        h = self.blocks(x[:, None].float())                 # B, C, 4, T
        h = h.flatten(1, 2).transpose(1, 2)                  # B, T, C*4
        h, _ = self.gru(self.drop(h))
        return self.head(self.drop(h))[..., 0]


def load_model(path, device="cuda") -> FlightCallNet:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = FlightCallNet(**ck.get("hparams", {})).to(device)
    m.load_state_dict(ck["state_dict"])
    return m.eval()


@torch.no_grad()
def frame_probs(model: FlightCallNet, snr: torch.Tensor, chunk=4096, overlap=256) -> np.ndarray:
    """Run the network over a long SNR map in overlapping chunks."""
    n = snr.shape[1]
    out = torch.zeros(n, device=snr.device)
    start = 0
    while start < n:
        a = max(0, start - overlap)
        b = min(n, start + chunk + overlap)
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.sigmoid(model(snr[None, :, a:b]))[0].float()
        out[start:min(n, start + chunk)] = p[start - a:start - a + min(chunk, n - start)]
        start += chunk
    return out.cpu().numpy()


def pick_events(p: np.ndarray, threshold=0.5, min_frames=3, merge_gap_frames=5):
    """Frame probabilities -> list of (start_s, end_s, peak_s, peak_prob)."""
    on = p >= threshold
    if not on.any():
        return []
    idx = np.flatnonzero(np.diff(np.r_[0, on.astype(np.int8), 0]))
    starts, ends = idx[::2], idx[1::2]
    merged = []
    for s, e in zip(starts, ends):
        if merged and s - merged[-1][1] <= merge_gap_frames:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    events = []
    for s, e in merged:
        if e - s < min_frames:
            continue
        k = s + int(np.argmax(p[s:e]))
        events.append((s * FRAME_S, e * FRAME_S, k * FRAME_S, float(p[k])))
    return events
