"""Our own flight-call species classifier.

Trained on BirdVox-ANAFCC (Cramer et al. 2020, CC-BY-4.0): ~85,000 short clips
of night flight calls from the American Northeast, mostly sparrows and
warblers, plus noise / "tone" false alarms. Input is a 0.6-s window centered on
a call, turned into the same "dB above background" map the detector uses, but
with the background taken from the window itself (so it works on short clips).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flightcall_cnn import HOP, N_FFT, SNR_MAX_DB, SR, ConvBlock, mel_filterbank

CROP_S = 0.6
CROP_N = int(round(CROP_S * SR))

# 4-letter bird-banding codes -> English names (only the codes we train on).
CODE_NAMES = {
    "WTSP": "White-throated Sparrow", "SVSP": "Savannah Sparrow", "CHSP": "Chipping Sparrow",
    "ATSP": "American Tree Sparrow", "GHSP": "Grasshopper Sparrow", "INBU": "Indigo Bunting",
    "RBGR": "Rose-breasted Grosbeak", "AMRE": "American Redstart",
    "BTBW": "Black-throated Blue Warbler", "CAWA": "Canada Warbler",
    "COYE": "Common Yellowthroat", "MOWA": "Mourning Warbler", "OVEN": "Ovenbird",
    "YRWA": "Yellow-rumped Warbler", "PAWA": "Palm Warbler", "CSWA": "Chestnut-sided Warbler",
    "CMWA": "Cape May Warbler", "NWTH": "Northern Waterthrush", "BAWW": "Black-and-white Warbler",
    "WIWA": "Wilson's Warbler", "HOWA": "Hooded Warbler", "MAWA": "Magnolia Warbler",
    "BTNW": "Black-throated Green Warbler", "NAWA": "Nashville Warbler",
    "TEWA": "Tennessee Warbler", "BLBW": "Blackburnian Warbler", "NOPA": "Northern Parula",
    "Zeep": "Zeep-type call (warbler, species unknown)",
    "ShDbUp": "Short double-banded upsweep (sparrow/bunting, species unknown)",
    "NOISE": "Not a bird (noise / insect / tone)",
}


class CropFrontend:
    """0.6-s waveform crops (batch, CROP_N) at 24 kHz -> (batch, 64, frames)."""

    def __init__(self, device="cuda"):
        self.device = device
        self.window = torch.hann_window(N_FFT, device=device)
        self.mel = mel_filterbank().to(device)

    @torch.no_grad()
    def __call__(self, x) -> torch.Tensor:
        t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        spec = torch.stft(t, N_FFT, HOP, window=self.window, return_complex=True)
        mel = torch.einsum("mf,bft->bmt", self.mel, spec.abs() ** 2) + 1e-12
        # Short clips are zero-padded; take the background only from real audio.
        silent = mel.sum(dim=1, keepdim=True) < 1e-9
        floor = torch.where(silent, torch.nan, mel).nanmedian(dim=2, keepdim=True).values
        floor = torch.nan_to_num(floor, nan=1.0)
        snr = 10 * torch.log10(mel / floor)
        return snr.clamp(0, SNR_MAX_DB) / SNR_MAX_DB


class SpeciesNet(nn.Module):
    def __init__(self, n_classes: int, width=32, dropout=0.3):
        super().__init__()
        w = width
        self.blocks = nn.Sequential(ConvBlock(1, w, 2), ConvBlock(w, 2 * w, 2),
                                    ConvBlock(2 * w, 4 * w, 2), ConvBlock(4 * w, 8 * w, 2))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(8 * w * 2, n_classes)

    def forward(self, x):
        h = self.blocks(x[:, None].float())              # B, C, 4, T
        h = torch.cat([h.amax(dim=(2, 3)), h.mean(dim=(2, 3))], 1)
        return self.head(self.drop(h))


def center_crop(x: np.ndarray, sr=SR) -> np.ndarray:
    """Crop/pad a clip to CROP_S seconds around its loudest 50 ms (in 1-11 kHz)."""
    from scipy.signal import butter, sosfiltfilt
    if len(x) < CROP_N:
        x = np.pad(x, ((CROP_N - len(x)) // 2, CROP_N - len(x) - (CROP_N - len(x)) // 2))
    if len(x) == CROP_N:
        return x.astype(np.float32)
    sos = butter(4, [1000, 11000], "bandpass", fs=sr, output="sos")
    e = np.convolve(sosfiltfilt(sos, x) ** 2, np.ones(int(0.05 * sr)), "same")
    c = int(np.argmax(e))
    a = int(np.clip(c - CROP_N // 2, 0, len(x) - CROP_N))
    return x[a:a + CROP_N].astype(np.float32)


def load_model(path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = SpeciesNet(len(ck["classes"])).to(device)
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck["classes"]


@torch.no_grad()
def classify(model, frontend, crops: np.ndarray, batch=512) -> np.ndarray:
    """crops: (n, CROP_N) at 24 kHz -> (n, classes) probabilities."""
    out = []
    for i in range(0, len(crops), batch):
        out.append(F.softmax(model(frontend(crops[i:i + batch])), 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(model.head.weight)))
