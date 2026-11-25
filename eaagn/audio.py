"""Tolerant WAV I/O and causal log-mel front end.

SPRSound WAVs carry a malformed fmt chunk (nBlockAlign=4 for 16-bit mono), which
scipy rejects; we parse the RIFF chunks ourselves and trust channels/bits only.
"""
from __future__ import annotations

import struct

import numpy as np
import torch
import torch.nn as nn

SR = 8000          # SPRSound native rate; we never upsample
HOP = 80           # 10 ms frame advance
WINS = (200, 800)  # 25 ms and 100 ms causal analysis windows
N_MELS = 128
STEP = 4           # graph step = 4 frames = 40 ms


def read_wav(path: str):
    with open(path, "rb") as f:
        b = f.read()
    if b[:4] != b"RIFF" or b[8:12] != b"WAVE":
        raise ValueError(f"not a RIFF/WAVE file: {path}")
    pos, ch, sr, bits, data = 12, 1, None, 16, None
    while pos + 8 <= len(b):
        cid, size = b[pos:pos + 4], struct.unpack("<I", b[pos + 4:pos + 8])[0]
        body = b[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt, ch, sr = struct.unpack("<HHI", body[:8])
            bits = struct.unpack("<H", body[14:16])[0]
        elif cid == b"data":
            data = body
        pos += 8 + size + (size & 1)
    if data is None or sr is None:
        raise ValueError(f"missing fmt/data chunk: {path}")
    if bits == 16:
        x = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
    elif bits == 32:
        x = np.frombuffer(data[: len(data) // 4 * 4], dtype="<i4").astype(np.float32) / 2147483648.0
    elif bits == 8:
        x = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported bit depth {bits}: {path}")
    if ch > 1:
        x = x[: len(x) // ch * ch].reshape(-1, ch).mean(1)
    return x, sr


def mel_fbank(n_fft: int, sr: int = SR, n_mels: int = N_MELS, fmin: float = 50.0, fmax: float | None = None):
    fmax = fmax or sr / 2
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel2hz = lambda m: 700.0 * (10 ** (m / 2595.0) - 1.0)
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for m in range(n_mels):
        lo, c, hi = pts[m], pts[m + 1], pts[m + 2]
        up = (freqs - lo) / max(c - lo, 1e-9)
        down = (hi - freqs) / max(hi - c, 1e-9)
        fb[m] = np.maximum(0.0, np.minimum(up, down))
        if fb[m].sum() == 0:  # very narrow low bands: nearest bin
            fb[m, np.argmin(np.abs(freqs - c))] = 1.0
    return torch.from_numpy(fb)


class CausalLogMel(nn.Module):
    """Two-resolution log-mel where frame t only uses samples ending at (t+1)*HOP.

    Output (B, 2, N_MELS, T) with T = n_samples // HOP. Normalised with fixed
    training-set statistics (never per-recording, which would peek at the future).
    """

    def __init__(self, mean=None, std=None):
        super().__init__()
        for k, w in enumerate(WINS):
            self.register_buffer(f"win{k}", torch.hann_window(w, periodic=True))
            self.register_buffer(f"fb{k}", mel_fbank(w))
        self.register_buffer("mean", torch.zeros(2, N_MELS, 1) if mean is None else torch.as_tensor(mean).view(2, N_MELS, 1).float())
        self.register_buffer("std", torch.ones(2, N_MELS, 1) if std is None else torch.as_tensor(std).view(2, N_MELS, 1).float())

    def raw(self, wav: torch.Tensor) -> torch.Tensor:
        T = wav.shape[-1] // HOP
        wav = wav[..., : T * HOP]
        outs = []
        for k, w in enumerate(WINS):
            x = torch.nn.functional.pad(wav, (w - HOP, 0))
            fr = x.unfold(-1, w, HOP)                     # (B, T, w)
            spec = torch.fft.rfft(fr * getattr(self, f"win{k}"), n=w).abs() ** 2
            mel = spec @ getattr(self, f"fb{k}").T        # (B, T, M)
            outs.append(torch.log(mel + 1e-8).transpose(-1, -2))
        return torch.stack(outs, 1)                       # (B, 2, M, T)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return (self.raw(wav) - self.mean) / self.std
