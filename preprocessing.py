"""Audio loading, deterministic condition transforms, and chunking."""
from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import BinaryIO, Callable
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

LOG = logging.getLogger(__name__)
Condition = Callable[[np.ndarray, int, np.random.Generator], tuple[np.ndarray, int]]

def resample_audio(audio: np.ndarray, old_sr: int, new_sr: int) -> np.ndarray:
    """High quality rational polyphase resampling."""
    if old_sr == new_sr:
        return np.asarray(audio, dtype=np.float32)
    from math import gcd
    div = gcd(int(old_sr), int(new_sr))
    return resample_poly(audio, new_sr // div, old_sr // div).astype(np.float32)

def load_audio(path: str | Path | BinaryIO, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load float32 mono audio, resampling to target_sr when necessary."""
    audio, sr = sf.read(path if hasattr(path, 'read') else str(path), dtype='float32', always_2d=True)
    audio = audio.mean(axis=1, dtype=np.float32)
    if sr != target_sr:
        LOG.info('Resampling %s from %d Hz to %d Hz', path, sr, target_sr)
        audio = resample_audio(audio, sr, target_sr)
        sr = target_sr
    return np.clip(audio, -1.0, 1.0).astype(np.float32), int(sr)

def clean(audio: np.ndarray, sr: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    del rng
    return audio.copy(), sr

def sr8k(audio: np.ndarray, sr: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    del rng
    if sr != 16000:
        raise ValueError('sr8k condition requires 16 kHz input')
    return resample_audio(audio, sr, 8000), 8000

def quant8(audio: np.ndarray, sr: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    del rng
    q = np.round((np.clip(audio, -1, 1) + 1) * 127.5) / 127.5 - 1
    return q.astype(np.float32), sr

def noise_snr(audio: np.ndarray, sr: int, rng: np.random.Generator, target_db: float) -> tuple[np.ndarray, int]:
    """Add Gaussian white noise at target signal-to-noise ratio."""
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    if rms == 0:
        return audio.copy(), sr
    noise_rms = rms / (10.0 ** (target_db / 20.0))
    noise = rng.normal(0.0, noise_rms, size=audio.shape)
    return np.clip(audio + noise, -1.0, 1.0).astype(np.float32), sr

def parse_condition(condition: str) -> list[tuple[str, float | None]]:
    """Parse expressions such as clean or sr8k+noise_snr10+quant8."""
    parsed = []
    for item in condition.split('+'):
        if item in {'clean', 'sr8k', 'quant8'}:
            parsed.append((item, None))
        else:
            match = re.fullmatch(r'noise_snr(-?\d+(?:\.\d+)?)', item)
            if not match:
                raise ValueError(f'Unknown condition component: {item}')
            parsed.append(('noise_snr', float(match.group(1))))
    if not parsed:
        raise ValueError('Condition cannot be empty')
    return parsed

def apply_condition(audio: np.ndarray, sr: int, condition: str, seed: int) -> tuple[np.ndarray, int]:
    """Apply a composable condition in written order, deterministically."""
    rng = np.random.default_rng(seed)
    out = np.asarray(audio, dtype=np.float32).copy()
    for name, value in parse_condition(condition):
        if name == 'clean':
            out, sr = clean(out, sr, rng)
        elif name == 'sr8k':
            out, sr = sr8k(out, sr, rng)
        elif name == 'quant8':
            out, sr = quant8(out, sr, rng)
        else:
            out, sr = noise_snr(out, sr, rng, float(value))
    return out, sr

def segment_audio(audio: np.ndarray, sr: int, chunk_seconds: float = 5.0, hop_seconds: float = 5.0) -> tuple[np.ndarray, np.ndarray, int]:
    """Return non-padded chunks, their starts, and dropped trailing samples."""
    size, hop = int(round(chunk_seconds * sr)), int(round(hop_seconds * sr))
    if size <= 0 or hop <= 0:
        raise ValueError('Chunk and hop must be positive')
    starts = np.arange(0, max(0, len(audio) - size + 1), hop, dtype=np.int64)
    chunks = np.stack([audio[i:i + size] for i in starts]) if starts.size else np.empty((0, size), np.float32)
    covered = int(starts[-1] + size) if starts.size else 0
    dropped = max(0, len(audio) - covered)
    return chunks.astype(np.float32), starts.astype(np.float64) / sr, dropped
