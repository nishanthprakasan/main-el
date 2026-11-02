"""Per-chunk ecoacoustic indices using scikit-maad's documented API."""
from __future__ import annotations
import logging
import warnings
import numpy as np
import pandas as pd
from maad import features, sound
from config import PipelineConfig

LOG = logging.getLogger(__name__)
INDEX_COLUMNS = ('aci', 'ndsi', 'adi', 'bi', 'aei')

def _clip_band(name: str, band: tuple[float, float], nyquist: float) -> tuple[tuple[float, float] | None, bool]:
    lo, hi = band
    clipped = (max(0.0, lo), min(float(nyquist), hi))
    changed = clipped != band
    if changed:
        warnings.warn(f'{name} band {band} clipped to {clipped} Hz at Nyquist {nyquist:g}', RuntimeWarning)
        LOG.warning('%s band clipped from %s to %s Hz (Nyquist=%g)', name, band, clipped, nyquist)
    if clipped[1] <= clipped[0]:
        warnings.warn(f'{name} band is empty or degenerate after clipping: {clipped}', RuntimeWarning)
        LOG.warning('%s band is empty/degenerate after clipping: %s', name, clipped)
        return None, True
    return clipped, changed

def compute_chunk_indices(audio: np.ndarray, sr: int, cfg: PipelineConfig) -> tuple[dict[str, float], bool, list[str]]:
    """Compute five indices from one reusable amplitude and power spectrogram."""
    if not np.any(audio):
        return {name: float('nan') for name in INDEX_COLUMNS}, True, []
    amp, _, fn, _ = sound.spectrogram(audio, sr, nperseg=cfg.n_fft,
        noverlap=cfg.overlap, mode='amplitude', detrend=False)
    # scikit-maad defines its amplitude spectrogram as sqrt(power); square it
    # for the NDSI power input so the STFT itself is evaluated only once.
    power = np.square(amp, dtype=np.float32)
    nyq = sr / 2
    flags: list[str] = []
    bio, bclip = _clip_band('biophony', cfg.biophony_band_hz, nyq)
    anthro, aclip = _clip_band('anthrophony', cfg.anthrophony_band_hz, nyq)
    biband, biclip = _clip_band('BI', cfg.bi_band_hz, nyq)
    divband, dclip = _clip_band('ADI/AEI', cfg.diversity_band_hz, nyq)
    for n, clipped in [('biophony', bclip), ('anthrophony', aclip), ('bi', biclip), ('diversity', dclip)]:
        if clipped:
            flags.append(n)
    vals = {name: float('nan') for name in INDEX_COLUMNS}
    # ACI returns (per-frame, per-bin, sum) in scikit-maad 1.5.1.
    vals['aci'] = float(features.acoustic_complexity_index(amp)[2])
    if bio and anthro:
        ndsi, *_ = features.soundscape_index(power, fn, flim_bioPh=bio, flim_antroPh=anthro)
        vals['ndsi'] = float(ndsi)
    if divband:
        args = dict(fmin=divband[0], fmax=divband[1], bin_step=cfg.diversity_bin_hz,
                    dB_threshold=cfg.index_db_threshold)
        vals['adi'] = float(features.acoustic_diversity_index(amp, fn, **args))
        vals['aei'] = float(features.acoustic_eveness_index(amp, fn, **args))
    if biband:
        vals['bi'] = float(features.bioacoustics_index(amp, fn, flim=biband))
    return vals, False, flags

def compute_indices(chunks: np.ndarray, sr: int, cfg: PipelineConfig) -> pd.DataFrame:
    """Return one indices row per chunk, with silence and clipping metadata."""
    rows = []
    for chunk in chunks:
        values, silent, flags = compute_chunk_indices(chunk, sr, cfg)
        rows.append({**values, 'silent_flag': silent, 'band_clipping_flags': '|'.join(flags)})
    return pd.DataFrame(rows, columns=[*INDEX_COLUMNS, 'silent_flag', 'band_clipping_flags'])
