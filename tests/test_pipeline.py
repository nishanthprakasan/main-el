"""Synthetic, download-free regression tests for preprocessing and features."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from config import PipelineConfig
from preprocessing import apply_condition, segment_audio
from indices import compute_chunk_indices
from surfperch import embed_chunks
from pipeline import run

SR = 16000

def cfg(tmp_path):
    return PipelineConfig(output_dir=tmp_path, n_fft=1024, overlap=512)

def test_chunking_drops_tail():
    chunks, starts, dropped = segment_audio(np.zeros(23 * SR, np.float32), SR)
    assert chunks.shape == (4, 5 * SR)
    np.testing.assert_array_equal(starts, [0, 5, 10, 15])
    assert dropped == 3 * SR

def test_noise_determinism_and_snr():
    x = np.sin(2 * np.pi * 440 * np.arange(SR * 2) / SR).astype(np.float32) * 0.2
    a, _ = apply_condition(x, SR, 'noise_snr10', 42)
    b, _ = apply_condition(x, SR, 'noise_snr10', 42)
    c, _ = apply_condition(x, SR, 'noise_snr10', 43)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    noise = a - x
    snr = 20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) / np.sqrt(np.mean(noise.astype(np.float64) ** 2)))
    assert abs(snr - 10) < 0.5

def test_quant8_levels_and_condition_composition():
    x = np.linspace(-1, 1, 5000, dtype=np.float32)
    y, rate = apply_condition(x, SR, 'sr8k+quant8', 1)
    assert rate == 8000
    assert len(np.unique(y)) <= 256

def test_silent_is_nan_and_flagged(tmp_path):
    values, silent, _ = compute_chunk_indices(np.zeros(SR * 5, np.float32), SR, cfg(tmp_path))
    assert silent
    assert all(np.isnan(values[k]) for k in ('aci', 'ndsi', 'adi', 'bi', 'aei'))

@pytest.mark.parametrize(('tone_hz', 'expected_sign'), [(5000, 1), (1500, -1)])
def test_ndsi_tone_bands(tmp_path, tone_hz, expected_sign):
    x = 0.5 * np.sin(2 * np.pi * tone_hz * np.arange(SR * 5) / SR).astype(np.float32)
    values, _, _ = compute_chunk_indices(x, SR, cfg(tmp_path))
    assert values['ndsi'] * expected_sign > 0.9

def test_modulated_tone_aci_exceeds_white_noise(tmp_path):
    t = np.arange(SR * 5) / SR
    rng = np.random.default_rng(5)
    white = rng.normal(0, 0.001, t.shape).astype(np.float32)
    modulated = (0.5 * (0.5 + 0.5 * np.sin(2*np.pi*5*t)) * np.sin(2*np.pi*1000*t)).astype(np.float32)
    a = compute_chunk_indices(white, SR, cfg(tmp_path))[0]['aci']
    b = compute_chunk_indices(modulated, SR, cfg(tmp_path))[0]['aci']
    assert b > a

def test_band_clipping_warns_at_8khz(tmp_path):
    x = np.sin(2*np.pi*1000*np.arange(5*8000)/8000).astype(np.float32)
    with pytest.warns(RuntimeWarning, match='clipped'):
        compute_chunk_indices(x, 8000, cfg(tmp_path))

class FakeModel:
    sample_rate = 32000
    def embed(self, waveform):
        assert waveform.shape == (160000,)
        return {'embeddings': np.ones((1, 3, 1280), np.float32) * float(waveform.mean() + 1)}

def test_embeddings_shape_finite_and_repeatable(tmp_path):
    x = np.zeros((2, SR * 5), np.float32)
    settings = cfg(tmp_path)
    a = embed_chunks(x, SR, settings, FakeModel())
    b = embed_chunks(x, SR, settings, FakeModel())
    assert a.shape == (2, 1280)
    assert np.isfinite(a).all()
    np.testing.assert_array_equal(a, b)

@pytest.mark.integration
def test_surfperch_real_rate_resampling_similarity(tmp_path):
    """Opt-in smoke/invariance check; enable with RUN_SURFPERCH=1 and Kaggle access."""
    import os
    if os.environ.get('RUN_SURFPERCH') != '1':
        pytest.skip('Set RUN_SURFPERCH=1 to download/use the real model')
    from surfperch import load_surfperch
    model, settings = load_surfperch(), cfg(tmp_path)
    t = np.arange(SR * 5) / SR
    signal = np.sin(2*np.pi*700*t).astype(np.float32)[None, :]
    e16 = embed_chunks(signal, SR, settings, model)[0]
    down = apply_condition(signal[0], SR, 'sr8k', 0)[0][None, :]
    e8 = embed_chunks(down, 8000, settings, model)[0]
    similarity = float(np.dot(e16, e8) / (np.linalg.norm(e16)*np.linalg.norm(e8)))
    print('SurfPerch 16k vs 8k path cosine similarity:', similarity)
    assert np.isfinite(similarity)

def test_run_skips_completed_outputs(tmp_path, monkeypatch):
    import pipeline
    settings = cfg(tmp_path)
    (tmp_path / 'embeddings').mkdir()
    (tmp_path / 'indices').mkdir()
    np.save(tmp_path / 'embeddings' / 'rec__clean.npy', np.zeros((1, 1280), np.float32))
    pd.DataFrame({'aci': [1]}).to_parquet(tmp_path / 'indices' / 'rec__clean.parquet')
    def should_not_download(*args, **kwargs):
        raise AssertionError('completed output must be skipped before download')
    monkeypatch.setattr(pipeline, '_download', should_not_download)
    run(pd.DataFrame({'file_id': ['rec'], 'download_url': ['https://invalid/audio.wav']}), ['clean'], settings)
