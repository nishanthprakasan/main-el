"""SurfPerch embedding adapter for perch-hoplite."""
from __future__ import annotations
import logging
from typing import Any
import numpy as np
from scipy.signal import resample_poly
from config import PipelineConfig

LOG = logging.getLogger(__name__)

def load_surfperch(model_name: str = 'surfperch') -> Any:
    """Load the named model through perch-hoplite's model registry."""
    from perch_hoplite.zoo import model_configs
    return model_configs.load_model_by_name(model_name)

def _model_rate(model: Any) -> int:
    for obj in (model, getattr(model, 'config', None), getattr(model, 'audio_config', None)):
        if obj is None:
            continue
        for attr in ('sample_rate', 'sr', 'sampling_rate'):
            val = getattr(obj, attr, None)
            if isinstance(val, (int, float)):
                return int(val)
    return 32000

def _extract_embedding(output: Any) -> np.ndarray:
    """Extract the model's embedding tensor from common Hoplite outputs."""
    if isinstance(output, dict):
        for key in ('embeddings', 'embedding', 'embedding_0'):
            if key in output:
                output = output[key]
                break
        else:
            raise ValueError(f'No embedding key in model output: {list(output)}')
    elif hasattr(output, 'embeddings'):
        output = output.embeddings
    arr = np.asarray(output, dtype=np.float32)
    return arr

def embed_chunks(chunks: np.ndarray, sample_rate: int, cfg: PipelineConfig, model: Any | None = None) -> np.ndarray:
    """Embed chunks in batches; time/spatial axes are averaged to one vector."""
    model = model or load_surfperch(cfg.model_name)
    target_sr = _model_rate(model)
    expected_n = int(round(cfg.chunk_seconds * target_sr))
    outputs = []
    for start in range(0, len(chunks), cfg.embedding_batch_size):
        batch = []
        for chunk in chunks[start:start + cfg.embedding_batch_size]:
            resampled = resample_poly(chunk, target_sr, sample_rate).astype(np.float32) if target_sr != sample_rate else chunk.astype(np.float32)
            assert len(resampled) == expected_n, f'Model expects {expected_n} samples ({target_sr} Hz x {cfg.chunk_seconds}s); got {len(resampled)}'
            batch.append(resampled)
        for waveform in batch:
            raw = model.embed(waveform)
            arr = _extract_embedding(raw)
            # Perch wrappers can return per-window arrays with a singleton batch axis.
            arr = np.squeeze(arr)
            if arr.ndim == 1:
                vector = arr
            else:
                vector = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
            if vector.size != cfg.embedding_dim:
                raise ValueError(f'Expected {cfg.embedding_dim}-D SurfPerch vector, received {vector.size}; model output shape was {np.shape(raw)}')
            outputs.append(vector.astype(np.float32))
    result = np.stack(outputs) if outputs else np.empty((0, cfg.embedding_dim), dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError('SurfPerch produced non-finite embeddings')
    return result

def embed_file_mean(chunks: np.ndarray, sample_rate: int, cfg: PipelineConfig, model: Any | None = None) -> np.ndarray:
    """Average chunk vectors and L2-normalize to one recording vector."""
    vectors = embed_chunks(chunks, sample_rate, cfg, model)
    if not len(vectors):
        return np.full(cfg.embedding_dim, np.nan, dtype=np.float32)
    mean = vectors.mean(axis=0)
    norm = np.linalg.norm(mean)
    return (mean / norm if norm else mean).astype(np.float32)
