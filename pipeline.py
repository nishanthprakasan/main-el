"""Download, process, checkpoint, and collect reef audio features."""
from __future__ import annotations
import logging
import re
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, BinaryIO
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from config import PipelineConfig
from preprocessing import apply_condition, load_audio, segment_audio
from indices import compute_indices
from surfperch import embed_chunks

LOG = logging.getLogger(__name__)

def _safe_id(value: Any) -> str:
    value = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value)).strip('._')
    return value or 'file'

def _column(columns: list[str], explicit: str | None, candidates: tuple[str, ...], label: str) -> str:
    if explicit:
        if explicit not in columns:
            raise ValueError(f'Configured {label} column {explicit!r} missing. Available: {columns}')
        return explicit
    lower = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lower:
            return lower[candidate]
    raise ValueError(f'Could not infer {label} column from {columns}; set PipelineConfig.{label}_column')

def _output_paths(file_id: str, condition: str, cfg: PipelineConfig) -> tuple[Path, Path]:
    stem = f'{_safe_id(file_id)}__{_safe_id(condition)}'
    return cfg.output_dir / 'embeddings' / f'{stem}.npy', cfg.output_dir / 'indices' / f'{stem}.parquet'

def process_file(path: str | Path | BinaryIO, condition: str, cfg: PipelineConfig, file_id: str | None = None,
                 model: Any | None = None) -> tuple[Path, Path]:
    """Process one downloaded recording and atomically write condition outputs."""
    source_name = getattr(path, 'name', str(path))
    path_obj = Path(source_name) if not hasattr(path, 'read') else None
    file_id = _safe_id(file_id or (path_obj.stem if path_obj else 'audio'))
    emb_path, idx_path = _output_paths(file_id, condition, cfg)
    if emb_path.exists() and idx_path.exists():
        return emb_path, idx_path
    audio, sr = load_audio(path, cfg.pipeline_sample_rate)
    audio, sr = apply_condition(audio, sr, condition, cfg.seed)
    chunks, starts, dropped = segment_audio(audio, sr, cfg.chunk_seconds, cfg.hop_seconds)
    indices = compute_indices(chunks, sr, cfg)
    embeddings = embed_chunks(chunks, sr, cfg, model=model)
    if len(indices) != len(embeddings) or len(chunks) != len(indices):
        raise RuntimeError('Chunk, index, and embedding row counts differ')
    indices.insert(0, 'file_id', file_id)
    indices.insert(1, 'chunk_idx', np.arange(len(chunks), dtype=np.int64))
    indices.insert(2, 'start_s', starts)
    indices.insert(3, 'sample_rate', sr)
    indices.insert(4, 'condition', condition)
    indices['n_dropped_tail_samples'] = dropped
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    temp_emb = emb_path.with_suffix('.tmp.npy')
    temp_idx = idx_path.with_suffix('.tmp.parquet')
    np.save(temp_emb, embeddings.astype(np.float32))
    indices.to_parquet(temp_idx, index=False)
    temp_emb.replace(emb_path)
    temp_idx.replace(idx_path)
    return emb_path, idx_path

def _download(url: str, target: Path, cfg: PipelineConfig) -> Path:
    """Download with exponential backoff and raise after configured attempts."""
    target.parent.mkdir(parents=True, exist_ok=True)
    error = None
    for attempt in range(cfg.download_retries):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'reef-soundscape-pipeline/1.0'})
            with urllib.request.urlopen(request, timeout=120) as response, target.open('wb') as out:
                out.write(response.read())
            return target
        except Exception as exc:
            error = exc
            if target.exists():
                target.unlink()
            if attempt + 1 < cfg.download_retries:
                time.sleep(cfg.retry_backoff_seconds * (2 ** attempt))
    raise RuntimeError(f'Download failed after {cfg.download_retries} attempts: {url}') from error

def run(manifest_df: pd.DataFrame, conditions: list[str] | tuple[str, ...], cfg: PipelineConfig,
        model: Any | None = None) -> None:
    """Process manifest entries serially; transient audio is deleted afterward."""
    columns = list(manifest_df.columns)
    url_col = cfg.url_column or next((c for c in columns if c.lower() in {'url','download_url','download url','download_link','download link'}), None)
    path_col = cfg.filename_column or next((c for c in columns if c.lower() in {'path','local_path','file_path'}), None)
    if url_col is None and path_col is None:
        raise ValueError(f'Manifest needs a download URL or local path column. Columns: {columns}')
    id_col = cfg.id_column
    if id_col is None:
        id_col = next((c for c in columns if c.lower() in {'file_id','id','key','name','filename','file_name'}), None)
    LOG.info('Manifest columns=%s; URL=%s; path=%s; ID=%s', columns, url_col, path_col, id_col)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = cfg.download_dir or (Path(tempfile.gettempdir()) / 'reef_pipeline_downloads')
    archive_markers = cfg.output_dir / '.completed_archives'
    archive_markers.mkdir(parents=True, exist_ok=True)
    for row_num, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc='Recordings'):
        file_id = _safe_id(row[id_col] if id_col else Path(str(row[path_col])).stem if path_col else row_num)
        needed = [(cond, _output_paths(file_id, cond, cfg)) for cond in conditions]
        marker = archive_markers / f'{file_id}.txt'
        if marker.exists() and set(marker.read_text(encoding='utf-8').splitlines()) >= set(conditions):
            continue
        if all(emb.exists() and idx.exists() for _, (emb, idx) in needed):
            continue
        local_path: Path | None = None
        downloaded = False
        try:
            if path_col and pd.notna(row[path_col]) and Path(str(row[path_col])).exists():
                local_path = Path(str(row[path_col]))
            elif url_col and pd.notna(row[url_col]):
                url = str(row[url_col])
                suffix = Path(url.split('?')[0]).suffix or '.audio'
                local_path = _download(url, temp_dir / f'{file_id}{suffix}', cfg)
                downloaded = True
            else:
                raise ValueError(f'No usable audio path/URL for row {row_num}')
            if zipfile.is_zipfile(local_path):
                with zipfile.ZipFile(local_path) as archive:
                    members = [m for m in archive.infolist()
                               if not m.is_dir() and Path(m.filename).suffix.lower() in {'.wav', '.flac'}]
                    if not members:
                        raise ValueError(f'No WAV/FLAC recordings found in archive {local_path.name}')
                    for member in members:
                        member_id = _safe_id(f'{file_id}__{Path(member.filename).with_suffix("").as_posix()}')
                        for cond in conditions:
                            emb, idx = _output_paths(member_id, cond, cfg)
                            if emb.exists() and idx.exists():
                                continue
                            # ZipExtFile is seekable; soundfile reads one member at a time.
                            with archive.open(member, 'r') as audio_stream:
                                process_file(audio_stream, cond, cfg, file_id=member_id, model=model)
                marker.write_text('\n'.join(conditions), encoding='utf-8')
            else:
                for cond, (emb, idx) in needed:
                    if emb.exists() and idx.exists():
                        continue
                    process_file(local_path, cond, cfg, file_id=file_id, model=model)
        except Exception:
            LOG.exception('Failed processing manifest row %s (file_id=%s)', row_num, file_id)
            raise
        finally:
            if downloaded and local_path and local_path.exists():
                local_path.unlink()

def collect(cfg: PipelineConfig) -> tuple[Path, Path]:
    """Merge sorted per-recording outputs and assert strict row alignment."""
    index_files = sorted((cfg.output_dir / 'indices').glob('*.parquet'))
    tables, vectors = [], []
    for index_path in index_files:
        stem = index_path.stem
        embedding_path = cfg.output_dir / 'embeddings' / f'{stem}.npy'
        if not embedding_path.exists():
            raise FileNotFoundError(f'Missing aligned embedding file: {embedding_path}')
        table = pd.read_parquet(index_path)
        arr = np.load(embedding_path)
        if len(table) != len(arr):
            raise ValueError(f'Alignment failure in {stem}: {len(table)} rows vs {len(arr)} vectors')
        tables.append(table)
        vectors.append(arr.astype(np.float32))
    merged = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    embeddings = np.concatenate(vectors, axis=0) if vectors else np.empty((0, cfg.embedding_dim), np.float32)
    if len(merged) != len(embeddings):
        raise ValueError(f'Collected row mismatch: {len(merged)} indices vs {len(embeddings)} vectors')
    feature_path = cfg.output_dir / 'features.parquet'
    embedding_path = cfg.output_dir / 'embeddings.npy'
    merged.to_parquet(feature_path, index=False)
    np.save(embedding_path, embeddings)
    return feature_path, embedding_path
