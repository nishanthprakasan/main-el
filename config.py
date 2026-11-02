"""Central configuration for the reef soundscape pipeline."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    output_dir: Path = Path('/content/drive/MyDrive/reef_pipeline')
    download_dir: Optional[Path] = None
    chunk_seconds: float = 5.0
    hop_seconds: float = 5.0
    pipeline_sample_rate: int = 16000
    conditions: tuple[str, ...] = ('clean',)
    noise_snr_db: tuple[float, ...] = (20.0, 10.0, 5.0, 0.0)
    seed: int = 1234
    n_fft: int = 1024
    overlap: int = 512
    biophony_band_hz: tuple[float, float] = (2000.0, 8000.0)
    anthrophony_band_hz: tuple[float, float] = (0.0, 2000.0)
    bi_band_hz: tuple[float, float] = (2000.0, 8000.0)
    diversity_band_hz: tuple[float, float] = (0.0, 8000.0)
    diversity_bin_hz: float = 500.0
    index_db_threshold: float = -50.0
    url_column: Optional[str] = None
    id_column: Optional[str] = None
    filename_column: Optional[str] = None
    download_retries: int = 4
    retry_backoff_seconds: float = 2.0
    embedding_batch_size: int = 8
    embedding_dim: int = 1280
    model_name: str = 'surfperch'

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.download_dir is not None:
            self.download_dir = Path(self.download_dir)
        if self.chunk_seconds <= 0 or self.hop_seconds <= 0:
            raise ValueError('chunk_seconds and hop_seconds must be positive')
        if self.n_fft < 2 or not 0 <= self.overlap < self.n_fft:
            raise ValueError('Require n_fft >= 2 and 0 <= overlap < n_fft')
