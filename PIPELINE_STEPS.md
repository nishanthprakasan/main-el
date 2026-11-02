# Step-by-step: run the reef soundscape pipeline in Colab

This workflow runs preprocessing, ecoacoustic indices, and SurfPerch embedding extraction in Colab. It downloads one recording at a time from Figshare, writes the processed outputs to Google Drive, then deletes the temporary audio file. It does **not train a classifier**.

## 1. Prepare the project files

1. Put the project files in a Google Drive folder named `reef_pipeline` (or update the paths below). Include at least `config.py`, `preprocessing.py`, `indices.py`, `surfperch.py`, `pipeline.py`, and `requirements.txt`.
2. Open `colab_setup.ipynb` in Google Colab. You can upload that notebook to Drive or create a new Colab notebook and copy the cells from it.
3. In Colab, select **Runtime → Change runtime type** and choose a GPU runtime if one is available. A GPU is optional for inference; a free GPU is not guaranteed.

## 2. Install dependencies

Run the first cell in `colab_setup.ipynb` in order. It installs TensorFlow 2.20.0, perch-hoplite 1.0.2, then the remaining packages. This TensorFlow pin is selected because the original 2.18.0 pin was unavailable in the reported Colab Python environment.

If the install requests a runtime restart, choose **Runtime → Restart session**. After restart, continue from the Drive mount cell; rerun the install cell only if the packages are missing.

## 3. Mount Drive and fetch the Figshare manifest

Run the Drive cell to mount Google Drive at `/content/drive` and create:

```text
/content/drive/MyDrive/reef_pipeline
```

Run the manifest cell. It requests Figshare article `29958062`, version `2`, writes `manifest.csv` to that Drive folder, and displays its column names and first rows. Expected columns are `id`, `name`, `size`, `download_url`, and `computed_md5`. The pipeline recognizes `download_url` and `id` automatically. The run cell selects only `ken_H1.zip`, `ken_D1.zip`, `ken_N1.zip`, `ken_D2.zip`, and `ken_H2.zip`, and stops with an error if any are missing.

## 4. Make project modules available in Colab

The notebook's final cell inserts `WORK_DIR` into `sys.path`. Confirm the Python files from step 1 are in that folder. If you saved them in another folder, change `WORK_DIR` in the Drive cell to that folder.

The Colab runtime cannot import files that only exist on your local computer. Copy/upload the modules to Drive (or upload them to the active Colab session and change `sys.path`) before running the pipeline.

## 5. Configure and run

In the final notebook cell, configure the output folder and run the chosen conditions. For example:

```python
from config import PipelineConfig
from pipeline import run, collect

cfg = PipelineConfig(
    output_dir=WORK_DIR / 'outputs',
    seed=1234,
)

# Start with the clean condition. Conditions can also be combined, e.g.
# 'sr8k+quant8' or 'noise_snr10'.
conditions = ['clean']
run(manifest, conditions, cfg)

features_path, embeddings_path = collect(cfg)
print(features_path)
print(embeddings_path)
```

For the configured noise robustness conditions, make a list such as `['noise_snr20', 'noise_snr10', 'noise_snr5', 'noise_snr0']`. To compare clean, resampled, and quantized audio, use `['clean', 'sr8k', 'quant8', 'sr8k+quant8']`. Each condition is cached separately.

## 6. What happens for each recording

For each selected manifest row, `run()`:

1. Checks whether all requested outputs already exist; completed file/condition pairs are skipped.
2. Downloads the archive to Colab's temporary directory, one archive at a time, with retry and backoff. The largest supplied Kenya archive is about 11 GB, so make sure Colab has enough free temporary disk space. Set `PipelineConfig(download_dir=...)` if you need another location.
3. For ZIP input, opens each WAV/FLAC member in sequence, processes it, and moves on to the next member; it does not extract all the recordings together. It removes the temporary archive after its members are done.
4. Loads the audio as mono float32, resampling to 16 kHz if necessary.
5. Applies the selected condition to the full recording before segmentation.
6. Splits the conditioned signal into non-overlapping five-second chunks. A trailing partial chunk is dropped and its sample count is recorded.
7. Computes ACI, NDSI, ADI, BI, and AEI for each chunk. Silent chunks get NaN indices and a silence flag; clipped frequency bands are logged and recorded.
8. Resamples each chunk to the model sample rate as needed and calls SurfPerch for one embedding per chunk.
9. Saves embeddings and index rows to Drive.

If Colab disconnects, rerun the same cells. Existing completed outputs are skipped; an incomplete current file/condition may be processed again.

## 7. Check the final files

After `collect(cfg)` completes, the output directory should contain:

```text
outputs/
├── embeddings/                  # one float32 .npy array per file and condition
├── indices/                     # one .parquet table per file and condition
├── features.parquet             # merged index and metadata rows
└── embeddings.npy               # merged embeddings in matching row order
```

Each row of `features.parquet` corresponds to the vector in the same row of `embeddings.npy`. The collector checks this alignment. The per-chunk metadata includes file ID, chunk index, start time, sample rate, condition, silence flag, dropped-tail sample count, and band clipping flags.

## 8. Validate the model before a full run

The synthetic tests do not download the model. Run them from a Colab cell:

```python
%cd /content/drive/MyDrive/reef_pipeline
!pytest -m 'not integration'
```

Before processing the whole manifest, try one recording and check the logs and output shapes. The real SurfPerch load/inference path was not verified in the development environment. If you want to run the integration check, configure Kaggle access in Colab, set `RUN_SURFPERCH=1`, and run `pytest -m integration -s`. Confirm the loaded model's actual input rate and embedding shape in that run.

## Notes

- The configured NDSI defaults use anthrophony 0–2 kHz and biophony 2–8 kHz; review whether those bands suit the reef analysis.
- The pipeline uses the pretrained SurfPerch model to extract embeddings. A downstream classifier-training stage, including labeled data and a train/validation split, is not included here.
- Kaggle credentials are only relevant if the SurfPerch model download prompts for Kaggle authentication. Configure them in Colab secrets/environment; do not commit credentials to Drive with the project source.
