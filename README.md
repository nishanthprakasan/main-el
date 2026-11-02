# Reef soundscape stages 2–3

The workspace did not contain the user manifest, so no manifest headers could be inspected. `run()` reports all headers and infers `download_url`/`url` plus a common ID field; set `PipelineConfig.url_column`, `id_column`, or `filename_column` when the Figshare export uses other names.

NDSI defaults use anthrophony 0–2 kHz and biophony 2–8 kHz, the conventional terrestrial split adopted here as an explicit starting point for reef recordings. Reef-specific interpretation should review these bands. The scikit-maad calls follow its documented signatures; requirements pin 1.5.2. One amplitude spectrogram is computed per non-silent chunk and squared to provide the corresponding power spectrogram for NDSI. ACI, ADI, AEI, and BI receive amplitude; NDSI receives power.

For SurfPerch, the adapter calls `model.embed()` and extracts the `embeddings` output. If output has more than one dimension after squeeze, all axes but the final feature axis are averaged. The expected model is 32 kHz, five seconds and 1280 dimensions; rate is read from public model attributes when available, otherwise the known SurfPerch default is used. Because the model package is absent in this build environment, the loaded object’s exact attributes/output shape remain to be checked in Colab. Integration check is opt-in to avoid a Kaggle model download in unit tests.

## Colab notebook

Open `colab_setup.ipynb`. Its first cell installs TensorFlow 2.20.0 and perch-hoplite before the remaining pinned dependencies. TensorFlow 2.20.0 has a published CPython 3.13 wheel, while 2.18.0 does not support the Python runtime reported by some current Colab sessions. The next cells mount Drive, request Figshare article 29958062 version 2, save its file manifest to `MyDrive/reef_pipeline/manifest.csv`, and print the detected columns and sample rows. The final cell runs the pipeline after the Python files are available under `WORK_DIR` (upload/copy this project folder to Drive first, or change `WORK_DIR`). Audio downloads happen one recording at a time during `run()`.

Run synthetic tests with `pytest -m 'not integration'`. Real SurfPerch smoke test: set `RUN_SURFPERCH=1`; it requires Kaggle model access and can take time on the first download. All temporary downloads are removed after their row is processed; cached outputs are skipped on rerun.
