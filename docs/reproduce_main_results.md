# Reproduce main results (step-by-step)

This is the single detailed guide for environment setup, data/model preparation,
and full reproducibility pipeline reproduction.

## 1. Environment

Use the repository environment file:

```bash
conda env create -f environment.yml
conda activate array-tphasenet-test
```

## 2. Quick verification

Preflight validation:

```bash
python scripts/check_inputs.py --config config_1stat.yaml --stage benchmark
```

Dry-run full command sequence:

```bash
bash scripts/reproduce_benchmark.sh --config config_1stat.yaml --dry-run
```

Interactive walkthrough notebook:

```text
notebooks/reproduce_main_results_walkthrough.ipynb
```

## 3. Data and model artifacts

### 3.1 External source

- OSF bundle (full 2022 test data, pre-trained models, and many precomputed predictions):
  - [https://doi.org/10.17605/OSF.IO/27FPK](https://doi.org/10.17605/OSF.IO/27FPK)
- Scope note:
  - The OSF bundle does **not** contain the full multi-year training corpus used for model development.
  - The released pre-trained models in OSF were trained on the full training data and are provided for reproducible inference/evaluation.

### 3.2 Local folder structure

Expected layout:

```text
array-tphasenet/
  data/
  output/
    models/
    predictions/
    continuous/
```

`output/models/`, `output/predictions/`, and `output/continuous/` are used directly by scripts.

### 3.3 Artifact reference


| Artifact                          | Source                       | Description / Required for                                                  | Destination           | Required/optional                              |
| --------------------------------- | ---------------------------- | --------------------------------------------------------------------------- | --------------------- | ---------------------------------------------- |
| `*_labels_phase_detection.hdf5`   | repo dummy + OSF full        | Phase labels / Training and prediction on windowed data                     | `data/`               | Required                                       |
| `*_single_station_waveforms.hdf5` | repo dummy + OSF full        | Single station waveforms / Single-station and ensemble detection workflows  | `data/`               | Required for those workflows                   |
| `*_beams.hdf5`                    | repo dummy + OSF full        | Beams / Beam detection workflows (`zbeam`, `3cbeam`)                        | `data/`               | Required for beam detection workflows          |
| `*_array_waveforms.hdf5`          | repo dummy + OSF full        | Array waveforms / Array detection workflow (`arces25` / set2)               | `data/`               | Required for array detection workflow          |
| `*_arrivals.csv`                  | repo + OSF                   | Arrival-level metadata / evaluation and bookkeeping                         | `data/`               | Required for default evaluation configs        |
| `metadata_2022.csv`               | repo + OSF                   | Event-level metadata / evaluation (`theoretical_arrivals` or `vs_metadata`) | `data/`               | Required when those options are enabled        |
| `manual_picks_arces.dat`          | repo                         | Continuous-data evaluation ground truth for ARCES                           | `data/`               | Required for `evaluate_continuous.py`          |
| `saved_model_*.tf`                | produced locally or OSF      | `TensorFlow models / predict_on_testdata.py` and `predict_continuous.py`    | `output/models/`      | Required unless model is trained locally first |
| `predictions_*.npz`               | produced locally or OSF      | `Prediction (phase detections) on event windows / evaluate_on_testdata.py`  | `output/predictions/` | Required unless generated in current run       |
| Continuous waveform stream        | external FDSN (`UIB-NORSAR`) | `predict_continuous.py`                                                     | Retrieved at runtime  | Required for continuous prediction             |


### 3.4 OSF download and preparation helpers

Use the helper scripts to fetch data/model artifacts from OSF and place them in
the expected local folders.

Data files (`*.hdf5`) with automatic merge of known split files:

```bash
bash scripts/prepare_osf_data.sh
```

By default, this helper downloads all `.hdf5` files from the OSF project to
`downloads/osf/`, then runs merge commands for known split waveform files into
`data/`.

Model and prediction artifacts (`saved_model_*.tf`, `predictions_*.npz`):

```bash
bash scripts/prepare_osf_artifacts.sh
```

By default, this helper downloads matching artifacts to
`downloads/osf/artifacts/`, then copies models to `output/models/` and
predictions to `output/predictions/`. Only names directly in that downloads
folder are used (not files nested in subfolders).

If you prefer manual handling, some OSF files are split because of size limits.
Merge them before running scripts:

```bash
python split_data_for_repo.py merge --pattern "./Downloads/1statfullarray_2022_single_station_waveforms_000*.hdf5" --output data/1statfullarray_2022_single_station_waveforms.hdf5
python split_data_for_repo.py merge --pattern "./Downloads/array25arces_2022_array_waveforms_000*.hdf5" --output data/arces25_2022_array_waveforms.hdf5
```

Adjust `--pattern` to match your downloaded filenames if needed.

Preview only:

```bash
bash scripts/prepare_osf_data.sh --dry-run
bash scripts/prepare_osf_artifacts.sh --dry-run
```

Merge only for split data files (if chunk files are already downloaded):

```bash
bash scripts/prepare_osf_data.sh --merge-only
```

Skip OSF download and only copy already downloaded model/prediction artifacts:

```bash
bash scripts/prepare_osf_artifacts.sh --skip-download
```

Custom matching patterns (both helpers support repeatable `--pattern`):

```bash
bash scripts/prepare_osf_data.sh --pattern "1statfullarray_2022_*_000*.hdf5"
bash scripts/prepare_osf_artifacts.sh --pattern "saved_model_*.tf" --pattern "predictions_*.npz"
```

## 4. Optional wrapper for full reproducibility pipeline

Note: This section mirrors the quick-start pipeline summary in
`[README.md](../README.md)` for convenience.

For configuration options and field descriptions, see
`[docs/config_reference.md](./config_reference.md)`.

Before long runs, validate file/config prerequisites:

```bash
python scripts/check_inputs.py --config config_1stat.yaml --stage benchmark
```

Run the standard reproducibility command sequence automatically:

```bash
bash scripts/reproduce_benchmark.sh --config config_1stat.yaml
```

The preflight stage name `benchmark` is kept for compatibility and corresponds
to full reproducibility pipeline checks.

The wrapper runs all stages in sequence, including continuous prediction and
continuous evaluation.
If a config sets `run.only_predict: true` (for example
`config_1statfullarray.yaml` for ensemble detection with
`predict_with_other_model`), the wrapper skips `train` automatically and starts
from `predict-test`.
If `train` runs and already writes the needed test predictions (single test year
and no `predict_with_other_model` override), the wrapper skips `predict-test`
automatically.

## 5. Reproduce windowed test-data analyses

This is the manual, stage-by-stage walkthrough of the windowed event data part of  
`scripts/reproduce_benchmark.sh` (`train`, `predict-test`, `evaluate-test`).
Use the wrapper script for one-command execution; use this section when you want
to run or debug stages explicitly.

We recommend training prediction on a GPU when using the OSF data.

Configuration help: `[docs/config_reference.md](./config_reference.md)`.

Run these in order.

### 5.1 Single-station detection

```bash
python train.py --config config_1stat.yaml
python evaluate_on_testdata.py --config config_1stat.yaml
```

Main output:

- `output/performance_1stat_transphasenet.txt`

### 5.2 Ensemble detection from single-station model

```bash
python predict_on_testdata.py --config config_1statfullarray.yaml
python evaluate_on_testdata.py --config config_1statfullarray.yaml
```

Main output:

- `output/performance_1statfullarray_transphasenet_1stat_stack.txt`

### 5.3 Beam detection

```bash
python train.py --config config_zbeam.yaml
python evaluate_on_testdata.py --config config_zbeam.yaml
python train.py --config config_3cbeam.yaml
python evaluate_on_testdata.py --config config_3cbeam.yaml
```

Main outputs:

- `output/performance_zbeam_transphasenet.txt`
- `output/performance_3cbeam_transphasenet.txt`

### 5.4 Array detection (ARCES set 2)

```bash
python train.py --config config_arrayarces_set2.yaml
python predict_on_testdata.py --config config_arrayarces_set2.yaml
python evaluate_on_testdata.py --config config_arrayarces_set2.yaml
```

Main output:

- `output/performance_arrayarces_set2_transphasenet.txt`

### 5.5 Windowed outputs (files and figures)

`train.py` writes:

- `output/models/saved_model_<setting>.tf`
- `output/predictions/predictions_<setting>.npz` with `x`, `y`, `yhat`, `ids`, `arids`, `stations`
- `output/random_samples_pretrain_<setting>.png`
- `output/random_samples_<setting>.png`

`predict_on_testdata.py` writes:

- `output/predictions/predictions_<setting>.npz`
- if multiple test years are configured: `output/predictions/predictions_<setting>_<year>.npz`

`evaluate_on_testdata.py` writes:

- `output/performance_<setting>[suffix].txt` (precision/recall/F1/residual summary)
- `output/L_curve_<setting>[suffix].png` and `output/L_curve_<setting>[suffix].txt` when `evaluation.optimal_threshold: true`
- `output/residuals_<setting>[suffix].png` when `evaluation.overall_performance: true` and `evaluation.save_fig: true`
- `output/performance_vs_<metadata>_<setting>[suffix].png` when `evaluation.vs_metadata` is set
- `output/missing_picks_<setting>[suffix].png` when `evaluation.unpicked: true`

## 6. Reproduce fully labeled continuous-data analyses

This is the manual, stage-by-stage walkthrough of the continuous processing part of  
`scripts/reproduce_benchmark.sh` (`predict-continuous`,
`evaluate-continuous`). Use the wrapper script for one-command execution; use
this section when you want to run or debug continuous stages explicitly.

Configuration help: `[docs/config_reference.md](./config_reference.md)`.

For each selected config, run prediction first, then evaluation.
`predict_continuous.py` retrieves waveforms from the `UIB-NORSAR` FDSN service,
so network access and data availability for the selected time window are required.
The provided configs are set to a 1-hour quick test window by default
(`prediction.start_time: 2023-01-01T00:00:00`,
`prediction.end_time: 2023-01-01T01:00:00` in most configs). To run the full
4-day labeled interval, increase `prediction.end_time` in the selected config,
for example to `2023-01-05T00:00:00`.

### 6.1 Single-station detection continuous

```bash
python predict_continuous.py -c config_1stat.yaml
python evaluate_continuous.py --config config_1stat.yaml
```

### 6.2 Ensemble detection continuous

```bash
python predict_continuous.py -c config_1statfullarray_cont.yaml
python evaluate_continuous.py --config config_1statfullarray_cont.yaml
```

### 6.3 Beam detection continuous

```bash
python predict_continuous.py -c config_zbeam.yaml
python evaluate_continuous.py --config config_zbeam.yaml

python predict_continuous.py -c config_3cbeam.yaml
python evaluate_continuous.py --config config_3cbeam.yaml
```

### 6.4 Array detection continuous

```bash
python predict_continuous.py -c config_arrayarces_set2.yaml
python evaluate_continuous.py --config config_arrayarces_set2.yaml
```

### 6.5 Continuous outputs (probabilities, figures, and picks)

`predict_continuous.py` writes under `output/continuous/`:

- per-station probability traces (if `prediction.save_prob: true`):  
`<modelname>_<station>_<stacking>_<timestamp>.npz` with `t` and `y`
- optional waveform bundles (if `prediction.save_waveforms: true`):  
`<modelname>_<station>_<stacking>_<timestamp>_wave.npz` with `t`, `x`, and `y`
- optional combined array probabilities (if `prediction.combine_array_stations` is enabled):  
`<modelname>_<array>_<stacking>_combined_<method>.npz` with `t`, `y`, and `label`
- optional combined beam probabilities (if `prediction.combine_beams: true`):  
`<modelname>_<array>_maxbeam.npz` with `t`, `y`, and `label`

`evaluate_continuous.py` writes under `output/continuous/`:

- `performance_<model><suffix>_cont.txt`
- optional `L_curve_<model><suffix>_cont.png` when `evaluation.optimal_threshold: true` and `evaluation.save_fig: true`
- optional exported probability traces in MiniSEED format (if `prediction.save_prob: true`):
`predictions_<model>_<station>.msd` (station-by-station) and/or
`predictions_<model>_<method>_<station-or-array>.msd` (combined outputs)
- optional pick files (if `prediction.save_picks: true`):
`<output/continuous>/<model><suffix>_<p_threshold>_<s_threshold>/picks_<station>.json`
with detection time, phase, probability, duration, and threshold

In the provided configs, `prediction.save_prob` is typically enabled and
`prediction.save_picks` is typically disabled by default.

## 7. Quick output checklist

After runs complete, verify expected artifacts:

```bash
ls output/performance_*.txt
ls output/continuous/performance_*.txt
```

If files are missing, check:

- paths in the selected config file
- model/prediction artifacts in `output/models/` and `output/predictions/`
- data availability in `data/`

## 8. What is intentionally not bundled

- Full-size data and model bundles are stored externally (OSF) because of repository size limits.
- Prediction archives for ensemble and array detection are not bundled in OSF because of total-size limits; regenerate them by running `predict_on_testdata.py` on the provided 2022 OSF test data.
- The full multi-year raw training corpus is not included in this repository or in the OSF bundle.
