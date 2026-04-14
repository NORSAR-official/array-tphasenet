# Array phase detection using TPhaseNet models

Code and configurations for the paper **Adapting deep learning phase detectors for seismic array processing**.

## Documentation

- [Reproduce Main Results](docs/reproduce_main_results.md)
- [Configuration Reference](docs/config_reference.md)

## Data, Setup, and Reproduction

This repository contains code, scripts, configs, and small dummy data for quick checks.  
Full-size 2022 test data, pre-trained models, and test predictions are distributed separately at OSF:

[https://doi.org/10.17605/OSF.IO/27FPK](https://doi.org/10.17605/OSF.IO/27FPK)

Important scope note: the OSF bundle does not include the full multi-year training
corpus used for model development. It does include the released pre-trained models
that were trained on the full training data.

OSF helper scripts are included for automated artifact download/prep:

- `bash scripts/prepare_osf_data.sh` downloads `.hdf5` data files and merges known split chunks into `data/`.
- `bash scripts/prepare_osf_artifacts.sh` downloads `saved_model_*.tf` and `predictions_*.npz` into `output/models/` and `output/predictions/`.

## Environment setup

Use the repository environment file:

```bash
conda env create -f environment.yml
conda activate array-tphasenet-test
```

## Simple Reproducibility Pipeline

Use the preflight checker before running expensive stages:

```bash
python scripts/check_inputs.py --config config_1stat.yaml --stage benchmark
```

Available `--stage` options:

- `benchmark`: full reproducibility preflight (train + metadata checks)
- `train`: checks training/validation/test input files
- `predict-test`: checks test input files and required trained model
- `evaluate-test`: checks metadata and test prediction files
- `predict-continuous`: checks model availability for continuous prediction
- `evaluate-continuous`: checks manual picks and continuous prediction outputs

`scripts/reproduce_benchmark.sh` runs `scripts/check_inputs.py` internally for  
each stage it executes (disable with `--skip-preflight`).

Run the standard reproducibility pipeline with one command:

```bash
bash scripts/reproduce_benchmark.sh --config config_1stat.yaml
```

This script runs all stages listed above in order: `train`, `predict-test`,  
`evaluate-test`, `predict-continuous`, and `evaluate-continuous`.

If `run.only_predict: true` is set in the selected config (for example  
`config_1statfullarray.yaml for ensemble detection`), the `train` stage is skipped automatically and  
the pipeline starts from `predict-test`. Otherwise, `train` already produced test predictions and `predict-test` is skipped automatically.

The bundled repository data is a small dummy subset intended only for smoke
tests. Results from test-data evaluation and continuous prediction/evaluation on
this dummy data are not scientifically usable.

For array processing (instead of single-station processing), use one of these
array config files:

- `Ensembel detection: config_1statfullarray.yaml`
- `Beam detection with vertical component beams: config_zbeam.yaml`
- `Beam detection with three-component beams: config_3cbeam.yaml`
- `Array detection (ARCES only): config_arrayarces_set2.yaml`

## Training and Testing with Full 2022 Data

Comprehensive notebook walkthrough including OSF prep and stage-by-stage commands:

- `notebooks/reproduce_main_results_walkthrough.ipynb`

## Smoke tests

Run local smoke tests (test suite in tests/) to check core functionality and wiring, and that the repository is healthy:

```bash
pytest -m smoke -q
```

## Full Reproduction Guide

For complete workflow details (artifact mapping, stage-by-stage commands, and
continuous-data notes), see:

- [docs/reproduce_main_results.md](docs/reproduce_main_results.md)

For configuration key explanations and choosing the right config file, see:

- [docs/config_reference.md](docs/config_reference.md)
