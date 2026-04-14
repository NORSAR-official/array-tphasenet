# Configuration Reference

This document explains the repository config structure.  
`config_1stat.yaml` is the base reference config; the other config files reuse
the same top-level sections with different values.

Repository test-data note:

- Only a small dummy subset is bundled in git for smoke testing.
- Several defaults in `config_1stat.yaml` (for example `epochs: 1`,
`batch_size: 10`, and year lists set to `[2022]`) are set so the repository
can run quickly with that dummy subset.

## 1. Which config to use

- `config_1stat.yaml`: single-station training, test prediction/evaluation, and continuous processing
- `config_1statfullarray.yaml`: ensemble detection on windowed test data using a single-station model (`run.only_predict: true`)
- `config_1statfullarray_cont.yaml`: ensemble continuous detection/evaluation config
- `config_zbeam.yaml`: vertical beam detection workflow
- `config_3cbeam.yaml`: three-component beam detection workflow
- `config_arrayarces_set2.yaml`: ARCES array detection workflow

## 2. Top-level sections

All configs use the same main blocks:

- `run`
- `data`
- `training`
- `normalization`
- `augment`
- `model`
- `evaluation`
- `prediction`

## 3. Section details

### 3.1 `run`

- `outputdir`: root output directory for models, predictions, and figures
- `code_location`: repository root for scripts that need absolute path context
- `only_predict`: skip model training and run prediction/evaluation only
- `predict_with_other_model`: use the current data settings but load/test with
another model setting
- `gpu`: enable/disable GPU usage in scripts that support it

### 3.2 `data`

- `input_dataset_name`: short dataset prefix used in file/model names
- `input_datatype`: input type (`single_station_waveforms`, `array_waveforms`, `beams`)
- `inputdir`: data directory (for example where OSF data is placed)
- `train_years`, `valid_years`, `test_years`: year splits
- `sampling_rate`: waveform sampling frequency (Hz) used by training data
- `allowed_phases`: phase types used as labels
- `holdout`: optional holdout file/list with event IDs excluded from training
- `lower_frequency`, `upper_frequency`: filter settings used so continuous
preprocessing matches how HDF5 training data was prepared
- `use_these_arraystations` (array-detection configs): full station/channel list
available in array waveform files
- `extract_array_channels` (array-detection configs): subset of stations/channels
used as model input features
- `setname` (array-detection configs): short identifier appended to model/output
naming for array subsets (for example `arces_set2`)

### 3.3 `training`

- `epochs`, `batch_size`: training loop size
- `learning_rate`, `optimizer`, `weight_decay`, `l1_norm`, `l2_norm`, `dropout`: optimization/regularization controls
- `class_weights`: label weighting (default order in `config_1stat.yaml` is
noise, P, S)
- `reduce_lr_patience`, `early_stopping_patience`: callback behavior

### 3.4 `normalization`

- `mode`: normalization strategy (for example `std`)
- `channel_mode`: per-channel vs shared normalization behavior

### 3.5 `augment`

Controls training-time data augmentation:

- `ramp`, `new_size`, `taper`
- `add_noise`, `add_event`
- `drop_channel`
- `add_gap`, `max_gap_size`

### 3.6 `model`

- `type`: model architecture name (for example `transphasenet`)
- `filters`, `kernel_sizes`, `pooling_type`
- `activation`, `att_type`, `rnn_type`, `additive_att`
- `residual_attention`: architecture depth/attention schedule

### 3.7 `evaluation`

- `optimal_threshold`: compute thresholds from recall/precision curves
- `p_threshold`, `s_threshold`: fixed thresholds used if `optimal_threshold: false`
- `dt`, `dt_cont`: timing tolerances for event-window and continuous evaluation
- `save_fig`: save evaluation figures
- `overall_performance`: write performance summary files
- `vs_metadata`: optional grouped metric figures (`distance`, `station`, `catalog`)
- `unpicked`: optional missing-picks figure on test windows
- `snr_threshold`, `snr_mode`: optional SNR filtering behavior (windowed);
config comments note this is not implemented for continuous evaluation;
`snr_mode` controls whether max or min arrival SNR is used per event
- `common_events_with_model`: optional event intersection mode
- `picks`: manual picks file for continuous evaluation
- `theoretical_arrivals`: include theoretical Pn/Pg/Sn/Sg arrivals in selected
windowed metrics

### 3.8 `prediction`

Used by continuous prediction and combination logic, and by selected
test-data post-processing paths:

- `predict`: run detection vs reuse prior outputs
- `stations`: station or array identifiers to process
- `arrays`: mapping from array names to station lists; supports wildcard station
patterns (for example `AR*`) for ensemble workflows
- `skip_stations`: stations to exclude (for example known bad channels/stations)
- `start_time`, `end_time`, `window_length`, `step`: continuous processing
windowing (`window_length` is seconds per prediction/output chunk)
- `stacking`: time window overlap prediction combination method (`mean`, `median`, `std`, `pc25`)
- `output_dir`: continuous output directory
- `save_prob`, `save_waveforms`, `save_picks`: output content switches for
continuous outputs (probabilities, waveforms, thresholded picks)
- `combine_array_stations`: multi-station combination for ensemble detection workflow (`stack`, `vote`, `beam`, or `false`)
- `combine_beams`: combine beam outputs into max-beam series for beam detection workflow
- `detect_only`: skip per-station probability outputs and directly combine/save picks
- `num_processes`: number of parallel workers used in detect-only mode
- `p_beam_vel`, `s_beam_vel`: apparent velocities for beamforming
- `vote_threshold_p`, `vote_threshold_s`: phase-specific thresholds for vote-based combination
- `azimuths` (beam configs): fixed beam back-azimuth list for beam detection  
(for example `0..330` every 30 degrees)

Config comment note:

- `combine_array_stations` is also used during test-data ensemble processing,
not only for continuous runs.

## 4. Notes for reproducibility runs

- `scripts/reproduce_benchmark.sh` reads a single `--config` and runs staged commands.
- For ensemble continuous processing, use `config_1statfullarray_cont.yaml`.
- For ensemble windowed test processing, use `config_1statfullarray.yaml` (`only_predict` mode).
- Local/private overrides should be kept in local config files (for example names containing `local`) and not committed.

## 5. Workflow-specific settings from other configs

Settings below are important in non-`1stat` workflows and may not appear in
`config_1stat.yaml`:

### 5.1 Ensemble Detecion Workflows (`config_1statfullarray*.yaml`)

- `run.only_predict: true` in both ensemble configs (skip local training stage)
- `prediction.combine_array_stations: 'stack'` for ensemble combination
- `config_1statfullarray.yaml` (windowed test ensemble):
- `run.predict_with_other_model: '1stat'` to reuse single-station model artifacts
- `data.input_dataset_name: 1statfullarray` for ensemble test-output naming
- `prediction.stations: ['ARCES','SPITS','FINES','NORES']` in the provided config
- `config_1statfullarray_cont.yaml` (continuous ensemble):
- `data.input_dataset_name: 1stat` in the provided config
- `prediction.stations: ['ARCES']` in the provided config
- `prediction.step: 120` in the provided config (coarser step for faster test runs)

### 5.2 Beam Detection Workflows (`config_zbeam.yaml`, `config_3cbeam.yaml`)

- `data.input_datatype: beams`
- `prediction.combine_beams: true`
- `prediction.azimuths`: explicit beam directions
- `prediction.p_beam_vel` can be a list (velocity sweep), while `prediction.s_beam_vel` may be scalar

### 5.3 Array Detection Workflow (`config_arrayarces_set2.yaml`)

- `data.input_datatype: array_waveforms`
- `data.use_these_arraystations`: full candidate station/channel inventory
- `data.extract_array_channels`: selected station/channel subset for this model
- `data.setname`: subset identifier used in naming

