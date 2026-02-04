# Copyright 2026, Andreas Koehler, MIT license

"""
Generate Model Predictions on Test Data
========================================

This script loads a trained seismic phase detection model and generates predictions
on test data from specified years. Predictions are saved as compressed NumPy archives
for subsequent evaluation.

Workflow
--------
1. Load configuration from config.yaml
2. Configure GPU/CPU execution environment
3. Locate test data files for specified years
4. For each test file:
   - Load waveform and label data
   - Create data generator with appropriate preprocessing
   - Load trained model from disk
   - Generate predictions using the model
   - Extract metadata (event IDs, arrival IDs, station names)
   - Save predictions and metadata to .npz file

Configuration
-------------
Uses settings from config.yaml:
- data.test_years: Years to predict on (can be list for year-by-year output)
- model.type: Model architecture name
- run.gpu: Whether to use GPU acceleration
- run.predict_with_other_model: Optional override for model loading
- run.custom_outname: Optional custom name for output files
- data.extract_array_channels: Use array channel extraction
- data.setname: Array configuration identifier

Output Files
------------
Saves to outputs/predictions_{setting}.npz (or with year suffix if multiple years):
- x: Input waveforms (may differ from raw if preprocessing applied)
- y: True labels
- yhat: Model predictions
- ids: Event identifiers
- arids: Arrival identifiers
- stations: Station names

Usage
-----
Run with default config file::

    python predict_on_testdata.py

Or specify a custom config file::

    python predict_on_testdata.py --config model_configs/config_array9arces.yaml

Notes
-----
- Predictions are generated month-by-month to manage memory usage
- Mixed precision (float16) is enabled for faster inference
"""

import argparse
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from omegaconf import OmegaConf
from setup_config import add_root_paths, get_config_dir, dict_to_namespace
from train_utils import CustomStopper
from tensorflow.keras import mixed_precision
from train_utils import get_data_files, create_data_generator, get_model, get_predictions
import matplotlib.pyplot as plt


# ============================================================================
# Configuration Loading
# ============================================================================

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='Generate model predictions on test data.',
    epilog='Example: python predict_on_testdata.py --config model_configs/config_array9arces.yaml'
)
parser.add_argument(
    '--config',
    type=str,
    default=None,
    help='Path to training configuration file (default: config.yaml or tf/config.yaml)'
)
cmd_args = parser.parse_args()

print('Reading config ...')

# Load config from command line argument or try default locations
if cmd_args.config:
    args = OmegaConf.load(cmd_args.config)
else:
    # Try GPU-specific path first, fallback to standard location
    try:
        config_dir = 'tf/'
        args = OmegaConf.load(f'{config_dir}/config.yaml')
    except FileNotFoundError:
        config_dir = get_config_dir()
        args = OmegaConf.load(f'{config_dir}/config.yaml')

# Resolve all interpolations and convert to namespace
args_dict = OmegaConf.to_container(args, resolve=True)
args = OmegaConf.create(args_dict)
OmegaConf.set_struct(args, False)
cfg = dict_to_namespace(args)

# Set default values for optional configuration parameters
if not hasattr(cfg.data, "extract_array_channels"):
    setattr(cfg.data, "extract_array_channels", False)
if not hasattr(cfg.data, "setname"):
    setattr(cfg.data, "setname", False)
if not hasattr(cfg.data, "remove_zero_channels"):
    setattr(cfg.data, "remove_zero_channels", False)
if not hasattr(cfg.run, "custom_outname"):
    setattr(cfg.run, "custom_outname", False)
if not hasattr(cfg.data, "noise_waveforms"):
    setattr(cfg.data, "noise_waveforms", False)

print('Config read.')


# ============================================================================
# Hardware Configuration
# ============================================================================

if cfg.run.gpu:
    # Enable GPU with dynamic memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
else:
    # Force CPU-only execution
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Enable mixed precision for faster inference
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


# ============================================================================
# Data Location Configuration
# ============================================================================

inputdir = cfg.data.inputdir
outputdir = cfg.run.outputdir


# ============================================================================
# Test Data Loading and Prediction Loop
# ============================================================================

testing_years = cfg.data.test_years

# Get test data files for specified years
# Processing month-wise to avoid memory issues
testing_files = get_data_files(inputdir, testing_years, cfg)

# Loop over test files (one per month/year)
for i, testing_file in enumerate(testing_files[0]):
    # Extract noise file if available
    if testing_files[2] is None:
        noisefile = None
    else:
        noisefile = testing_files[2][i]
    
    # Create data generator for current test file
    test_dataset, nchannels = create_data_generator(
        [[testing_file], [testing_files[1][i]], [noisefile]], 
        cfg, 
        training=False
    )

    # -----------------------------------------------------------------------
    # Model Loading
    # -----------------------------------------------------------------------
    
    model_type = cfg.model.type
    
    # Determine model path based on configuration
    if cfg.run.predict_with_other_model:
        dataset = cfg.run.predict_with_other_model
    else:
        dataset = cfg.data.input_dataset_name
    
    setting = f'{dataset}_{model_type}'
    
    if cfg.data.extract_array_channels:
        setting = f'array{cfg.data.setname}_{model_type}'
    
    model = tf.keras.models.load_model(
        f'{outputdir}/models/saved_model_{setting}.tf', 
        compile=False
    )

    # -----------------------------------------------------------------------
    # Prediction Generation
    # -----------------------------------------------------------------------
    
    # Redefine output naming (may differ from model loading path)
    
    if cfg.run.custom_outname:
        setting = f'{dataset}_{cfg.run.custom_outname}_{model_type}'
    
    if cfg.run.predict_with_other_model:
        setting += f'_{cfg.run.predict_with_other_model}'
    
    # Generate predictions
    xte, true, pred, sample_weight = get_predictions(
        cfg, test_dataset, model
    )
    
    # -----------------------------------------------------------------------
    # Metadata Extraction
    # -----------------------------------------------------------------------
    
    # Extract original data from generator
    x = [a['x'] for a in test_dataset.super_sequence.data]
    y = [a['y'] for a in test_dataset.super_sequence.data]
    
    # Extract event and arrival identifiers
    ids = [a['event_id'] for a in test_dataset.super_sequence.data]
    arids = [a['arrival_ids'] for a in test_dataset.super_sequence.data]
    stations = [a['station'] for a in test_dataset.super_sequence.data]
    
    # -----------------------------------------------------------------------
    # Save Predictions
    # -----------------------------------------------------------------------
    
    outputfile = f'{outputdir}/predictions/predictions_{setting}.npz'
    
    # Add year suffix if processing multiple years
    if len(testing_years) > 1:
        outputfile = f'{outputdir}/predictions/predictions_{setting}_{testing_years[i]}.npz'
    
    np.savez(
        outputfile,
        x=xte,
        y=true,
        yhat=pred,
        ids=np.array(ids),
        arids=arids,
        stations=stations
    )
    
    print(f'Predictions saved to {outputfile}')
