#!/usr/bin/env python
# Copyright 2026 Andreas Koehler, MIT license
"""
Continuous Phase Detection Prediction Script
=============================================

This script performs continuous phase detection on seismic waveform data using
trained deep learning models. It processes data in sliding windows and can
optionally combine predictions from multiple array stations or beams.

Purpose
-------
Applies a trained phase detection model to continuous seismic data streams,
generating time-continuous P-wave and S-wave phase probability traces. Supports:
- Single station or multi-station array processing
- Sliding window prediction with configurable overlap
- Array station combination (stacking, voting, beamforming)
- Beam combination for enhanced detection
- Parallel processing for large datasets

Workflow
--------
1. Configuration Loading
   - Load main config.yaml (model, prediction settings)
   - Load separate data generation config if specified
   - Configure GPU settings

2. Phase Detection (if cfg.prediction.predict = True)
   If data.input_datatype = 'beams':
     - Load array geometry
     - Create beams at specified azimuths
     - Run phase detection on beams
     - Save beam predictions and/or picks
   Else:
     - Load trained model
     - Process continuous data in sliding windows
     - Apply model to each window
     - Combine overlapping windows (median/mean/percentile)
     - Save probability traces and/or picks

3. Array Station Combination (if cfg.prediction.combine_array_stations)
   - Load predictions from individual array stations
   - Combine using specified method (stack/vote/beam)
   - Save combined array prediction

4. Beam Combination (if cfg.prediction.combine_beams)
   - Load beam predictions
   - Combine and visualize maximum beam
   - Save combined beam prediction

Configuration
-------------
Main Config (config.yaml):
    model:
        type: Model architecture (transphasenet, etc.)
    
    data:
        data_config: Path to data generation config (REQUIRED)
        input_dataset_name: Dataset identifier
        sampling_rate: Sampling rate in Hz
    
    prediction:
        predict: Whether to run prediction (True/False)
        gpu: Use GPU for prediction (True/False)
        stations: List of station codes or array name
        start_time: Start time (ISO format: 'YYYY-MM-DDTHH:MM:SS')
        end_time: End time
        window_length: Processing window in seconds (e.g., 3600)
        step: Sliding window step in seconds
        stacking: Combination method ('median', 'mean', 'std', 'p25')
        output_dir: Output directory for results
        save_prob: Save probability traces (True/False)
        save_waveforms: Save waveforms (True/False)
        save_picks: Save discrete picks (True/False)
        combine_array_stations: Array combination methods
                                (False or ['stack'], ['vote'], ['beam'])
        combine_beams: Combine detection beams (True/False)
        detect_only: Direct combination without individual outputs
        num_processes: Number of parallel processes
        
        # Beam detection settings (when data.input_datatype = 'beams')
        azimuths: List of back-azimuths [deg] or False for grid search
        p_beam_vel: P-wave apparent velocity(ies) [km/s] (float or list)
        s_beam_vel: S-wave apparent velocity(ies) [km/s] (float or list)
        geometry_file: Path to array geometry JSON (optional)

Data Config (specified by data.data_config):
    Contains settings for waveform retrieval and preprocessing
    from the seismic database.

Usage
-----
Basic continuous prediction:
    python predict_contineous.py -c config.yaml

With custom config:
    python predict_contineous.py -c /path/to/custom_config.yaml

The script will:
    1. Load configuration and model
    2. Process continuous data in time windows
    3. Generate phase probability traces
    4. Optionally combine array stations
    5. Save results to cfg.prediction.output_dir

Output Files
------------
Probability traces (if save_prob=True):
    {output_dir}/{modelname}_{station}_{stacking}_{timestamp}.npz
        Contains: 'y' (probabilities), 't' (times), 'label' (channels)

Waveforms (if save_waveforms=True):
    {output_dir}/{modelname}_{station}_{stacking}_{timestamp}_wave.npz
        Contains: 'X' (waveforms), 't' (times)

Combined array predictions:
    {output_dir}/{modelname}_{array}_{stacking}_combined_{method}.npz
        Contains combined predictions from all array stations

Beam predictions:
    {output_dir}/{modelname}_{array}_maxbeam.npz
        Contains maximum beam predictions

Notes
-----
- Requires data_config file to be specified in config.yaml under data.data_config
- GPU memory growth is enabled to prevent OOM errors
- Processing is done in overlapping windows for smooth transitions
- Array combination requires station geometry
- Beam combination requires pre-computed beam predictions

See Also
--------
evaluate_contineous.py : Evaluate continuous predictions
contineous_prediction_utils.py : Contains beam_phase_detection and related functions

Dependencies
------------
- tensorflow: Deep learning model inference
- numpy: Array operations
- omegaconf: Configuration management
- obspy: Seismological data structures
"""

from obspy.clients.fdsn import Client
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from omegaconf import OmegaConf
from setup_config import get_config_dir,dict_to_namespace
from utils import combine_phase_detections
from contineous_prediction_utils import (
    phase_detection,
    beam_phase_detection,
    load_cont_detections,
    load_cont_beams
)
from beamforming import load_array_geometries
import argparse
import numpy as np

# =============================================================================
# COMMAND-LINE ARGUMENT PARSING
# =============================================================================

parser = argparse.ArgumentParser(
    description='Continuous phase detection prediction on seismic data.',
    epilog='Example: python predict_contineous.py -c config.yaml'
)
parser.add_argument(
    '-c', '--config_file_name',
    default='config.yaml',
    help='Main configuration file (contains model, data, and prediction settings)'
)
args = parser.parse_args()
cfg_file = args.config_file_name

# Initialize seismic database client
client = Client('UIB-NORSAR')

if __name__ == '__main__': 

    # ═════════════════════════════════════════════════════════════════════════
    # CONFIGURATION LOADING
    # ═════════════════════════════════════════════════════════════════════════
    
    print('Reading config ...')
    config_dir = get_config_dir()
    
    # Load main configuration file (model + prediction settings)
    args = OmegaConf.load(f'{config_dir}/{cfg_file}')
    args_dict = OmegaConf.to_container(args, resolve=True)
    args = OmegaConf.create(args_dict)
    OmegaConf.set_struct(args, False)
    cfg = dict_to_namespace(args)
    
    # Extract prediction settings from main config
    cfg_pred = cfg.prediction
    cfg_model = cfg
    cfg_data = cfg.data
    
    # Set default values for optional fields
    if not hasattr(cfg_model.data, "extract_array_channels"): 
        setattr(cfg_model.data, "extract_array_channels", False)
    if not hasattr(cfg_model.data, "setname"): 
        setattr(cfg_model.data, "setname", False)
    if not hasattr(cfg_pred, "skip_stations"):
        setattr(cfg_pred, "skip_stations", False)
    
    print('Config read.')

    # ═════════════════════════════════════════════════════════════════════════
    # GPU CONFIGURATION
    # ═════════════════════════════════════════════════════════════════════════
    
    if cfg.run.gpu:
        # Enable GPU processing with memory growth to prevent OOM errors
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f'GPU enabled: {len(gpus)} device(s) available')
            except RuntimeError as e:
                print(f'GPU configuration error: {e}')
    else:
        # Disable GPU, use CPU only
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print('CPU mode enabled')

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE DETECTION ON CONTINUOUS DATA
    # ═════════════════════════════════════════════════════════════════════════
    # Process continuous waveform data through trained model
    # For beams: applies beamforming to array data before phase detection
    # For other types: runs standard phase detection on station waveforms
    
    if cfg_pred.predict:
        print('\n' + '='*80)
        if 'beams' in cfg_model.data.input_datatype :
            print('RUNNING BEAM PHASE DETECTION ON ARRAY DATA')
            print('='*80)
            beam_phase_detection(client, cfg_pred, cfg_model)
            print('Beam phase detection complete.')
        else:
            print('RUNNING PHASE DETECTION ON CONTINUOUS DATA')
            print('='*80)
            phase_detection(client, cfg_pred, cfg_model, cfg_data)
            print('Phase detection complete.')

    # Exit early if only running detection without combination
    if cfg_pred.detect_only:
        print('Detect-only mode: exiting after phase detection.')
        exit()

    # ═════════════════════════════════════════════════════════════════════════
    # ARRAY STATION COMBINATION (OPTIONAL)
    # ═════════════════════════════════════════════════════════════════════════
    # Combine predictions from multiple array stations using stacking, voting,
    # or beamforming to improve detection performance
    
    if cfg_pred.combine_array_stations:
        print('\n' + '='*80)
        print('COMBINING ARRAY STATION PREDICTIONS')
        print('='*80)
        print(f'Combination method: {cfg_pred.combine_array_stations}')
        
        # Load individual station predictions
        pred_stream, array = load_cont_detections(cfg_pred, cfg_model)
        
        # Load geometry if beam combination is requested
        geometry = None
        if array and cfg_pred.combine_array_stations == 'beam':
            geometry_file = getattr(cfg_pred, 'geometry_file', None)
            geometries = load_array_geometries(geometry_file)
            geometry = geometries.get(array)
            if geometry:
                print(f'Loaded geometry for {array}: {len(geometry)} stations')
            else:
                print(f'[WARN] No geometry found for {array}, beam combination may fail')
        
        # Combine predictions using specified method
        st_comb = combine_phase_detections(
            pred_stream, False, cfg_pred, cfg_model, geometry=geometry, cont=True
        )

        if not array:
            print(f'No array identifier found. Using station list: {cfg_pred.stations} for combined output file name')
            array = "-".join(cfg_pred.stations)
        
        # Save combined predictions
        folder = cfg_pred.output_dir
        model_type = f'{cfg_model.data.input_dataset_name}_{cfg_model.model.type}'
        output_file = (f'{folder}/{model_type}_{array}_{cfg_pred.stacking}_'
                      f'combined_{cfg_pred.combine_array_stations}.npz')
        
        np.savez(
            output_file,
            t=st_comb[0].times('timestamp'),
            y=np.transpose(np.array([tr.data for tr in st_comb])),
            label=[tr.stats.station + '_' + tr.stats.channel for tr in st_comb]
        )
        print(f'Saved combined array predictions: {output_file}')

    # ═════════════════════════════════════════════════════════════════════════
    # BEAM COMBINATION (OPTIONAL)
    # ═════════════════════════════════════════════════════════════════════════
    # Combine detection beam predictions for maximum beam processing
    
    if cfg_pred.combine_beams:
        print('\n' + '='*80)
        print('COMBINING BEAM PREDICTIONS')
        print('='*80)
        
        # Load beam predictions
        pred_stream, array = load_cont_beams(cfg_pred, cfg_model)
        
        # Combine beams (no geometry needed - beams are already formed)
        st_comb = combine_phase_detections(
            pred_stream, False, cfg_pred, cfg_model, cont=True
        )
        
        # Visualize combined beam
        st_comb.plot()
        
        # Save combined beam predictions
        folder = cfg_pred.output_dir
        model_type = f'{cfg_model.data.input_dataset_name}_{cfg_model.model.type}'
        output_file = f'{folder}/{model_type}_{array}_maxbeam.npz'
        
        np.savez(
            output_file,
            t=st_comb[0].times('timestamp'),
            y=np.transpose(np.array([tr.data for tr in st_comb])),
            label=[tr.stats.station + '_' + tr.stats.channel for tr in st_comb]
        )
        print(f'Saved combined beam predictions: {output_file}')
    
    print('\n' + '='*80)
    print('PROCESSING COMPLETE')
    print('='*80)
