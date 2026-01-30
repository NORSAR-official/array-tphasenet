"""
Training Utilities for Seismic Phase Detection Models
======================================================

This module provides comprehensive utilities for training deep learning models
for seismic phase picking and earthquake detection, specifically designed for
PhaseNet-based architectures and their variants.

Main Components
---------------
Data Loading and Generators
    - EQDatareader: Main data generator for loading and augmenting seismic waveforms
    - DropDetection: Wrapper for handling class weights and label formatting
    - create_data_generator: Factory function for creating configured data generators
    
Waveform Preprocessing
    - waveform_normalize: Normalize waveforms by max amplitude or standard deviation
    - waveform_crop, waveform_crop_new: Crop waveforms with optional smart cropping
    - create_overlapping_windows: Generate overlapping windows for training
    
Data Augmentation
    - waveform_add_noise: Add realistic noise to waveforms
    - waveform_add_gap: Simulate data gaps
    - waveform_drop_channel: Channel dropout augmentation
    - waveform_taper: Apply Tukey taper to waveform edges
    - label_smoothing: Smooth pick labels with Gaussian or triangular windows
    
Model Creation
    - get_model: Factory function for instantiating and compiling models
    - get_splitoutput_losses_and_weights: Configure losses for split-output models
    
Metrics and Callbacks
    - keras_f1: F1-score metric for phase picking
    - CustomStopper: Early stopping with custom behavior
    - GradientNormLogger: Monitor gradient norms during training
    - NanGuard: Detect and handle NaN values in training
    
Prediction and Evaluation
    - get_predictions: Generate predictions on test data
    - plot_random_samples: Visualize random samples from dataset

Usage Example
-------------
>>> from train_utils import create_data_generator, get_model
>>> 
>>> # Create data generator
>>> train_gen, n_channels = create_data_generator(
...     files=[['data/1stat_2022_single_station_waveforms.hdf5'],['data/1stat_2022_labels_phase_detection.hdf5']], 
...     config=config, 
...     training=True
... )
>>> 
>>> # Create and compile model
>>> model = get_model(config, n_channels)
>>> 
>>> # Train
>>> model.fit(train_gen, epochs=100, callbacks=[...])

Notes
-----
- All waveform functions expect numpy arrays with shape (n_samples, n_channels)
- Label arrays typically have shape (n_samples, n_classes) where n_classes=3 
  for [noise, P-wave, S-wave]

Author: Erik Myklebust, Andreas Koehler, Tord Stangeland, Steffen Mæland
License: MIT

See Also
--------
models : Neural network architectures
"""

from typing import Tuple, List, Optional, Dict, Any, Union, Callable
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tensorflow.keras.backend as K
import h5py
from tqdm import tqdm
from random import shuffle, choices
from collections import defaultdict
import random
import string
import models as nm
from omegaconf.errors import ConfigAttributeError
from scipy.signal.windows import tukey, gaussian, triang
import os
import glob
import json
import matplotlib.pyplot as plt



def keras_f1(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate F1-score for binary classification.
    
    Computes the F1-score as the harmonic mean of precision and recall.
    Suitable for use as a Keras metric during training.
    
    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth binary labels of shape (batch_size, n_samples).
    y_pred : tf.Tensor
        Predicted probabilities of shape (batch_size, n_samples).
        
    Returns
    -------
    tf.Tensor
        Scalar F1-score averaged over the batch.
        
    Notes
    -----
    Uses clipping and epsilon values to avoid division by zero.
    """     
    
    def recall(y_true, y_pred):
        'Recall metric. Only computes a batch-wise average of recall. Computes the recall, a metric for multi-label classification of how many relevant items are selected.'

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=-1)
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=-1)
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        'Precision metric. Only computes a batch-wise average of precision. Computes the precision, a metric for multi-label classification of how many selected items are relevant.'

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=-1)
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=-1)
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def waveform_normalize(
    X: npt.NDArray[np.float32], 
    mode: str = 'max', 
    channel_mode: str = 'local'
) -> npt.NDArray[np.float32]:
    """
    Normalize waveform data by removing mean and scaling.
    
    First removes the mean from each waveform, then scales by either
    the maximum amplitude or standard deviation. Normalization can be
    applied per-channel (local) or globally across all channels.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_channels)
        Raw waveform data to normalize.
    mode : {'max', 'std'}, default='max'
        Normalization method:
        - 'max': Scale by maximum absolute amplitude
        - 'std': Scale by standard deviation
    channel_mode : {'local', 'global'}, default='local'
        Normalization scope:
        - 'local': Normalize each channel independently
        - 'global': Normalize using statistics across all channels
        
    Returns
    -------
    ndarray of shape (n_samples, n_channels)
        Normalized waveforms with zero mean and unit scale.
        
    """

    X -= np.mean(X, axis=0, keepdims=True)

    if mode == 'max':
        if channel_mode == 'local':
            m = np.max(X, axis=0, keepdims=True)
        else:
            m = np.max(X, keepdims=True)
    elif mode == 'std':
        if channel_mode == 'local':
            m = np.std(X, axis=0, keepdims=True)
        else:
            m = np.std(X, keepdims=True)
    else:
        raise NotImplementedError(
            f'Not supported normalization mode: {mode}')

    m[m == 0] = 1
    return X / m

def waveform_crop(
    x: npt.NDArray[np.float32], 
    y: npt.NDArray[np.float32], 
    new_length: int, 
    testing: bool = False
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Crop waveform and label arrays to specified length.
    
    Parameters
    ----------
    x : ndarray of shape (n_samples, n_channels)
        Waveform data to crop.
    y : ndarray of shape (n_samples, n_classes)
        Corresponding label data to crop.
    new_length : int
        Target length in samples for the cropped waveform.
    testing : bool, default=False
        If True, performs center cropping for reproducibility.
        If False, performs random cropping.

    Returns
    -------
    x_cropped : ndarray of shape (new_length, n_channels)
        Cropped waveform data.
    y_cropped : ndarray of shape (new_length, n_classes)
        Cropped label data.
        
    Notes
    -----
    This is the legacy cropping function. Consider using `waveform_crop_new`
    which supports smart cropping to preserve picks.
    """

    if testing: 
        y1 = len(x)//2 - new_length//2 #consistent for testing
    else:
        y1 = np.random.randint(0, len(x) - new_length)
    x = x[y1:y1 + new_length]
    y = y[y1:y1 + new_length]
    return x, y

def waveform_crop_new(
    x: npt.NDArray[np.float32], 
    y: npt.NDArray[np.float32], 
    new_length: int, 
    testing: bool = False, 
    smart_crop: bool = True
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Crop waveform with intelligent pick-preserving strategy.
    
    Improved cropping function that can intelligently position the crop
    window to include seismic phase picks when available, improving
    training efficiency.

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_channels)
        Waveform data to crop.
    y : ndarray of shape (n_samples, n_classes)
        Corresponding label data. Expected to have picks indicated
        by non-zero values in P-wave and S-wave channels.
    new_length : int
        Target length in samples for the cropped waveform.
    testing : bool, default=False
        If True, performs center cropping for reproducibility.
        If False, uses random or smart cropping strategy.
    smart_crop : bool, default=True
        If True, attempts to position the crop window to include
        at least one P or S pick when available. Falls back to
        random cropping if no picks are found.

    Returns
    -------
    x_cropped : ndarray of shape (new_length, n_channels)
        Cropped waveform data.
    y_cropped : ndarray of shape (new_length, n_classes)
        Cropped label data.
        
    Notes
    -----
    Smart cropping searches for non-zero values in the P-wave (channel 1)
    and S-wave (channel 2) of the label array, then positions the crop
    window to include these picks when possible.
    """

    if new_length is None or new_length >= len(x):
        return x, y

    if testing:
        # Center crop for consistent testing
        y1 = len(x) // 2 - new_length // 2
    elif smart_crop:
        # Smart cropping: try to include picks
        y1 = _find_smart_crop_position(y, new_length)
    else:
        # Standard random cropping
        y1 = np.random.randint(0, len(x) - new_length)

    x = x[y1 : y1 + new_length]
    y = y[y1 : y1 + new_length]
    return x, y

def calculate_windows_per_sample(waveform_length, window_length, overlap_seconds, sampling_rate):
    """
    Calculate how many overlapping windows will be created from a waveform.
    
    Parameters
    ----------
    waveform_length : int
        length of the original waveform in samples
    window_length : int
        length of each window in samples
    overlap_seconds : float
        overlap between consecutive windows in seconds
    sampling_rate : float
        sampling rate in Hz
        
    Returns
    -------
    int
        number of windows that will be created
    """
    if window_length >= waveform_length:
        return 1
    
    # Convert overlap from seconds to samples
    overlap_samples = int(overlap_seconds * sampling_rate)
    
    # Calculate step size (how much to advance for each window)
    step_size = window_length - overlap_samples
    
    # Ensure step size is at least 1 sample
    if step_size <= 0:
        step_size = 1
    
    # Calculate number of windows
    num_windows = max(1, (waveform_length - window_length) // step_size + 1)
    return num_windows

def create_overlapping_windows(x, y, window_length, overlap_seconds, sampling_rate, random_start_offset=True, max_offset_seconds=None):
    """
    Create multiple overlapping windows from a single waveform sample.

    Parameters
    ----------
    x : numpy array
        waveforms with shape (time, channels)
    y : numpy array
        labels with shape (time, label_channels)
    window_length : int
        length of each window in samples
    overlap_seconds : float
        overlap between consecutive windows in seconds
    sampling_rate : float
        sampling rate in Hz
    random_start_offset : bool
        if True, add random offset to start time to avoid discrete step patterns
    max_offset_seconds : float, optional
        maximum random offset in seconds. If None, uses step_size in seconds

    Returns
    -------
    windows_x : list of numpy arrays
        list of windowed waveforms
    windows_y : list of numpy arrays
        list of windowed labels
    noise_only_count : int
        number of windows containing only noise (no P or S picks)
    """
    if window_length >= len(x):
        # Check if the single window has picks
        #has_picks = _window_has_picks(y)
        #noise_only_count = 0 if has_picks else 1
        #return [x], [y], noise_only_count
        return [x], [y], 0

    
    # Convert overlap from seconds to samples
    overlap_samples = int(overlap_seconds * sampling_rate)
    
    # Calculate step size (how much to advance for each window)
    step_size = window_length - overlap_samples
    
    # Ensure step size is at least 1 sample
    if step_size <= 0:
        step_size = 1
    
    # Calculate random start offset if enabled
    if random_start_offset:
        if max_offset_seconds is None:
            # Use step size as maximum offset
            max_offset_samples = step_size
        else:
            max_offset_samples = int(max_offset_seconds * sampling_rate)
        
        # Random offset between 0 and max_offset_samples
        random_offset = np.random.randint(0, max_offset_samples + 1)
    else:
        random_offset = 0
    
    windows_x = []
    windows_y = []
    noise_only_count = 0
    window_count = 0
    
    start = random_offset
    while start + window_length <= len(x):
        end = start + window_length
        win_x = x[start:end].copy()
        win_y = y[start:end].copy()
        
        #has_picks = _window_has_picks(win_y)
        #if not has_picks:
        #    noise_only_count += 1
        noise_only_count = 0
        
        windows_x.append(win_x)
        windows_y.append(win_y)
        start += step_size
        window_count += 1
    
    return windows_x, windows_y, noise_only_count

def _window_has_picks(y):
    """
    Check if a window contains any P or S picks.
    
    Parameters
    ----------
    y : numpy array
        labels with shape (time, label_channels)
        
    Returns
    -------
    bool
        True if window contains P or S picks, False if only noise
    """
    if y.shape[1] == 9:
        # 9-phase format: ['PN', 'PG', 'P', 'SN', 'SG', 'S', 'PB', 'SB', 'D']
        # P phases: channels 0, 1, 2, 6 (PN, PG, P, PB)
        # S phases: channels 3, 4, 5, 7 (SN, SG, S, SB)
        p_channels = [0, 1, 2, 6]
        s_channels = [3, 4, 5, 7]
        p_picks = sum(np.sum(y[:, ch] > 0.1) for ch in p_channels)
        s_picks = sum(np.sum(y[:, ch] > 0.1) for ch in s_channels)
        has_picks = (p_picks > 0 or s_picks > 0)
        return has_picks
    elif y.shape[1] == 4:
        # Split-output format: [p_noise, p_pick, s_noise, s_pick]
        p_picks = np.sum(y[:, 1] > 0.1)  # P pick channel
        s_picks = np.sum(y[:, 3] > 0.1)  # S pick channel
        has_picks = (p_picks > 0 or s_picks > 0)
        return has_picks
    elif y.shape[1] >= 3:
        # Standard format: [noise, P, S, ...]
        p_picks = np.sum(y[:, 1] > 0.1)  # P channel
        s_picks = np.sum(y[:, 2] > 0.1)  # S channel
        has_picks = (p_picks > 0 or s_picks > 0)
        return has_picks
    else:
        # Unknown format, assume it has picks to be safe
        return True

def _find_smart_crop_position(y, new_length):
    """
    Find the best crop position to include at least one P or S pick.
    
    Parameters
    ----------
    y : numpy array
        labels with shape (time, channels)
    new_length : int
        desired crop length
        
    Returns
    -------
    int
        starting position for crop
    """
    original_length = len(y)

    # Identify pick locations (depends on label format)
    pick_locations = []

    if y.shape[1] == 4:
        # Split-output format: [p_noise, p_pick, s_noise, s_pick]
        p_picks = np.where(y[:, 1] > 0.1)[0]  # P pick locations
        s_picks = np.where(y[:, 3] > 0.1)[0]  # S pick locations
        pick_locations = np.concatenate([p_picks, s_picks])
    elif y.shape[1] >= 3:
        # Standard format: [noise, P, S, ...]
        p_picks = np.where(y[:, 1] > 0.1)[0]  # P pick locations
        s_picks = np.where(y[:, 2] > 0.1)[0]  # S pick locations
        pick_locations = np.concatenate([p_picks, s_picks])

    # If no picks found, fall back to random cropping
    if len(pick_locations) == 0:
        return np.random.randint(0, original_length - new_length)

    # Find all valid crop positions that include at least one pick
    valid_positions = []
    for pick_loc in pick_locations:
        # For each pick, find crop positions that would include it
        min_start = max(0, pick_loc - new_length + 1)
        max_start = min(original_length - new_length, pick_loc)

        for start_pos in range(min_start, max_start + 1):
            if start_pos not in valid_positions:
                valid_positions.append(start_pos)

    if len(valid_positions) == 0:
        # Fallback to random if something went wrong
        return np.random.randint(0, original_length - new_length)

    # Randomly select from valid positions to maintain some randomness
    return np.random.choice(valid_positions)



def waveform_drop_channel(
    x: npt.NDArray[np.float32], 
    channel: int
) -> npt.NDArray[np.float32]:
    """
    Zero out a specific channel in the waveform.
    
    Used as a data augmentation technique to improve model robustness
    to missing or faulty sensor channels.
    
    Parameters
    ----------
    x : ndarray of shape (n_samples, n_channels)
        Waveform data.
    channel : int
        Index of the channel to zero out (0-based).

    Returns
    -------
    ndarray of shape (n_samples, n_channels)
        Waveform with the specified channel set to zero.
        
    Notes
    -----
    This function modifies the array in-place for efficiency.
    """

    x[..., channel] = 0
    return x

def waveform_add_gap(
    x: npt.NDArray[np.float32], 
    max_size: float
) -> npt.NDArray[np.float32]:
    """
    Simulate a data gap by zeroing a random segment of the waveform.
    
    Helps models learn to handle missing data and telemetry failures
    commonly encountered in real-world seismic monitoring.
    
    Parameters
    ----------
    x : ndarray of shape (n_samples, n_channels)
        Waveform data.
    max_size : float
        Maximum gap size as a fraction of total waveform length (0.0 to 1.0).
        The actual gap size is randomly chosen between 0 and max_size.

    Returns
    -------
    ndarray of shape (n_samples, n_channels)
        Waveform with a random segment zeroed out.
        
    Notes
    -----
    The gap position is randomly selected. The gap affects all channels
    simultaneously, simulating a complete station outage.
    """

    l = x.shape[0]
    gap_start = np.random.randint(0, int((1 - max_size) * l))
    gap_end = np.random.randint(gap_start, gap_start + int(max_size * l))
    x[gap_start:gap_end] = 0
    return x

def waveform_add_noise(
    x: npt.NDArray[np.float32], 
    noise: float
) -> npt.NDArray[np.float32]:
    """
    Add Gaussian noise scaled by waveform amplitude.
    
    Augments training data by adding channel-wise Gaussian noise
    with standard deviation proportional to each channel's amplitude.
    
    Parameters
    ----------
    x : ndarray of shape (n_samples, n_channels)
        Waveform data.
    noise : float
        Noise level multiplier. The noise standard deviation for each
        channel will be `noise * max_amplitude_of_channel`.

    Returns
    -------
    ndarray of shape (n_samples, n_channels)
        Waveform with added Gaussian noise.
        
    Notes
    -----
    Noise is added independently to each channel, with scaling
    proportional to that channel's maximum amplitude. This preserves
    the relative signal-to-noise characteristics across channels.
    """

    m = x.max(axis=0)
    N = np.random.normal(scale=m * noise, size=x.shape)
    return x + N

def waveform_taper(
    x: npt.NDArray[np.float32], 
    alpha: float = 0.04
) -> npt.NDArray[np.float32]:
    """
    Apply Tukey window taper to waveform edges.
    
    Smoothly tapers the waveform to zero at both ends to reduce
    edge effects and spectral leakage during windowing operations.
    
    Parameters
    ----------
    x : ndarray of shape (n_samples, n_channels)
        Waveform data.
    alpha : float, default=0.04
        Shape parameter of the Tukey window, representing the fraction
        of the window inside the cosine tapered region. 0 <= alpha <= 1.
        - alpha = 0: rectangular window (no tapering)
        - alpha = 1: Hann window (maximum tapering)

    Returns
    -------
    ndarray of shape (n_samples, n_channels)
        Tapered waveform data.
        
    Notes
    -----
    The taper is applied identically to all channels.
    """
    w = tukey(x.shape[0], alpha)
    return x*w[:,np.newaxis]

def label_smoothing(
    y: npt.NDArray[np.float32], 
    f: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """
    Smooth phase pick labels by convolving with a Gaussian kernel.
    
    Adds temporal uncertainty to discrete pick times, creating softer
    target distributions that can improve model training and better
    represent pick uncertainty.
    
    Parameters
    ----------
    y : ndarray of shape (n_samples, n_classes)
        Binary label array where non-zero values indicate phase picks.
    f : ndarray of shape (kernel_size,)
        Gaussian convolution kernel. Width is typically controlled by
        the 'ramp' parameter in the configuration file.

    Returns
    -------
    ndarray of shape (n_samples, n_classes)
        Smoothed labels, clipped to [0, 1] and normalized so each
        channel's maximum value is 1.
        
    Notes
    -----
    Each label channel is convolved independently. The function normalizes
    each channel after convolution to maintain unit maximum amplitude.
    """
    y = np.asarray([np.convolve(b, f, mode='same') for b in y.T]).T
    m = np.amax(y, axis=0, keepdims=True)
    m[m == 0] = 1
    y /= m
    return np.clip(y, 0.0, 1.0)

def label_smoothing_dual(y, gaussian_kernel):
    """
    Smooth only P and S channels in a label array, ignoring noise.

    Parameters
    ----------
    y : numpy array
        Labels with shape (time, channels) where channels can be:
        - 3 channels: [noise, P, S] 
        - 4 channels: [p_noise, p_pick, s_noise, s_pick]
    gaussian_kernel : numpy array
        Gaussian kernel controlled by ramp parameter in config file

    Returns
    -------
    y_out : numpy array
        Smoothed labels with same shape, where only P and S channels are smoothed
    """
    y_out = y.copy()

    if y.shape[1] == 4:
        # 4-channel format: [p_noise, p_pick, s_noise, s_pick]
        # Only smooth P and S pick channels (indices 1 and 3)
        for chan in [1, 3]:  # p_pick and s_pick channels
            tmp = np.convolve(y_out[:, chan], gaussian_kernel, mode='same')
            m = np.max(tmp)
            if m < 1e-8:
                m = 1.0
            y_out[:, chan] = np.clip(tmp / m, 0.0, 1.0)

        # Regenerate noise channels as 1 - pick
        y_out[:, 0] = 1.0 - y_out[:, 1]  # p_noise = 1 - p_pick
        y_out[:, 2] = 1.0 - y_out[:, 3]  # s_noise = 1 - s_pick

    else:
        # 3-channel format: [noise, P, S]
        # Only smooth P and S channels (indices 1 and 2)
        for chan in [1, 2]:  # skip noise channel (0)
            tmp = np.convolve(y_out[:, chan], gaussian_kernel, mode='same')
            m = np.max(tmp)
            if m < 1e-8:
                m = 1.0
            y_out[:, chan] = np.clip(tmp / m, 0.0, 1.0)

        # Zero out the noise channel since we'll regenerate it
        y_out[:, 0] = 0.0

    return y_out


class EQDatareader(tf.keras.utils.Sequence):
    """
    Data generator for loading and augmenting seismic waveform data.
    
    A Keras Sequence class that handles batch loading of earthquake waveforms
    from HDF5 files with on-the-fly data augmentation including noise addition,
    channel dropout, gap simulation, and smart cropping to preserve phase picks.
    
    Parameters
    ----------
    files : list of tuple
        List of (waveform_file, label_file) pairs pointing to HDF5 datasets.
    augment : bool, default=False
        Enable data augmentation (noise, gaps, channel dropout).
    batch_size : int, default=32
        Number of samples per batch.
    new_length : int, optional
        Target waveform length in samples. If None, uses original length.
    taper_alpha : float, default=0.01
        Tukey window parameter for edge tapering.
    add_noise : float, default=0.0
        Probability of adding Gaussian noise to a sample.
    add_event : float, default=0.0
        Probability of adding an event waveform to noise samples.
    drop_channel : float, default=0.0
        Probability of zeroing out a random channel.
    add_gap : float, default=0.0
        Probability of adding a data gap.
    max_gap_size : float, default=0.0
        Maximum gap size as fraction of waveform length.
    norm_mode : {'max', 'std'}, default='max'
        Normalization method for waveforms.
    norm_channel_mode : {'local', 'global'}, default='global'
        Apply normalization per-channel or globally.
    include_noise : bool, default=False
        Include noise-only samples in the dataset.
    testing : bool, default=False
        If True, uses deterministic center cropping instead of random.
    ramp : int, default=0
        Width of Gaussian kernel for label smoothing (0 = no smoothing).
    shuffle : bool, default=False
        Shuffle samples at each epoch.
    beamlabel : bool or str, default=False
        Use beam-formed data as labels.
    holdout : bool or list, default=False
        Event IDs to hold out from training.
    extract_array_channels : bool or list, default=False
        Indices of specific channels to extract from multi-channel arrays.
    remove_zero_channels : bool, default=True
        Remove samples with all-zero channels.
    modeltype : str, default='transphasenet'
        Model architecture type, affects label formatting.
    smart_crop : bool, default=False
        Enable intelligent cropping to preserve phase picks.
    is_dual_decoder : bool, default=False
        Format labels for split-output models (separate P and S branches).
    use_overlapping_windows : bool, default=False
        Generate overlapping windows from each waveform.
    overlap_seconds : float, default=150.0
        Overlap duration in seconds when using overlapping windows.
    sampling_rate : float, default=40.0
        Sampling rate in Hz, used for overlap calculations.
    validation_mode : bool, default=False
        Special mode for validation data handling.
        
    Attributes
    ----------
    data : list
        Loaded waveforms and labels from all files.
    ramp_filter : ndarray or None
        Precomputed Gaussian kernel for label smoothing.
        
    Notes
    -----
    - Smart cropping searches for P and S picks in labels and positions
      the crop window to include at least one pick when possible.
    - Overlapping windows mode generates multiple windows per waveform,
      useful for validation to ensure complete coverage.
    - The class inherits from tf.keras.utils.Sequence for efficient
      multiprocessing support during training.
    """
    def __init__(self, 
         files, 
         augment=False,
         batch_size=32,
         new_length=None,
         taper_alpha=0.01,
         add_noise=0.0, 
         add_event=0.0,
         drop_channel=0.0,
         add_gap=0.0,
         max_gap_size=0.0,
         norm_mode='max',
         norm_channel_mode='global',
         fill_value=0.0,
         include_noise=False,
         testing=False,
         ramp=0,
         file_buffer=-1,
         shuffle=False,
         beamlabel=False,
         holdout=False,
         extract_array_channels=False,
         remove_zero_channels=True,
         modeltype='transphasenet',  # Add modeltype parameter with default value
         smart_crop=False,  # Add smart_crop parameter with default value
         is_dual_decoder=False,
         use_overlapping_windows=False,  # Add overlapping windows parameter
         overlap_seconds=150.0,  # Add overlap in seconds parameter
         sampling_rate=40.0,  # Add sampling rate parameter
         validation_mode=False,  # Add validation mode parameter
    ) -> None:

        super().__init__()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.augment = augment
        self.new_length = new_length
        self.taper_alpha = taper_alpha
        self.add_noise = add_noise
        self.drop_channel = drop_channel
        self.add_event = add_event
        self.add_gap = add_gap
        self.norm_mode = norm_mode
        self.norm_channel_mode = norm_channel_mode
        self.fill_value = fill_value
        self.max_gap_size = max_gap_size
        self.include_noise = include_noise
        self.testing = testing 
        self.ramp = ramp 
        self.f = gaussian(201, ramp)
        self.file_buffer = file_buffer
        self.files = files
        self.beamlabel = beamlabel
        self.modeltype = modeltype  # Set modeltype as instance variable
        self.smart_crop = smart_crop  # Set smart_crop as instance variable
        self.use_overlapping_windows = use_overlapping_windows
        self.overlap_seconds = overlap_seconds
        self.sampling_rate = sampling_rate
        self.validation_mode = validation_mode
        
        # Note: Overlapping windows statistics removed since we now use lazy loading
        # Check if we're using the dual-decoder model
        #is_dual_decoder = self.modeltype == 'splitoutputtransphasenet'
        if self.modeltype.startswith('splitoutput'): self.is_dual_decoder = True
        else: self.is_dual_decoder = False
        
        self.data = []

        if self.file_buffer < 0:
            if not self.include_noise:
                for file,file_labels in tqdm(zip(self.files[0],self.files[1]),total=len(self.files[0])):
                    self._load_file([file,file_labels,None],self.beamlabel,holdout,extract_array_channels,remove_zero_channels)
            else :
                for file,file_labels,file_noise in tqdm(zip(self.files[0],self.files[1],self.files[2]),total=len(self.files[0])):
                    self._load_file([file,file_labels,file_noise],self.beamlabel,holdout,extract_array_channels,remove_zero_channels)
        print('Number of samples:', len(self.data))
        
        # Note: Overlapping windows are now created on-demand during training
        # to prevent memory explosion during initialization
        
        self.on_epoch_end()

    def _label_decoder_splitmodel(self, lab, w, x, y):
        """
        Decode raw labels into probability matrix, properly mapping phase types to P and S channels.
        
        Based on config phase list ['PN', 'PG', 'P', 'SN', 'SG', 'S', 'PB', 'SB', 'D']:
        - P phases: [0, 1, 2, 6] (PN, PG, P, PB)
        - S phases: [3, 4, 5, 7] (SN, SG, S, SB)
        - Phase 8: D (detection/noise/background windows)
        """
        # Define phase type mappings based on the actual phase list
        P_PHASES = [0, 1, 2, 6]  # PN, PG, P, PB
        S_PHASES = [3, 4, 5, 7]  # SN, SG, S, SB

        # Always use 3-channel format for modern phase detection: [noise, P, S]
        # This fixes the issue where raw labels had 9 phase types but we want 3 channels
        target_channels = 3
        prob = np.zeros((w, x, target_channels), dtype=np.float32)
        prob[:, :, 0] = 1.0  # Initialize all as noise

        # Group labels by event
        lab_dict = defaultdict(list)
        for l in lab:
            lab_dict[l[0]].append(l)

        for i in range(w):
            for l in lab_dict[i]:
                start_time = int(l[1])
                end_time = int(l[2])
                phase_type = int(l[3])

                # Use proper phase mapping for 3-channel output
                # Skip noise windows (phase 8 with duration > 0)
                if phase_type == 8 and end_time > start_time:
                    continue

                # Only process point picks (duration = 0)
                if end_time == start_time:
                    # Bounds check to prevent index errors
                    if start_time >= x:
                        continue  # Skip picks at or beyond the end of the waveform

                    if phase_type in P_PHASES:
                        # P pick
                        prob[i, start_time, 0] = 0.0  # Remove noise
                        prob[i, start_time, 1] = 1.0  # Set P pick
                    elif phase_type in S_PHASES:
                        # S pick
                        prob[i, start_time, 0] = 0.0  # Remove noise
                        prob[i, start_time, 2] = 1.0  # Set S pick
                    # Ignore other phase types (phase 2=P, 5=S not in our mapping) - keep as noise

        return prob


    def _label_decoder(self,lab,w,x,y):
        prob = np.zeros((w,x,y))
        lab_dict = defaultdict(list)
        for l in lab:
            lab_dict[l[0]].append(l)
        for i in range(w) :
            for l in lab_dict[i] :
                prob[i,l[1]:l[2]+1,l[3]] = 1.
        return prob


    def _load_file(self, filename, beamlabel,holdout=False,extract_array_channels=False,remove_zero_channels=True):
        with h5py.File(filename[0]) as f:
            x = f['X'][:]
            ids = f['event_id'][:]
            arrival_ids = f['arrivals'][:]
            stations = f['station'][:]
        if not beamlabel or 'ADD' in beamlabel :
            with h5py.File(filename[1]) as f:
                labels = f['labels'][:]
                ids2 = f['event_id'][:]
        if beamlabel  :
            inputdir = '/'.join(filename[0].split('/')[:-1])
            with h5py.File('_'.join([inputdir+'/'+beamlabel.split('_')[1]]+[filename[0].split('_')[1]]+['beams.hdf5'])) as f:
                labels_2 = f['X'][:]
                ids2_2 = f['event_id'][:]
                stations2 = f['station'][:]
        decoded = False
        if not np.array_equal(ids,ids2) and ( not beamlabel or 'ADD' in  beamlabel ) :
            print(f"Data and labels are not equal {len(ids)} {len(ids2)}")
            print('Attempting to find same event IDs in labels as in data ...')
            num_labels = np.max(np.transpose(labels)[3])
            if self.is_dual_decoder: 
                labels = self._label_decoder_splitmodel(labels,len(ids2),len(x[0]),num_labels+1)
            else :
                labels = self._label_decoder(labels,len(ids2),len(x[0]),num_labels+1)
            if all(item in ids2 for item in ids) :
                labels = [lab for i,lab in enumerate(labels) if ids2[i] in ids]
                ids2 = [i for i in ids2 if i in ids]
                decoded = True
            else :
                print('Not all event IDs in data found in labels')
                exit()
        if (not beamlabel or 'ADD' in beamlabel ) and not decoded :
            num_labels = np.max(np.transpose(labels)[3])
            if self.is_dual_decoder :
                labels = self._label_decoder_splitmodel(labels,len(x),len(x[0]),num_labels+1)
            else : 
                labels = self._label_decoder(labels,len(x),len(x[0]),num_labels+1)

        if extract_array_channels :
            #print(x.shape,len(labels))
            x = np.take(x, extract_array_channels, axis=2)
            # remove all events where at least one channel is zero
            if remove_zero_channels :
                channel_has_only_zeros = (x == 0).all(axis=1)
                event_has_bad_channel = channel_has_only_zeros.any(axis=1)
                keep_mask = ~event_has_bad_channel
                x = x[keep_mask]
                ids = ids[keep_mask]
                arrival_ids = arrival_ids[keep_mask]
                stations = stations[keep_mask]
                labels = labels[keep_mask]
                ids2 = ids2[keep_mask]
           #print(x.shape,len(labels))

        counter = 0
        for _id, waveform, label, arids, stat in zip(ids, x, labels, arrival_ids, stations ):
        #for _id, waveform, label, et in zip(ids, x, labels, event_type):
            et = 'event'
            if et == 'noise' : continue
            if holdout :
                if _id.decode("utf-8") in holdout and not self.testing :
                    #print('Leaving out event for training',_id)
                    continue
                if _id.decode("utf-8") not in holdout and self.testing :
                    #print('Leaving out event for testing',_id)
                    continue
            if beamlabel :
                idx=np.where((ids2_2 ==_id) & (stations2 == stat))[0]
                if len(idx) == 0 : continue
                label2=labels_2[idx][0]


            if (not beamlabel or 'ADD' in beamlabel) and self.ramp > 0:
                if self.is_dual_decoder:
                    # For split output model, we need to create 4 channels:
                    # [p_noise, p_pick, s_noise, s_pick]
                    # First get the original P and S channels
                    p_pick = label[:, 1]  # P channel
                    s_pick = label[:, 2]  # S channel
                    
                    # Create noise channels as 1 - pick
                    p_noise = 1.0 - p_pick
                    s_noise = 1.0 - s_pick
                    
                    # Stack into 4 channels
                    label = np.stack([p_noise, p_pick, s_noise, s_pick], axis=-1)
                    
                    # Apply smoothing if needed
                    if self.ramp > 0:
                        label = label_smoothing_dual(label, self.f)
                else:
                    # Single-head approach: original smoothing
                    label = label_smoothing(label, self.f)
                    label = np.clip(label, 0.0, 1.0)  # be safe

            if beamlabel :
                if 'ADD' in beamlabel : label = np.concatenate((label,label2),axis=1)

            # Store original waveform for lazy loading (overlapping windows created on-demand)
            self.data.append({'x':waveform.astype(np.float32), 
                              'y':label.astype(np.float32), 
                              'event_id': _id,
                              'arrival_ids': arids,
                              'station': stat,
                              'et':et})
            counter += 1
        #print("Data read:",counter,filename[0],len(self.data))
        if self.include_noise:
            nlabels = label.shape[-1]
            with h5py.File(filename[2]) as f:
                x = f['X'][:]
            if extract_array_channels :
                x = np.take(x, extract_array_channels, axis=2)
                if remove_zero_channels :
                    channel_has_only_zeros = (x == 0).all(axis=1)
                    event_has_bad_channel = channel_has_only_zeros.any(axis=1)
                    keep_mask = ~event_has_bad_channel
                    x = x[keep_mask]
            # hard-coded noise not more than 25%
            for waveform in x[:int(len(self.data)/4.)]:
                if self.is_dual_decoder:
                    # For noise data in dual-decoder case, create 4-channel output
                    # All noise channels are 1.0, all pick channels are 0.0
                    label = np.zeros((waveform.shape[0], 4), dtype=np.float32)
                    label[:, 0] = 1.0  # p_noise
                    label[:, 2] = 1.0  # s_noise
                else:
                    # For single-head typical approach
                    label = np.zeros((waveform.shape[0], nlabels), dtype=np.float32)
                    label[:, 0] = 1.0   # noise channel = 1.0 (P and S remain 0)

                et = 'noise'
                self.data.append({'x':waveform.astype(np.float32),
                              'y':label.astype(np.float32),
                              'event_id': None,
                              'arrival_ids': None,
                              'station': None,
                              'et':et})
            
    def on_epoch_end(self):
        if self.file_buffer > 0:
            for file in choices(self.files, k=self.file_buffer):
                self._load_file(file,self.beamlabel,holdout)
                
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            shuffle(self.indexes)
            
    def __len__(self):
        if self.use_overlapping_windows and not self.testing and self.new_length is not None:
            # Calculate windows per sample (should be consistent across all samples)
            if not hasattr(self, '_windows_per_sample') or self._windows_per_sample is None:
                # Calculate once and cache it
                if len(self.indexes) > 0:
                    sample_id = self.indexes[0]
                    sample_x = self.data[sample_id]['x'].copy()
                    self._windows_per_sample = calculate_windows_per_sample(
                        len(sample_x), self.new_length, self.overlap_seconds, self.sampling_rate
                    )
                else:
                    self._windows_per_sample = 1
            
            # Calculate adjusted batch size
            adjusted_batch_size = max(1, self.batch_size // self._windows_per_sample)
            return int(np.floor(len(self.indexes) / adjusted_batch_size))
        else:
            return int(np.floor(len(self.indexes) / self.batch_size))
    
    def __getitem__(self, item):
        # Calculate adjusted batch size for overlapping windows
        if self.use_overlapping_windows and not self.testing and self.new_length is not None:
            # Calculate windows per sample (should be consistent across all samples)
            if not hasattr(self, '_windows_per_sample') or self._windows_per_sample is None:
                # Calculate once and cache it
                if len(self.indexes) > 0:
                    sample_id = self.indexes[0]
                    sample_x = self.data[sample_id]['x'].copy()
                    self._windows_per_sample = calculate_windows_per_sample(
                        len(sample_x), self.new_length, self.overlap_seconds, self.sampling_rate
                    )
                else:
                    self._windows_per_sample = 1
            
            # Adjust base batch size to maintain target final batch size
            adjusted_batch_size = max(1, self.batch_size // self._windows_per_sample)
            start_idx = item * adjusted_batch_size
            end_idx = min((item + 1) * adjusted_batch_size, len(self.indexes))
            ids = self.indexes[start_idx:end_idx]
        else:
            # Original behavior: use configured batch size
            ids = self.indexes[item * self.batch_size:(item + 1) * self.batch_size]
        
        # Generate base samples
        base_samples = list(map(self.data_generation, ids))
        
        # Handle overlapping windows if enabled
        if self.use_overlapping_windows and not self.testing and self.new_length is not None:
            all_X = []
            all_y = []
            
            # Debug output (only print once per epoch at batch 0)
            if item == 0 and not hasattr(self, '_debug_printed'):
                mode = "VALIDATION" if self.validation_mode else "TRAINING"
                print(f"\n{'='*60}")
                print(f"DEBUG: Overlapping windows ENABLED for {mode}")
                print(f"       Window length: {self.new_length} samples ({self.new_length/self.sampling_rate:.1f} seconds)")
                print(f"       Overlap: {self.overlap_seconds} seconds")
                print(f"       Random offset: {'NO (deterministic)' if self.validation_mode else 'YES (random)'}")
                print(f"       Base samples in this batch: {len(base_samples)}")
                self._debug_printed = True
            
            for x, y in base_samples:
                # Create overlapping windows from the base sample
                # For training: add random start offset to avoid discrete step patterns
                # For validation: use deterministic windows (no random offset)
                if self.validation_mode:
                    # Validation mode: no random offset, deterministic windows
                    windows_x, windows_y, _ = create_overlapping_windows(
                        x, y, self.new_length, self.overlap_seconds, self.sampling_rate,
                        random_start_offset=False
                    )
                else:
                    # Training mode: use random offset
                    step_size_seconds = (self.new_length - int(self.overlap_seconds * self.sampling_rate)) / self.sampling_rate
                    windows_x, windows_y, _ = create_overlapping_windows(
                        x, y, self.new_length, self.overlap_seconds, self.sampling_rate,
                        random_start_offset=True, max_offset_seconds=step_size_seconds
                    )
                
                # Apply final processing to each window
                processed_windows_x = []
                processed_windows_y = []
                
                for win_x, win_y in zip(windows_x, windows_y):
                    # Apply taper
                    if self.taper_alpha > 0:
                        win_x = waveform_taper(win_x, self.taper_alpha)
                    
                    # Apply normalization
                    if self.norm_mode is not None:
                        win_x = waveform_normalize(win_x, mode=self.norm_mode, channel_mode=self.norm_channel_mode)
                    
                    # Handle NaN values
                    win_x[np.isnan(win_x)] = self.fill_value
                    win_y[np.isnan(win_y)] = self.fill_value
                    
                    processed_windows_x.append(win_x)
                    processed_windows_y.append(win_y)
                
                # Add all processed windows to the batch
                all_X.extend(processed_windows_x)
                all_y.extend(processed_windows_y)
            
            # Stack all windows into batch format
            if len(all_X) > 0:
                X = np.stack(all_X, axis=0)
                y = np.stack(all_y, axis=0)
                
                # Debug output (only for first batch)
                if item == 0 and hasattr(self, '_debug_printed'):
                    windows_per_sample = len(all_X) / len(base_samples) if len(base_samples) > 0 else 0
                    print(f"       Windows per sample: {windows_per_sample:.1f}")
                    print(f"       Final batch size: {X.shape[0]} samples")
                    print(f"{'='*60}\n")
                    # Reset flag for next epoch
                    delattr(self, '_debug_printed')
            else:
                # Fallback if no windows created
                X, y = zip(*base_samples)
                y = np.stack(y, axis=0)
                X = np.stack(X, axis=0)
        else:
            # Original behavior: no overlapping windows
            X, y = zip(*base_samples)
            y = np.stack(y, axis=0)
            X = np.stack(X, axis=0)

        # Ensure consistent label dimension based on model type
        if self.is_dual_decoder:
            # For SplitOutputTransPhaseNet, we need 4 channels
            # Don't split here - DropDetection will do the splitting for dual-decoder
            if y.shape[-1] == 3:
                # If we have 3 channels [noise, P, S] but need 4 [p_noise, p_pick, s_noise, s_pick]
                # Convert to appropriate 4-channel format
                noise = y[..., 0:1]
                p_pick = y[..., 1:2]
                s_pick = y[..., 2:3]
                p_noise = 1.0 - p_pick
                s_noise = 1.0 - s_pick
                y = np.concatenate([p_noise, p_pick, s_noise, s_pick], axis=-1)
        else :
            y = np.split(y, y.shape[-1], axis=-1)

        return X, y
        
    def data_generation(self, _id):
        
        x = self.data[_id]['x'].copy()
        y = self.data[_id]['y'].copy()
        event_type = self.data[_id]['et']
        
        # Apply data augmentation FIRST (on full waveforms)
        if self.augment:
            if event_type == 'noise':
                if np.random.random() < self.drop_channel:
                    x = waveform_drop_channel(x, np.random.choice(np.arange(x.shape[1])))
                if np.random.random() < self.add_gap:
                    x = waveform_add_gap(x, self.max_gap_size)
            
            else:
                if np.random.random() < self.add_event:
                    second = random.choice(self.data)
                    if second['et'] != 'noise':
                        second_x = second['x']
                        second_y = second['y']
                        
                        roll = np.random.randint(0, second_y.shape[0])
                        second_x = np.roll(second_x, roll, axis=0)
                        second_y = np.roll(second_y, roll, axis=0)
                        
                        scale = 1/np.random.uniform(1,10)
                        
                        second_x *= scale
                        second_y *= scale
                        
                        x += second_x
                        y = np.amax([y, second_y], axis=0)
                
                if np.random.random() < self.add_noise:
                    x = waveform_add_noise(x, np.random.uniform(0.01,0.15))
                if np.random.random() < self.drop_channel:
                    x = waveform_drop_channel(x, np.random.choice(np.arange(x.shape[1])))
                if np.random.random() < self.add_gap:
                    x = waveform_add_gap(x, self.max_gap_size)
        
        # Apply cropping AFTER augmentation (skip if overlapping windows will be used)
        if not (self.use_overlapping_windows and not self.testing and self.new_length is not None):
            if self.smart_crop:
                x, y = waveform_crop_new(x, y, self.new_length, self.testing, self.smart_crop)
            else:
                x, y = waveform_crop(x, y, self.new_length, self.testing)
        
        # Apply final processing
        if self.taper_alpha > 0:
            x = waveform_taper(x, self.taper_alpha)
        
        if self.norm_mode is not None:
            x = waveform_normalize(x, mode=self.norm_mode, channel_mode=self.norm_channel_mode)
        
        x[np.isnan(x)] = self.fill_value
        y[np.isnan(y)] = self.fill_value
        
        return x, y
    
    
def create_class_weights(cw, y):
    sw = np.take(np.array(cw), np.argmax(y, axis=-1))
    return sw

def process_split_output_labels(raw_labels):
    """
    Process raw label arrays for the dual-decoder (split-output) model.

    Handles two cases:
      (time,3) => old approach [noise, P, S]
      (time,4) => new approach [p_noise, p_pick, s_noise, s_pick]

    Parameters
    ----------
    raw_labels : numpy.ndarray
        The raw label array of shape (batch, time, channels).
        For 3 channels: [noise, P, S]
        For 4 channels: [p_noise, p_pick, s_noise, s_pick]

    Returns
    -------
    tuple:
        p_target : numpy.ndarray
            Target for P-head with shape (batch, time, 1) - binary P probability (0 or 1)
        s_target : numpy.ndarray
            Target for S-head with shape (batch, time, 1) - binary S probability (0 or 1)
    """
    # Safety check for potential issues
    if not isinstance(raw_labels, np.ndarray):
        print(f"WARNING: process_split_output_labels received non-ndarray type: {type(raw_labels)}")
        raw_labels = np.asarray(raw_labels, dtype=np.float32)

    # Ensure we have at least a 3D tensor (batch, time, channels)
    if raw_labels.ndim < 3:
        if raw_labels.ndim == 2:
            # Assume (time, channels), add batch dimension
            raw_labels = raw_labels[np.newaxis, ...]
            print(f"WARNING: Added batch dimension to labels, now shape={raw_labels.shape}")
        else:
            raise ValueError(f"Expected at least 2D array, got shape {raw_labels.shape}")

    # shape is either (batch,time,3) or (batch,time,4)
    c = raw_labels.shape[-1]

    if c == 4:
        # We already have: [p_noise, p_pick, s_noise, s_pick]
        # For binary classification, we want the pick probability (channel 1 and 3)
        p_target = raw_labels[..., 1:2]  # [p_pick] - single channel
        s_target = raw_labels[..., 3:4]  # [s_pick] - single channel
        return p_target, s_target

    elif c >= 3:
        # old code for single array [noise, P, S]
        p_channel = raw_labels[..., 1:2]  # P picks - keep as single channel
        s_channel = raw_labels[..., 2:3]  # S picks - keep as single channel
        return p_channel, s_channel

    else:
        raise ValueError(f"Expected at least 3 channels, got shape {raw_labels.shape}")


class DropDetection(tf.keras.utils.Sequence):
    """
    Wrapper generator for handling class weights and label formatting.
    
    Wraps an EQDatareader to provide class-weighted samples and format
    labels for different model architectures (standard vs split-output).
    
    Parameters
    ----------
    super_sequence : EQDatareader
        Underlying data generator that provides waveforms and labels.
    p_classes : list of int
        Label indices corresponding to P-wave picks.
    s_classes : list of int
        Label indices corresponding to S-wave picks.
    class_weights : list of float, optional
        Sample weights for each class [noise, P, S] for loss computation.
    distance_weighting : bool, default=False
        Apply distance-based weighting to samples.
    beamlabel : bool or str, default=False
        Using beam-formed labels flag.
    modeltype : str, default='transphasenet'
        Model architecture type. 'splitoutput*' models get special
        label formatting with separate P and S branches.
        
    Notes
    -----
    For split-output models, this class reformats standard 3-channel labels
    [noise, P, S] into separate binary outputs for P and S branches, enabling
    independent task-specific decoder heads.
    """

    _has_printed_debug = False  # Class variable to track if debug info has been printed

    def __init__(self, 
                 super_sequence, 
                 p_classes, 
                 s_classes,
                 class_weights=None,
                 distance_weighting=False,
                 beamlabel=False,
                 modeltype='transphasenet'):
        self.super_sequence = super_sequence
        self.p_classes = p_classes
        self.s_classes = s_classes
        self.beamlabel = beamlabel
        self.class_weights = class_weights
        self.distance_weighting = distance_weighting
        self.modeltype = modeltype
       
    def __len__(self):
        return len(self.super_sequence)
    def __getitem__(self, idx):
        if self.modeltype.startswith('splitoutput'):
            return self._getitem_splitmodel(idx)
        else:
            return self._getitem_default(idx)
            
    def _getitem_splitmodel(self, idx):
        data = self.super_sequence[idx]

        batch_x, batch_y = data
        if not self._has_printed_debug:
            print(f"[DEBUG] 'data' len=2 -> X.shape={np.shape(batch_x)} Y.shape={np.shape(batch_y)}")

        if not self._has_printed_debug:
            print(f"[DEBUG] batch_x.shape={np.shape(batch_x)}")

        # Distance-based weighting placeholder
        dist_w = np.ones((len(batch_x),), dtype=np.float32)

        # 1) If beamlabel but no 'ADD', we assume 2-channel single-head
        if self.beamlabel and 'ADD' not in str(self.beamlabel):
            y = np.asarray(batch_y, dtype=np.float32)
            if not self._has_printed_debug:
                print(f"[DEBUG] beamlabel w/o ADD => y.shape={y.shape}")
            sw = np.ones((len(batch_x), y.shape[1]), dtype=np.float32) * dist_w[:, None]
            return batch_x, y, sw

        # 2) Multi-head if 'splitoutputtransphasenet'
        if self.modeltype.startswith('splitoutput'):
            if isinstance(batch_y, dict):
                lab = batch_y.get('labels', None)
                if not self._has_printed_debug:
                    print(f"[DEBUG] SplitOutputTransPhaseNet: batch_y is dict with keys={list(batch_y.keys())}")
            else:
                lab = batch_y
                if not self._has_printed_debug and isinstance(batch_y, np.ndarray):
                    print(f"[DEBUG] SplitOutputTransPhaseNet: input shape={batch_y.shape}")

            # Ensure lab is in the right format for processing
            if isinstance(lab, list):
                # If lab is already split (e.g., a list of arrays)
                # Check if it's from previous splitting or is raw data
                if len(lab) == 1:
                    # Single element list - just take the first element
                    lab = lab[0]
                elif len(lab) in [3, 4]:
                    # This might be already split into channels
                    # We need to reconstruct it for process_split_output_labels
                    lab = np.concatenate(lab, axis=-1)
                else:
                    # Some other list format we don't recognize
                    if not self._has_printed_debug:
                        print(f"[WARNING] Unexpected lab format: list of {len(lab)} elements")

            # Convert to numpy array if not already
            lab = np.asarray(lab, dtype=np.float32)
            if not self._has_printed_debug:
                print(f"[DEBUG] lab initial shape={lab.shape}")

            # Auto-fix for transposed shape, e.g., (9,16,12000,1) => want (16,12000,9)
            if lab.ndim == 4:
                if lab.shape[1] == len(batch_x) and lab.shape[0] != len(batch_x):
                    if not self._has_printed_debug:
                        print(f"[DEBUG] Reordering lab from {lab.shape} to fix transposition")
                    lab = np.transpose(lab, (1, 2, 0, 3))  # => (16,12000,9,1)
                    lab = np.squeeze(lab, axis=-1)  # => (16,12000,9)
                    if not self._has_printed_debug:
                        print(f"[DEBUG] lab after transpose => shape={lab.shape}")

            # Now we expect shape=(batch, time, >=3)
            if lab.ndim < 3 or lab.shape[-1] < 3:
                print(f"WARNING: SplitOutputTransPhaseNet expects at least 3 channels, got shape {lab.shape}")
                # Try to recover by creating a proper shape
                if lab.ndim == 2:
                    # Expand to 3D with at least 4 channels
                    temp = np.zeros((lab.shape[0], lab.shape[1], 4), dtype=np.float32)
                    # Copy whatever data we have
                    temp[:, :, :min(lab.shape[-1], 4)] = lab[:, :, :min(lab.shape[-1], 4)]
                    lab = temp

            # Check for extreme values that might cause NaN
            if np.any(np.isnan(lab)) or np.any(np.isinf(lab)):
                print(f"WARNING: Found NaN/Inf in SplitOutputTransPhaseNet label batch")
            max_val = np.max(np.abs(lab))
            if max_val > 10:
                print(f"WARNING: Extreme value {max_val} in SplitOutputTransPhaseNet label batch")

            # Process labels to get proper one-hot targets for each head
            p_label, s_label = process_split_output_labels(lab)

            if not self._has_printed_debug:
                print(f"[DEBUG] SplitOutputTransPhaseNet: processed p_shape={p_label.shape}, s_shape={s_label.shape}")
                self.__class__._has_printed_debug = True  # Set flag to avoid printing again

            # Initialize sample weights for each head
            sw_p = np.ones((len(batch_x), p_label.shape[1]), dtype=np.float32)
            sw_s = np.ones_like(sw_p)

            # Apply class weights if provided
            if self.class_weights and len(self.class_weights) >= 3:
                # Normalize weights for each head
                p_total = self.class_weights[0] + self.class_weights[1]  # noise + P
                s_total = self.class_weights[0] + self.class_weights[2]  # noise + S

                p_weights = np.array([self.class_weights[0] / p_total, self.class_weights[1] / p_total])
                s_weights = np.array([self.class_weights[0] / s_total, self.class_weights[2] / s_total])

                for i in range(len(batch_x)):
                    # For P head - apply normalized weights
                    p_classes = np.argmax(p_label[i], axis=-1)
                    sw_p[i] = p_weights[p_classes]

                    # For S head - apply normalized weights
                    s_classes = np.argmax(s_label[i], axis=-1)
                    sw_s[i] = s_weights[s_classes]

            # Apply distance weighting if enabled
            sw_p *= dist_w[:, None]
            sw_s *= dist_w[:, None]

            self.__class__._has_printed_debug = True
            return batch_x, (p_label, s_label), (sw_p, sw_s)

        # 3) Single-head phasenet with or without beams
        if isinstance(batch_y, dict) and 'labels' in batch_y and 'beams' in batch_y and 'ADD' in str(self.beamlabel):
            lab = np.asarray(batch_y['labels'], dtype=np.float32)
            bms = np.asarray(batch_y['beams'], dtype=np.float32)
            if not self._has_printed_debug:
                print(f"[DEBUG] single-head w/ beams => lab={lab.shape}, beams={bms.shape}")
            y = np.concatenate([lab, bms], axis=-1)
        else:
            y = np.asarray(batch_y, dtype=np.float32)
            if not self._has_printed_debug:
                print(f"[DEBUG] single-head => y.shape={y.shape}")

            # Attempt transpose fix if user data is 4D (channels, batch, time, 1)
            if y.ndim == 4 and y.shape[1] == len(batch_x) and y.shape[0] != len(batch_x):
                if not self._has_printed_debug:
                    print(f"[DEBUG] Reordering single-head y from {y.shape} to fix transposition")
                y = np.transpose(y, (1, 2, 0, 3))  # => (batch, time, channels, 1)
                y = np.squeeze(y, axis=-1)  # => (batch, time, channels)
                if not self._has_printed_debug:
                    print(f"[DEBUG] y after transpose => shape={y.shape}")

        sw = np.ones((len(batch_x), y.shape[1]), dtype=np.float32) * dist_w[:, None]

        if self.class_weights and y.shape[-1] >= 3:
            cw_array = np.array(self.class_weights, dtype=np.float32)
            if not self._has_printed_debug:
                print(f"[DEBUG] applying class_weights => {cw_array} to first 3 channels")
            for i in range(len(batch_x)):
                cidx = np.argmax(y[i, :, :3], axis=-1)
                for t, c in enumerate(cidx):
                    if c < len(cw_array):
                        sw[i, t] *= cw_array[c]

        self.__class__._has_printed_debug = True
        return batch_x, y, sw


    def _getitem_default(self, idx):
        
        data = self.super_sequence.__getitem__(idx)
        batch_x, batch_y = data
            
        distance_weight = np.ones((len(batch_x), 1))

        if self.beamlabel and 'ADD' not in self.beamlabel :
            y = np.concatenate([batch_y[0],batch_y[1]], axis=-1)
            return batch_x, y, distance_weight

        p = np.concatenate([batch_y[i] for i in self.p_classes], axis=-1)
        s = np.concatenate([batch_y[i] for i in self.s_classes], axis=-1)

        p = np.max(p, axis=-1, keepdims=True)
        s = np.max(s, axis=-1, keepdims=True)
        p = np.clip(p, 0, 1)
        s = np.clip(s, 0, 1)
        n = np.clip(1 - p - s, 0, 1)
        if self.beamlabel and 'ADD' in self.beamlabel :
            y={'labels': np.concatenate([n, p, s], axis=-1),
               'beams': np.concatenate([batch_y[-2], batch_y[-1]], axis=-1)}
        #elif self.modeltype == 'splitoutputtransphasenet'  :
        elif self.modeltype.startswith('splitoutput'):
            noise_p = np.clip(1 - p, 0, 1)
            noise_s = np.clip(1 - s, 0, 1)
            y = np.concatenate([noise_p, p, noise_s, s], axis=-1)
        else : y = np.concatenate([n, p, s], axis=-1)
        sw = create_class_weights(self.class_weights, y) if not self.class_weights is None else np.ones((len(s),1))
        sw = distance_weight * sw
        
        return batch_x, y, sw

    def on_epoch_end(self):
        self.super_sequence.on_epoch_end()

def angle_diff_tf(true,pred,sample_weight=None):
    true = tf.math.angle(tf.complex(true[:, 0], true[:, 1]))
    pred = tf.math.angle(tf.complex(pred[:, 0], pred[:, 1]))

    diff = tf.math.atan2(tf.math.sin(true-pred), tf.math.cos(true-pred))
    if sample_weight:
        return sample_weight * diff
    return diff

class AngleMAE(tf.keras.metrics.Metric):
    def __init__(self, name='angle_mse', **kwargs):
        super(AngleMAE, self).__init__(name=name, **kwargs)
        self.angle_mae = self.add_weight(name='amae', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):

        values = tf.math.abs(angle_diff_tf(y_true,y_pred))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)
        self.angle_mae.assign_add(tf.reduce_mean(values))
        self.count.assign_add(1)

    def result(self):
        return self.angle_mae / self.count
    

class IOU(tf.keras.metrics.Metric):
    def __init__(self, name='iou', threshold=0.5, **kwargs):
        super(IOU, self).__init__(name=name, **kwargs)
        self.iou = self.add_weight(name='iou', initializer='zeros')
        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = tf.math.ceil(y_true - self.threshold)
        y_pred = tf.math.ceil(y_pred - self.threshold)
        res = tf.math.reduce_sum(tf.clip_by_value(y_true * y_pred, 0.0, 1.0), axis=(1,2))
        tot = tf.math.reduce_sum(tf.clip_by_value(y_true + y_pred, 0.0, 1.0), axis=(1,2))
        values = res/(tot + tf.keras.backend.epsilon())
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)
            
        self.iou.assign_add(tf.reduce_mean(values))

    def result(self):
        return self.iou
    

class TruePositives(tf.keras.metrics.Metric):
    def __init__(self, 
                 name='tp',
                 dt=1,
                 **kwargs):
        super(TruePositives, self).__init__(name=name, **kwargs)
        self.dt = dt
        self.reset_state()
        
    def reset_state(self):
        self.data = []
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        
        n_dims = y_pred.shape[-1]
        
        if n_dims > 1:
            y_true = y_true[...,-2:]
            y_pred = y_pred[...,-2:]

        it = tf.math.argmax(y_true, axis=1)
        ip = tf.math.argmax(y_pred, axis=1)
            
        values = tf.where(abs(it-ip) < self.dt, tf.ones_like(it), tf.zeros_like(it))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)
            
        self.data.append(tf.reduce_mean(values))
            
    def result(self):
        return tf.reduce_mean(self.data)
    
def recall_metric(dt=1.0):
    def recall(y_true, y_pred, sample_weight=None):
        n_dims = y_pred.shape[-1]
        
        if n_dims > 1:
            y_true = y_true[...,1:]
            y_pred = y_pred[...,1:]

        it = tf.math.argmax(y_true, axis=1)
        ip = tf.math.argmax(y_pred, axis=1)
            
        values = tf.where(abs(it-ip) < dt, tf.ones_like(it), tf.zeros_like(it))
        values = tf.cast(values, tf.float32)
        #if sample_weight is not None:
        #    sample_weight = tf.cast(sample_weight, tf.float32)
        #    sample_weight = tf.broadcast_to(sample_weight, values.shape)
        #    values = tf.multiply(values, sample_weight)
        return tf.reduce_mean(values)
    return recall

def kl_divergence_metric():
    def kld(y_true, y_pred, sample_weight=None):
        _, p, s = tf.split(y_true, 3, axis=-1)
        _, pt, st = tf.split(y_pred, 3, axis=-1)
        p = tf.squeeze(p)
        s = tf.squeeze(s)
        pt = tf.squeeze(pt)
        st = tf.squeeze(st)
        return (tf.keras.metrics.kl_divergence(p, pt) + tf.keras.metrics.kl_divergence(s, st)) / 2
    return kld

def js_divergence(p, q):
    pm = tf.reduce_sum(p, axis=-1, keepdims=True)
    pw = tf.where(pm < tf.keras.backend.epsilon(), tf.ones_like(pm), pm)
    qm = tf.reduce_sum(q, axis=-1, keepdims=True)
    qw = tf.where(qm < tf.keras.backend.epsilon(), tf.ones_like(qm), qm)
    p /= pm
    q /= qm
    m = (p + q) / 2
    return (tf.keras.metrics.kl_divergence(p, m) + tf.keras.metrics.kl_divergence(q, m) / 2)

def js_divergence_metric():
    def jsd(y_true, y_pred, sample_weight=None):
        _, p, s = tf.split(y_true, 3, axis=-1)
        _, pt, st = tf.split(y_pred, 3, axis=-1)
        p = tf.squeeze(p)
        s = tf.squeeze(s)
        pt = tf.squeeze(pt)
        st = tf.squeeze(st)
        return (js_divergence(p,pt) + js_divergence(s,st)) / 2
    return jsd

class CustomStopper(tf.keras.callbacks.EarlyStopping):
    def __init__(self, 
                 monitor='val_loss',
                 min_delta=0, 
                 patience=0, 
                 verbose=0, 
                 mode='auto',
                 restore_best_weights=False, 
                 start_epoch=1): # add argument for starting epoch
        super(CustomStopper, self).__init__(monitor=monitor, 
                                            min_delta=min_delta, 
                                            patience=patience, 
                                            verbose=verbose, 
                                            restore_best_weights=restore_best_weights,
                                            mode=mode)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)

class GradientNormLogger(tf.keras.callbacks.Callback):
    """Log the global gradient norm and gradient statistics to console and WandB."""

    def __init__(self, data, log_frequency=1, log_wandb=True):
        super().__init__()
        self.data = data
        self.log_frequency = log_frequency
        self.log_wandb = log_wandb

    def on_epoch_end(self, epoch, logs=None):

        def to_tensor_tree(struct):
            """Recursively convert anything that isn't already a Tensor."""
            return tf.nest.map_structure(
                lambda t: tf.convert_to_tensor(t) if not tf.is_tensor(t) else t,
                struct,
            )

        if (epoch + 1) % self.log_frequency != 0:
            return

        # Grab a single batch from the dataset or Sequence
        try:
            batch = self.data[0]
        except Exception:
            batch = next(iter(self.data))
        x, y = batch[0], batch[1]
        x = to_tensor_tree(x)
        y = to_tensor_tree(y)

        sample_weight = batch[2] if len(batch) > 2 else None
        if sample_weight is not None:
            sample_weight = to_tensor_tree(sample_weight)

        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.model.compiled_loss(
                y, y_pred, sample_weight=sample_weight, regularization_losses=self.model.losses
            )

        grads = tape.gradient(loss, self.model.trainable_variables)
        valid_grads = [g for g in grads if g is not None]

        if not valid_grads:
            tf.print("Warning: No valid gradients found at epoch", epoch + 1)
            return

        # Calculate gradient statistics
        global_norm = tf.linalg.global_norm(valid_grads)
        grad_norms = [tf.linalg.global_norm([g]) for g in valid_grads]

        # Calculate statistics
        max_grad_norm = tf.reduce_max(grad_norms)
        min_grad_norm = tf.reduce_min(grad_norms)
        mean_grad_norm = tf.reduce_mean(grad_norms)

        # Count NaN and Inf gradients
        nan_count = 0
        inf_count = 0
        for g in valid_grads:
            nan_count += tf.reduce_sum(tf.cast(tf.math.is_nan(g), tf.int32))
            inf_count += tf.reduce_sum(tf.cast(tf.math.is_inf(g), tf.int32))

            
            
class CategoricalFocalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, 
                 alpha=0.25, 
                 gamma=2.0, 
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='categorical_focal_crossentropy'):
        super(CategoricalFocalCrossentropy, self).__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        #y_pred = y_pred + epsilon
        # Clip the prediction value
        y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
        # Calculate cross entropy
        cross_entropy = -y_true*K.log(y_pred)
        # Calculate weight that consists of  modulating factor and weighting factor
        weight = self.alpha * y_true * K.pow((1-y_pred), self.gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.sum(loss, axis=-1)
        return loss
        
        
def create_data_generator(
    files: List[Tuple[str, str]], 
    config: Any, 
    training: bool = True, 
    validation: bool = False
) -> Tuple[Any, int]:
    include_noise = False
    if config.data.noise_waveforms :
        include_noise = True
    phaselist = config.data.allowed_phases 
    p_classes = [i for i, e in enumerate(phaselist) if 'P' in e] #ALL SUBCLASSES OF P
    s_classes = [i for i, e in enumerate(phaselist) if 'S' in e] #ALL SUBCLASSES OF S
    if config.data.holdout :
        holdout = []
        if config.run.gpu : infile = 'tf/'+config.data.holdout.split('/')[-1]
        else : infile = config.data.holdout
        with open(infile,'r') as fin:
            for line in fin:
                line=line.strip().split()
                holdout.append(line[0])
    else : holdout = False

    if config.data.extract_array_channels :
        ind_keep = []

        if config.run.gpu:
            inputdir = 'data'
        else:
            inputdir = config.data.inputdir
        with open(f'{inputdir}/{config.data.input_dataset_name}_{config.data.test_years[0]}_{config.data.input_datatype}_channels.json', 'r') as f:
            output_channel_order = json.load(f)
            for i,channel in enumerate(output_channel_order) :
                target_n_channels = 0
                for stat in config.data.extract_array_channels :
                    if stat[1] == '1c' :
                        comp = ['Z']
                        target_n_channels += 1
                    else :
                        comp = ['Z','N','E']
                        target_n_channels += 3
                    for c in comp :
                        chan = f'{stat[0]}.{c}'
                        if channel == chan :
                            ind_keep.append(i)
        if len(ind_keep) != target_n_channels :
            print('One of the wished components was not found!')
            exit()
    else : ind_keep = False

    # Get smart_crop from config, defaulting to False if not specified
    smart_crop = getattr(config.augment, 'smart_crop', False)
    
    # Get overlapping windows parameters from config
    use_overlapping_windows = getattr(config.augment, 'use_overlapping_windows', False)
    use_overlapping_windows_validation = getattr(config.augment, 'use_overlapping_windows_validation', False)
    overlap_seconds = getattr(config.augment, 'overlap_seconds', 150.0)
    
    # Determine whether to use overlapping windows based on training/validation mode
    if validation and use_overlapping_windows_validation:
        # Use overlapping windows for validation only
        use_ovl_windows = True
    elif training and not validation and use_overlapping_windows:
        # Use overlapping windows for training only
        use_ovl_windows = True
    else:
        use_ovl_windows = False
        
    dataset = EQDatareader(files=files, 
        new_length=int(config.data.sampling_rate*config.augment.new_size),
        batch_size=config.training.batch_size,
        taper_alpha=config.augment.taper,
        add_noise=config.augment.add_noise,
        add_event=config.augment.add_event,
        drop_channel=config.augment.drop_channel,
        add_gap=config.augment.add_gap,
        max_gap_size=config.augment.max_gap_size,
        augment=training,  # Keep original behavior: augmentation for both training and validation
        norm_mode=config.normalization.mode,
        norm_channel_mode=config.normalization.channel_mode,
        shuffle=training,  # Keep original behavior: shuffle for both training and validation
        testing=not training,
        include_noise=include_noise,
        ramp=config.augment.ramp,
        beamlabel=config.training.use_beam_as_label,
        holdout=holdout,
        modeltype=config.model.type,  # Pass modeltype from config
        smart_crop=smart_crop,  # Pass smart_crop from config
        use_overlapping_windows=use_ovl_windows,  # Pass overlapping windows from config
        overlap_seconds=overlap_seconds,  # Pass overlap in seconds from config
        sampling_rate=config.data.sampling_rate,  # Pass sampling rate from config
        validation_mode=validation,  # Pass validation mode flag
        extract_array_channels=ind_keep)
    nchannels=dataset.data[0]['x'].shape[1]

    return DropDetection(dataset, 
                         p_classes=p_classes, 
                         s_classes=s_classes,
                         class_weights=config.training.class_weights,
                         beamlabel=config.training.use_beam_as_label,
                         modeltype=config.model.type),nchannels

def get_data_files(
    indir: str, 
    years: List[int], 
    config: Any
) -> Tuple[List[Tuple[str, str]], Optional[List[str]], Optional[List[str]]]:
    datatype = config.data.input_datatype
    dataset = config.data.input_dataset_name
    files = [indir + f'{dataset}_{y}_{datatype}.hdf5' for y in years]
    files = list(filter(os.path.exists, files))
    tmp = 'labels_phase_detection'
    files_labels = [indir + f'{dataset}_{y}_{tmp}.hdf5' for y in years]
    files_labels = list(filter(os.path.exists, files_labels))
    # if monthly output:
    if len(files_labels) == 0 :
        files_labels = [indir + f'{dataset}_{y[:4]}_{tmp}.hdf5' for y in years]
        files_labels = list(filter(os.path.exists, files_labels))

    if config.data.noise_waveforms  :
        tmp = datatype+'_noise'
        files_noise = [indir + f'{dataset}_{y}_{tmp}.hdf5' for y in years]
        files_noise = list(filter(os.path.exists, files_noise))
    else : files_noise = None

    if len(files) != len(files_labels) :
        print("Different number of data and label files!")
        exit()
    return [files,files_labels,files_noise]


def correlation_coefficient(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den
    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return r

def correlation_coefficient_loss(y_true, y_pred):
    # this does not penalize negaitv correlation!
    #return 1 - K.square(correlation_coefficient(y_true, y_pred))
    # this should
    corr=correlation_coefficient(y_true, y_pred)
    return 1 - corr - tf.where(corr<0,-corr,0)
    
def get_model(config: Any, nchannels: int) -> Callable:

    # Enable or disable XLA JIT compilation depending on config
    jit_compile_flag = getattr(config.training, "jit_compile", True)
    
    model_type = config.model.type
    
    input_shape = (int(config.data.sampling_rate*config.augment.new_size), nchannels)
    
    filters = config.model.filters
    kernelsizes = config.model.kernel_sizes
    dropout = config.training.dropout
    pool_type = config.model.pooling_type
    try :
        activation = config.model.activation
    except ConfigAttributeError :
        activation = 'relu'

    if config.model.type.startswith('splitoutput'): 

        opt_class = type(tf.keras.optimizers.get(config.training.optimizer))

        # Set up gradient clipping options
        clipnorm  = getattr(config.training, "clipnorm", 5.0)
        # Stronger per-value clipping – use sane default (1.0) unless user specifies one
        clipvalue = getattr(config.training, "clipvalue", 1.0)

        # Build optimizer arguments
        opt_kwargs = {
            'learning_rate': config.training.learning_rate,
            'weight_decay':  config.training.weight_decay,
            # Always add per-value clipping; keeps individual weights/gradients bounded
            'clipvalue': clipvalue,
        }
        # Also keep global-norm clipping if user wants it (>0)
        if clipnorm is not None and clipnorm > 0:
            opt_kwargs['clipnorm'] = clipnorm
    
        opt = opt_class(**opt_kwargs)
        opt.iterations       # forcebuild


    else :
        opt = type(tf.keras.optimizers.get(config.training.optimizer))
        #opt = type(tf.keras.optimizers.Adam(clipvalue=1.0))  # Example for gradient clipping if infinite loss
        opt = opt(learning_rate=config.training.learning_rate, 
              weight_decay=config.training.weight_decay)
            
    if config.training.use_beam_as_label and 'ADD' not in config.training.use_beam_as_label  : num_classes = 2
    elif config.training.use_beam_as_label and 'ADD' in config.training.use_beam_as_label : num_classes = 5
    elif config.model.type == 'splitoutputtransphasenet' : num_classes = 4
    elif config.model.type == 'splitoutputbranch': num_classes = 4
    elif config.model.type == 'splitoutputtransphasenetdepthwise': num_classes = 4
    elif config.model.type == 'splitoutputbranchdepthwise': num_classes = 4
    else : num_classes = 3
    kw = dict(num_classes=num_classes,
                    dropout_rate=dropout,
                    filters=filters,
                    kernelsizes=kernelsizes,
                    pool_type=pool_type,
                    activation=activation, 
                    kernel_regularizer=tf.keras.regularizers.L1L2(config.training.l1_norm,
                                                                config.training.l2_norm),
                    output_activation='softmax')
        
    if model_type == 'phasenet':
        try :
            kw['conv_type'] = config.model.conv_type
        except ConfigAttributeError:
            kw['conv_type'] = 'default'

    model_class = nm.PhaseNet
        
    if model_type == 'transphasenet':
        kw['residual_attention'] = config.model.residual_attention
        kw['att_type'] = config.model.att_type
        kw['rnn_type'] = config.model.rnn_type
        kw['additive_att'] = config.model.additive_att
        try:
            kw['num_heads'] = config.model.num_heads
        except ConfigAttributeError:
            pass  # Use default from model
        model_class = nm.TransPhaseNet
        
    if model_type == 'splitoutputtransphasenet':
        # Remove num_classes from kw for SplitOutputTransPhaseNet
        if 'num_classes' in kw:
            del kw['num_classes']
        kw['residual_attention'] = config.model.residual_attention
        kw['att_type'] = config.model.att_type
        kw['rnn_type'] = config.model.rnn_type
        kw['additive_att'] = config.model.additive_att
        try:
            kw['num_heads'] = config.model.num_heads
        except ConfigAttributeError:
            pass  # Use default from model
        model_class = nm.SplitOutputTransPhaseNet
        
    if model_type == 'splitoutputtransphasenetdepthwise':
        # Remove num_classes from kw for DepthwiseSplitOutputTransPhaseNet
        if 'num_classes' in kw:
            del kw['num_classes']
        kw['num_channels'] = nchannels
        kw['residual_attention'] = config.model.residual_attention
        kw['att_type'] = config.model.att_type
        kw['rnn_type'] = config.model.rnn_type
        kw['additive_att'] = config.model.additive_att
        try:
            kw['num_heads'] = config.model.num_heads
        except ConfigAttributeError:
            pass  # Use default from model
        model_class = nm.DepthwiseSplitOutputTransPhaseNet
        
    if model_type == 'depthwisetransphasenet':
        kw['residual_attention'] = config.model.residual_attention
        kw['att_type'] = config.model.att_type
        kw['rnn_type'] = config.model.rnn_type
        kw['additive_att'] = config.model.additive_att
        kw['num_channels'] = nchannels
        try:
            kw['num_heads'] = config.model.num_heads
        except ConfigAttributeError:
            pass  # Use default from model
        model_class = nm.DepthwiseTransPhaseNet
        
    if model_type == 'epick':
        kw['residual_attention'] = config.model.residual_attention
        try:
            kw['num_heads'] = config.model.num_heads
        except ConfigAttributeError:
            pass  # Use default from model
        model_class = nm.EPick

    if model_type == 'splitoutputbranch':
        # Remove num_classes (handled internally by the model)
        if 'num_classes' in kw:
            del kw['num_classes']

        kw['residual_attention'] = config.model.residual_attention
        kw['att_type'] = config.model.att_type
        kw['rnn_type'] = config.model.rnn_type
        kw['additive_att'] = config.model.additive_att
        try:
            kw['num_heads'] = config.model.num_heads
        except ConfigAttributeError:
            pass  # Use default from model

        # Forward branch_at from config, default to len(filters)-1 if absent
        kw['branch_at'] = getattr(config.model, 'branch_at', len(config.model.filters) - 1)

        model_class = nm.SplitOutputTransPhaseNetBranch

    if model_type == 'splitoutputbranchdepthwise':
        # Remove num_classes (handled internally by the model)
        if 'num_classes' in kw:
            del kw['num_classes']

        kw['num_channels'] = nchannels
        kw['residual_attention'] = config.model.residual_attention
        kw['att_type'] = config.model.att_type
        kw['rnn_type'] = config.model.rnn_type
        kw['additive_att'] = config.model.additive_att
        try:
            kw['num_heads'] = config.model.num_heads
        except ConfigAttributeError:
            pass  # Use default from model

        # Forward branch_at from config, default to len(filters)-1 if absent
        kw['branch_at'] = getattr(config.model, 'branch_at', len(config.model.filters) - 1)

        model_class = nm.SplitOutputBranchDepthwise

    def create_model():
        model = model_class(**kw)
        if config.training.use_beam_as_label :
            #loss2 = tf.keras.losses.MeanSquaredError() # also check MAE, Huber,
            loss2 = correlation_coefficient_loss
            loss_weights = None
            #metrics2 = tf.keras.metrics.MeanSquaredError()
            metrics2 = correlation_coefficient
            if 'ADD' not in config.training.use_beam_as_label :
                loss = loss2
                metrics = metrics2

        if not config.training.use_beam_as_label or 'ADD' in config.training.use_beam_as_label:
            if config.model.type.startswith('splitoutput'):
                # Get losses and weights from config
                loss, loss_weights = get_splitoutput_losses_and_weights(config)

                # For the split-output network each branch (P, S) outputs a
                # single probability value per time-sample (shape: [B,T,1]).
                # Therefore we can feed the tensors directly to `keras_f1`
                # without any channel slicing.
                #def f1_scalar(yt, yp):
                #    return keras_f1(tf.squeeze(yt, axis=-1), tf.squeeze(yp, axis=-1))
                # But still need to the queezing!

               # def f1_scalar(yt, yp):
               #     yt = tf.cast(yt, tf.float32)
               #     yp = tf.cast(yp, tf.float32)
               #     # Drop the singleton channel without using squeeze (safer for unknown dims)
               #     if yt.shape.rank == 3:
               #         yt = yt[..., 0]   # (B, T, 1) -> (B, T)
               #         yp = yp[..., 0]
               #     # Ensure rank-2 shape even if static dims are None
               #     yt = tf.reshape(yt, [tf.shape(yt)[0], -1])   # (B, T)
               #     yp = tf.reshape(yp, [tf.shape(yp)[0], -1])
               #     return keras_f1(yt, yp)

                metrics = [
                    [
                        tf.keras.metrics.BinaryAccuracy(name='p_acc'),
                        #tf.keras.metrics.MeanMetricWrapper(f1_scalar, name='p_f1'),
                        tf.keras.metrics.MeanMetricWrapper(
                        #lambda a, b: f1_scalar(a, b), name='f1_p'),
                        lambda a, b: keras_f1(a[..., 0], b[..., 0]), name='f1_p'),
                    ],
                    [
                        tf.keras.metrics.BinaryAccuracy(name='s_acc'),
                        #tf.keras.metrics.MeanMetricWrapper(f1_scalar, name='s_f1'),
                        tf.keras.metrics.MeanMetricWrapper(
                        #lambda a, b: f1_scalar(a, b), name='f1_s'),
                        lambda a, b: keras_f1(a[..., 0], b[..., 0]), name='f1_s'),
                    ],
                ]

                # Compile with two heads
                model.compile(
                    optimizer=opt,
                    loss=loss,
                    loss_weights=loss_weights,
                    metrics=metrics,
                    jit_compile=jit_compile_flag,
                )
                return model
            else:
                loss = tf.keras.losses.CategoricalCrossentropy()
                loss_weights = None
                metrics = [
                    tf.keras.metrics.MeanMetricWrapper(
                        lambda a, b: keras_f1(a[..., 1], b[..., 1]), name='f1_p'
                    ),
                    tf.keras.metrics.MeanMetricWrapper(
                        lambda a, b: keras_f1(a[..., 2], b[..., 2]), name='f1_s'
                    ),
                ]

        if config.training.use_beam_as_label and 'ADD' in config.training.use_beam_as_label :
            loss={
                  'labels': loss,
                  'beams': loss2 
                 }
            loss_weights={
                  'labels': 0.5,
                  'beams': 0.5
                 }
            metrics={
                  'labels': metrics,
                  'beams': metrics2
                 }

    
        if not config.model.type.startswith('splitoutput'):
            model.compile(optimizer=opt,
                loss=loss,
                    loss_weights=loss_weights,
                    sample_weight_mode="temporal",
                    metrics=metrics)
        else :
            model.compile(
            optimizer=opt,
            loss=loss,
            loss_weights=loss_weights,
            metrics=metrics,
            jit_compile=jit_compile_flag,
        )
        return model
        
    return create_model()


def get_predictions(
    config: Any, 
    test_dataset: Any, 
    model: tf.keras.Model
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        
    #PREDICT
    true = []
    pred = []
    
    sample_weight = []
    xte = []
    
    for x, y, sw in tqdm(test_dataset):
        xte.append(x)
        if config.training.use_beam_as_label and 'ADD' in config.training.use_beam_as_label:
            m = np.zeros((y['labels'].shape[0],))
        elif config.model.type.startswith('splitoutput'):
            # For split output, y is already [p_label, s_label]
            m = np.zeros((y[0].shape[0],))
        else:
            m = np.zeros((y.shape[0],))

        a = model.predict_on_batch(x)
        if config.model.type.startswith('splitoutput'):
            p_out, s_out = a
            p = np.concatenate([np.zeros_like(p_out),p_out,np.zeros_like(s_out), s_out], axis=-1)
            y = np.concatenate(y, axis=-1)
        else:
            p = a

        sample_weight.append(np.asarray(sw))
        pred.append(p)
        true.append(y)

    xte, true, pred, sample_weight = map(lambda x: np.concatenate(x, axis=0), 
                                                                  [xte, true, pred, sample_weight])

    return xte, true, pred, sample_weight 

def get_splitoutput_losses_and_weights(config: Any) -> Tuple[List, List[float]]:
    """
    Get losses and weights for splitoutputtransphasenet model.
    Uses config to control weights between P and S heads.

    Args:
        config: Configuration object containing training parameters

    Returns:
        tuple: (loss_list, loss_weights) where:
            - loss_list = [loss_for_head_p, loss_for_head_s]
            - loss_weights = [weight_p, weight_s] from config
    """
    # Create base losses for P and S heads
    base_bce_p = tf.keras.losses.BinaryCrossentropy()
    base_bce_s = tf.keras.losses.BinaryCrossentropy()

    # Wrap with NaN-safe detection
    bce_p = NaNSafeLoss(base_bce_p, name='p_loss_safe')
    bce_s = NaNSafeLoss(base_bce_s, name='s_loss_safe')

    # Get weights from config - default to equal weighting if not specified
    try:
        # For split output, we use:
        # class_weights[0] = noise weight
        # class_weights[1] = P weight
        # class_weights[2] = S weight
        if len(config.training.class_weights) >= 3:
            # Normalize weights to sum to 1 for each head
            p_total = config.training.class_weights[0] + config.training.class_weights[1]
            s_total = config.training.class_weights[0] + config.training.class_weights[2]

            # Equal weighting between heads
            head_weights = [0.5, 0.5]
        else:
            head_weights = [0.5, 0.5]  # Default equal weighting between heads
    except:
        head_weights = [0.5, 0.5]  # Default equal weighting between heads

    return [bce_p, bce_s], head_weights

class NanGuard(tf.keras.callbacks.Callback):
    def __init__(self, watch_grads=False):
        super().__init__()
        self.watch_grads = watch_grads

    def on_train_batch_end(self, batch, logs=None):
        for w in self.model.trainable_weights:
            if tf.reduce_any(tf.math.logical_or(tf.math.is_nan(w), tf.math.is_inf(w))):
                print(f'\nNaN/Inf detected in {w.name}')
                self.model.stop_training = True
                return
        if self.watch_grads:
            # Gradient checking can be added here if desired, e.g. by inspecting
            # `self.model.optimizer.get_gradients(...)`.
            pass


class NaNSafeLoss(tf.keras.losses.Loss):
    """
    Wrapper loss that detects NaN/Inf values and provides debugging information.
    """
    def __init__(self, base_loss, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.base_loss = base_loss

    def call(self, y_true, y_pred):
        # Calculate base loss
        loss = self.base_loss(y_true, y_pred)

        # Identify NaN / Inf elements
        loss_is_nan = tf.math.is_nan(loss)
        loss_is_inf = tf.math.is_inf(loss)

        # Replace bad values with a large finite constant – this ops runs fine in graph mode
        loss_safe = tf.where(loss_is_nan | loss_is_inf, tf.constant(1e6, dtype=loss.dtype), loss)

        # Debug print removed for XLA compatibility

        return loss_safe

def plot_random_samples(
    model: tf.keras.Model,
    data_generator: Any,
    num_samples: int = 5,
    random_seed: int = 42,
    sampling_rate: float = 40.0,
    model_name: Optional[str] = None,
    save_path: Optional[str] = None,
    log_wandb: bool = True,
) -> None:
    """
    Plot waveforms, labels, and predictions for a few samples from a data generator.

    Parameters
    ----------
    model : tf.keras.Model
        Trained model (single or split-decoder).
    data_generator : tf.keras.utils.Sequence
        A generator that yields (x, y, sample_weight) or (x, [y1, y2], [w1, w2]) for
        split-output. Typically a DropDetection instance.
    num_samples : int
        Number of random samples to plot.
    random_seed : int
        Seed for reproducible random sampling.
    sampling_rate : float
        Sampling rate (Hz) to convert sample index to seconds.
    model_name : str, optional
        Name of the model for plot title/saving. If None, tries to infer from model.
    save_path : str, optional
        If provided, saves the plot to this path. If None, uses 'random_samples_{model_name}.png'
    log_wandb : bool
        Whether to log the plot to wandb if wandb is being used.
    """
    np.random.seed(random_seed)

    # Try to infer model name if not provided
    if model_name is None:
        try:
            model_name = model.name
        except:
            model_name = "unknown_model"

    # CRITICAL FIX: Set the underlying EQDatareader to testing mode for center cropping
    # This prevents random cropping from losing picks during plotting
    original_testing_state = None
    if hasattr(data_generator, 'super_sequence') and hasattr(data_generator.super_sequence, 'testing'):
        original_testing_state = data_generator.super_sequence.testing
        data_generator.super_sequence.testing = True
        print(f"[PLOT DEBUG] Temporarily enabled testing mode for center cropping")

    try:
        # 1) Grab one full batch from the generator (now with center cropping)
        batch_x, batch_y, *rest = data_generator[0]  # or any valid index

        # 2) Run the model prediction on this batch
        y_pred = model.predict(batch_x, verbose=0)

        # 3) Pick indices to plot, preferring samples with picks
        batch_size = len(batch_x)
        if batch_size < num_samples:
            print(f"Batch size={batch_size} < num_samples={num_samples}, reducing num_samples.")
            num_samples = batch_size

        # Find samples with picks for better visualization
        samples_with_picks = []
        samples_without_picks = []
        for i in range(batch_size):
            has_picks = False
            if isinstance(batch_y, (list, tuple)) and len(batch_y) == 2:
                # Split-output format - now using single-channel format (0 or 1)
                # batch_y[0] and batch_y[1] have shape (batch, time, 1) with binary values
                p_picks = np.sum(batch_y[0][i, :, 0] > 0.1)  # Changed from index 1 to 0
                s_picks = np.sum(batch_y[1][i, :, 0] > 0.1)  # Changed from index 1 to 0
                has_picks = (p_picks > 0 or s_picks > 0)
            else:
                # Standard format
                picks = np.sum(batch_y[i, :, 1:] > 0.1)
                has_picks = (picks > 0)

            if has_picks:
                samples_with_picks.append(i)
            else:
                samples_without_picks.append(i)

        # Prefer samples with picks, but include some without picks if needed
        if len(samples_with_picks) >= num_samples:
            idxs = np.random.choice(samples_with_picks, size=num_samples, replace=False)
            print(f"[PLOT DEBUG] Selected {num_samples} samples all with picks")
        elif len(samples_with_picks) > 0:
            # Mix samples with and without picks
            picks_to_select = len(samples_with_picks)
            no_picks_to_select = num_samples - picks_to_select

            idxs = np.concatenate([
                samples_with_picks,
                np.random.choice(samples_without_picks, size=no_picks_to_select, replace=False)
            ])
            np.random.shuffle(idxs)
            print(f"[PLOT DEBUG] Selected {picks_to_select} samples with picks, {no_picks_to_select} without")
        else:
            # No samples with picks found, use random selection
            idxs = np.random.choice(batch_size, size=num_samples, replace=False)
            print(f"[PLOT DEBUG] No samples with picks found, using random selection")

        def plot_single_sample(i, ax_wave, ax_label, ax_pred):
            """Helper to plot a single sample's waveforms and predictions."""
            # Plot waveforms
            waveforms = batch_x[i]
            num_channels = waveforms.shape[-1]
            timesteps = np.arange(len(waveforms)) / sampling_rate

            # Plot waveforms with offsets
            offset = 0
            for ch in range(num_channels):
                ax_wave.plot(timesteps, waveforms[:, ch] + offset, label=f'Chan {ch}')
                offset += np.max(np.abs(waveforms[:, ch])) * 1.2
            ax_wave.set_title(f"Sample {i}: Input waveforms")
            ax_wave.set_xlabel("Time (s)")
            ax_wave.legend(loc='upper right', fontsize='small')

            # Handle different output types
            if isinstance(batch_y, (list, tuple)) and len(batch_y) == 2:
                # Split-output model - now single-channel format
                p_true = batch_y[0][i]  # shape (time, 1) - binary P probability
                s_true = batch_y[1][i]  # shape (time, 1) - binary S probability
                ax_label.plot(timesteps, p_true[:, 0], 'r-', label='True P')
                ax_label.plot(timesteps, s_true[:, 0], 'b-', label='True S')
                # Plot noise as inverse (1 - picks)
                ax_label.plot(timesteps, 1.0 - p_true[:, 0], 'r--', alpha=0.3, label='Noise for P')
                ax_label.plot(timesteps, 1.0 - s_true[:, 0], 'b--', alpha=0.3, label='Noise for S')
                ax_label.set_ylim([0, 1.1])
                ax_label.set_title("Ground Truth (Split-output)")
                ax_label.legend(loc='upper right', fontsize='small')

                # Handle predictions - y_pred is a list of [p_pred, s_pred]
                p_pred_i = y_pred[0][i]  # shape (time, 1)
                s_pred_i = y_pred[1][i]  # shape (time, 1)

                # Ensure we have the right shape
                if p_pred_i.ndim == 1:
                    p_pred_i = p_pred_i.reshape(-1, 1)
                if s_pred_i.ndim == 1:
                    s_pred_i = s_pred_i.reshape(-1, 1)

                # Plot predictions - for split output, we just plot the single channel
                ax_pred.plot(timesteps, p_pred_i[:, 0], 'r-', label='Pred P')
                ax_pred.plot(timesteps, s_pred_i[:, 0], 'b-', label='Pred S')
                ax_pred.set_ylim([0, 1.1])
                ax_pred.set_title("Prediction (Split-output)")
                ax_pred.legend(loc='upper right', fontsize='small')

            else:
                # Single-decoder output
                true_i = batch_y[i]  # shape (time, #labels)
                num_label_channels = true_i.shape[-1]
                label_colors = ['k', 'r', 'b', 'g', 'm']  # extend if needed

                for ch in range(num_label_channels):
                    label_name = 'Noise' if ch == 0 else 'P' if ch == 1 else 'S' if ch == 2 else f'ch{ch}'
                    ax_label.plot(
                        timesteps, true_i[:, ch], label=f'True {label_name}', color=label_colors[ch % len(label_colors)]
                    )
                ax_label.set_title(f"Ground Truth (Single-output)")
                ax_label.set_ylim([0, 1.1])
                ax_label.legend(loc='upper right', fontsize='small')

                # Handle predictions - check if y_pred is list or array
                if isinstance(y_pred, (list, tuple)):
                    # Handle list format (could be from split-output or single-output)
                    if len(y_pred) == 2:
                        # Split-output format but we're in single-output branch - convert to single output
                        p_pred = y_pred[0][i]  # shape (time, 1)
                        s_pred = y_pred[1][i]  # shape (time, 1)
                        # Create combined format: [noise, P, S] where noise = 1 - P - S
                        noise_pred = 1.0 - p_pred[:, 0] - s_pred[:, 0]
                        pred_i = np.stack([noise_pred, p_pred[:, 0], s_pred[:, 0]], axis=-1)
                    else:
                        # Single element list
                        pred_i = y_pred[0][i]
                else:
                    # Array format
                    pred_i = y_pred[i] if y_pred.ndim == 3 else y_pred[i][0]
                num_pred_channels = pred_i.shape[-1]
                for ch in range(num_pred_channels):
                    label_name = 'Noise' if ch == 0 else 'P' if ch == 1 else 'S' if ch == 2 else f'ch{ch}'
                    ax_pred.plot(
                        timesteps, pred_i[:, ch], label=f'Pred {label_name}', color=label_colors[ch % len(label_colors)]
                    )
                ax_pred.set_title(f"Prediction (Single-output)")
                ax_pred.set_ylim([0, 1.1])
                ax_pred.legend(loc='upper right', fontsize='small')

        # Create figure and plot
        fig, axes = plt.subplots(num_samples, 3, figsize=(16, 3 * num_samples))
        if num_samples == 1:
            axes = np.array([axes])

        for row_i, sample_idx in enumerate(idxs):
            plot_single_sample(sample_idx, axes[row_i, 0], axes[row_i, 1], axes[row_i, 2])

        plt.suptitle(f"Random Samples - {model_name}", y=1.02)
        plt.tight_layout()

        # Save plot if requested
        if save_path is None:
            save_path = f'random_samples_{model_name}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved plot to {save_path}")

        # Log to wandb if requested and available
        if log_wandb and 'wandb' in globals() and wandb.run is not None:
            wandb.log({"Random Samples": wandb.Image(fig), "random_samples_path": save_path})

        plt.close(fig)
        return save_path


    finally:
        # Restore original testing state
        if original_testing_state is not None and hasattr(data_generator, 'super_sequence'):
            data_generator.super_sequence.testing = original_testing_state
            print(f"[PLOT DEBUG] Restored original testing mode: {original_testing_state}")







