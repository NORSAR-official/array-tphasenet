"""
Training Script for Seismic Phase Detection Models
===================================================

This script orchestrates the complete training pipeline for deep learning models
designed for seismic phase picking and earthquake detection. It handles configuration
loading, data preparation, model creation, training with callbacks, and result saving.

Workflow
--------
1. **Configuration**: Load and validate training configuration from config.yaml
2. **Hardware Setup**: Configure GPU/CPU execution, mixed precision, and XLA compilation
3. **Data Loading**: Prepare training, validation, and test datasets with augmentation
4. **Model Creation**: Instantiate the specified model architecture
5. **Training**: Train with callbacks (early stopping, learning rate scheduling, etc.)
6. **Evaluation**: Generate predictions on test data
7. **Persistence**: Save trained model and prediction results

Configuration File
------------------
The script expects a `config.yaml` file in the current directory with the following
main sections:

- **run**: Execution settings (GPU, output paths)
- **data**: Dataset specifications (years, input directory, sampling rate)
- **training**: Hyperparameters (learning rate, batch size, epochs, callbacks)
- **augment**: Data augmentation parameters (noise, gaps, cropping)
- **normalization**: Waveform normalization settings
- **model**: Architecture type and parameters (filters, attention, etc.)
- **evaluation**: Metrics and thresholds

Example
-------
To run training::

    $ python train.py

The script will:
- Read config.yaml from the current directory
- Load data from paths specified in the config
- Train the model with GPU if available
- Save outputs to the configured output directory

Environment Variables
---------------------
- CUDA_VISIBLE_DEVICES: Control GPU visibility (set to '-1' for CPU-only)
- TF_CPP_MIN_LOG_LEVEL: TensorFlow logging verbosity (default: '2')

Output Files
------------
- saved_model_{setting}.tf: Trained model in TensorFlow SavedModel format
- predictions_{setting}.npz: Test set predictions and ground truth labels
- random_samples_{setting}.png: Visualization of sample predictions

Notes
-----
- Mixed precision (float16) is enabled by default for faster training on modern GPUs
- XLA JIT compilation can be toggled via config for performance optimization
- Early stopping with warmup prevents premature convergence
- Random seed can be fixed via config for reproducibility

See Also
--------
train_utils : Data loading, augmentation, and model factory functions
models : Neural network architectures for phase detection
evaluate_on_testdata : Evaluation metrics and performance analysis

Author: Erik Myklebust, Andreas Koehler, Tord Stangeland, Steffen Mæland
License: MIT

"""

from typing import Dict, List, Tuple, Any, Optional
import argparse
import numpy as np
import numpy.typing as npt
import os

# Disable XLA JIT globally – CuDNN RNN ops are unsupported under XLA on TF 2.16
# os.environ.setdefault('TF_XLA_FLAGS', '--tf_xla_auto_jit=0')
# os.environ.setdefault('TF_XLA_ENABLE_XLA_DEVICES', 'false')

# Limit XLA autotune workspace to avoid huge one-off allocations during the
# first compiled batch (TRT / Triton kernel search). 4 MiB is plenty.
# If the variable already exists we leave it untouched so that power-users can
# override it.

# maybe created issue for depth-wise models :
#os.environ.setdefault(
#    "XLA_FLAGS", "--xla_gpu_autotune_level=1"
#)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from omegaconf import OmegaConf
from setup_config import add_root_paths, get_config_dir, dict_to_namespace
from train_utils import CustomStopper, GradientNormLogger, NanGuard
from tensorflow.keras import mixed_precision
from train_utils import get_data_files, create_data_generator, get_model, get_predictions
from train_utils import plot_random_samples

# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND LINE ARGUMENTS
# ═══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(
    description='Train seismic phase detection models',
    formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    '-c', '--config',
    type=str,
    default='./config.yaml',
    help='Path to configuration file (default: ./config.yaml)'
)
cli_args = parser.parse_args()

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION LOADING
# ═══════════════════════════════════════════════════════════════════════════════
print(f'Reading config from {cli_args.config} ...')
args = OmegaConf.load(cli_args.config)
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
if not hasattr(cfg.run, "fix_seed"): 
    setattr(cfg.run, "fix_seed", False)

print('Config read.')

# ═══════════════════════════════════════════════════════════════════════════════
# HARDWARE CONFIGURATION (GPU/CPU)
# ═══════════════════════════════════════════════════════════════════════════════
if cfg.run.gpu:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
else:
    # Explicitly disable GPU when running in CPU-only mode
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    gpus = []

# Disable soft device placement - fail fast if GPU ops are not available
tf.config.set_soft_device_placement(False)

# Set random seed for reproducibility if specified
if cfg.run.fix_seed:
    tf.keras.utils.set_random_seed(1234)
    print("[INFO] Random seed fixed for reproducibility")

# Verify GPU availability when GPU mode is requested
if cfg.run.gpu:
    if not gpus:
        raise RuntimeError(
            "[ERROR] No GPU devices detected even though cfg.run.gpu=True. "
            "Verify that you are running on a GPU node and that "
            "CUDA_VISIBLE_DEVICES is set correctly."
        )

# ═══════════════════════════════════════════════════════════════════════════════
# PRECISION & COMPILATION SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════
# Mixed precision and XLA compilation can significantly improve training performance.
# These can be overridden in config.yaml under the `training` section.

use_mixed = getattr(cfg.training, "mixed_precision", True)
use_xla = getattr(cfg.training, "jit_compile", True)

# Configure mixed precision for faster training on modern GPUs
if use_mixed:
    mixed_precision.set_global_policy("mixed_float16")
    print("[INFO] Mixed precision enabled (float16 compute, float32 variables)")
else:
    mixed_precision.set_global_policy("float32")
    print("[INFO] Mixed precision disabled – using float32")

# Enable/disable XLA (Accelerated Linear Algebra) JIT compilation
tf.config.optimizer.set_jit(use_xla)
print(f"[INFO] XLA JIT {'enabled' if use_xla else 'disabled'}")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
inputdir = cfg.data.inputdir
outputdir = cfg.run.outputdir

# Extract year ranges for train/validation/test splits
training_years = cfg.data.train_years
testing_years = cfg.data.test_years
validation_years = cfg.data.valid_years

# Get HDF5 file paths for each dataset split
training_files = get_data_files(inputdir, training_years, cfg)
validation_files = get_data_files(inputdir, validation_years, cfg)
testing_files = get_data_files(inputdir, testing_years, cfg)

# Create data generators with appropriate augmentation settings
train_dataset, _ = create_data_generator(
    training_files, cfg, training=True, validation=False
)
valid_dataset, _ = create_data_generator(
    validation_files, cfg, training=True, validation=True
)
test_dataset, nchannels = create_data_generator(
    testing_files, cfg, training=False, validation=False
)

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT NAMING
# ═══════════════════════════════════════════════════════════════════════════════
# Generate a descriptive name for this training run based on configuration
model_type = cfg.model.type
dataset = cfg.data.input_dataset_name

# Build experiment name from configuration parameters
setting = f'{dataset}_{model_type}'

if cfg.data.holdout:
    setting = f'{dataset}_holdout_{model_type}'
    
if cfg.normalization.channel_mode == 'local':
    setting = f'{dataset}_localnorm_{model_type}'
    
if cfg.data.extract_array_channels:
    setting = f'array{cfg.data.setname}_{model_type}'
    
if cfg.run.custom_outname:
    setting = f'{dataset}_{cfg.run.custom_outname}_{model_type}'
    
if cfg.data.extract_array_channels and cfg.run.custom_outname:
    setting = f'array{cfg.data.setname}_{cfg.run.custom_outname}_{model_type}'

print(f"[INFO] Experiment name: {setting}")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL CREATION
# ═══════════════════════════════════════════════════════════════════════════════
# Define input shape: (time_samples, channels)
input_shape = (
    int(cfg.data.sampling_rate * cfg.augment.new_size), 
    nchannels
)

# Create model instance and build with specified input shape
model = get_model(cfg, nchannels)
model.build((cfg.training.batch_size, *input_shape))

# Verify GPU execution capability with a test operation
if cfg.run.gpu:
    try:
        with tf.device("GPU:0"):
            _ = tf.linalg.matmul(tf.ones((2, 2)), tf.ones((2, 2)))
        print("[INFO] Verified GPU execution with test operation")
    except (RuntimeError, tf.errors.InvalidArgumentError) as e:
        raise RuntimeError(
            "[ERROR] TensorFlow detected GPU but cannot execute operations on it. "
            "Verify NVIDIA runtime is enabled (--gpus all) and CUDA drivers are compatible."
        ) from e

print(f'[INFO] Model parameters: {model.num_parameters:,}')

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

monitor_metric = 'val_loss'

# Configure callbacks based on model type
if cfg.model.type.startswith('splitoutput'):
    # Split-output models use specialized callbacks for monitoring
    callbacks = [
        # Early stopping with warmup (no stopping before epoch 5)
        CustomStopper(
            monitor_metric,
            mode='min',
            patience=cfg.training.early_stopping_patience,
            start_epoch=5,
            min_delta=1e-4,
            restore_best_weights=False,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
        GradientNormLogger(
            train_dataset, 
            log_frequency=1, 
            log_wandb=hasattr(cfg, "wandb") and getattr(cfg.wandb, "use", False)
        ),
        NanGuard(),
    ]

    # Configure learning rate scheduler
    lr_scheduler = getattr(cfg.training, 'lr_scheduler', 'reduce_lr_on_plateau')

    if lr_scheduler == 'cosine_annealing':
        # Cosine annealing with warm restarts
        initial_learning_rate = cfg.training.learning_rate
        first_decay_steps = cfg.training.epochs // 4

        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_learning_rate,
            first_decay_steps=first_decay_steps,
            t_mul=2.0,
            m_mul=0.5,
            alpha=0.5e-6 / initial_learning_rate,
        )

        def cosine_sched(epoch: int, _lr: float) -> float:
            return float(lr_schedule(epoch))

        callbacks.append(tf.keras.callbacks.LearningRateScheduler(cosine_sched))
        print(f"[INFO] Using cosine annealing learning rate schedule")
    else:
        # Reduce learning rate on plateau (default)
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor_metric,
                factor=np.sqrt(0.1),
                min_lr=0.5e-6,
                mode='min',
                patience=cfg.training.reduce_lr_patience,
            )
        )
        print(f"[INFO] Using ReduceLROnPlateau learning rate schedule")
else:
    # Standard models use basic callbacks
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor_metric,
            factor=np.sqrt(0.1),
            min_lr=0.5e-6,
            mode='min',
            patience=cfg.training.reduce_lr_patience
        ),
        CustomStopper(
            monitor_metric,
            mode='min',
            patience=cfg.training.early_stopping_patience,
            start_epoch=5,
            min_delta=1e-4,
            restore_best_weights=False
        ),
        tf.keras.callbacks.TerminateOnNaN()
    ]

# ═══════════════════════════════════════════════════════════════════════════════
# PRE-TRAINING VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
# Generate sample predictions before training to verify model output format
try:
    plot_random_samples(
        model=model,
        data_generator=train_dataset,
        num_samples=5,
        sampling_rate=cfg.data.sampling_rate,
        model_name=setting,
        save_path=f'{outputdir}/random_samples_pretrain_{setting}.png',
        log_wandb=hasattr(cfg, "wandb") and getattr(cfg.wandb, "use", False),
    )
    print(f"[INFO] Pre-training visualization saved to {outputdir}/random_samples_pretrain_{setting}.png")
except tf.errors.InvalidArgumentError as e:
    print(f"[WARN] Skipping initial plot due to XLA/CuDNN incompatibility: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(valid_dataset)}")
print(f"Epochs: {cfg.training.epochs}")
print(f"Batch size: {cfg.training.batch_size}")
print("="*80 + "\n")

model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=cfg.training.epochs,
    callbacks=callbacks
)

# ═══════════════════════════════════════════════════════════════════════════════
# POST-TRAINING VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("GENERATING POST-TRAINING VISUALIZATIONS")
print("="*80)

plot_random_samples(
    model=model,
    data_generator=train_dataset,
    num_samples=5,
    sampling_rate=cfg.data.sampling_rate,
    model_name=setting,
    save_path=f'{outputdir}/random_samples_{setting}.png',
    log_wandb=hasattr(cfg, "wandb") and getattr(cfg.wandb, "use", False),
)
print(f"[INFO] Post-training visualization saved to {outputdir}/random_samples_{setting}.png")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL SAVING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("SAVING MODEL")
print("="*80)

model_path = f'{outputdir}/models/saved_model_{setting}.tf'
model.save(model_path, save_format="tf")
print(f"[INFO] Model saved to {model_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST SET EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("EVALUATING ON TEST SET")
print("="*80)

xte, true, pred, sample_weight = get_predictions(
    cfg, test_dataset, model
)

# Extract metadata for each test sample
ids = [a['event_id'] for a in test_dataset.super_sequence.data]
arids = [a['arrival_ids'] for a in test_dataset.super_sequence.data]
stations = [a['station'] for a in test_dataset.super_sequence.data]

# Save predictions and ground truth
predictions_path = f'{outputdir}/predictions/predictions_{setting}.npz'
np.savez(
    predictions_path,
    x=xte,
    y=true,
    yhat=pred,
    ids=np.array(ids),
    arids=arids,
    stations=stations
)
print(f"[INFO] Predictions saved to {predictions_path}")

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"Experiment: {setting}")
print(f"Model: {model_path}")
print(f"Predictions: {predictions_path}")
print("="*80 + "\n")

