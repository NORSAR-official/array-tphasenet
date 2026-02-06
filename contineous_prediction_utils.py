# Copyright 2026, Andreas Koehler, MIT license

"""
Continuous Prediction Utilities
================================

This module provides utilities for continuous seismic phase detection including
the LivePhaseNet class for live prediction from data streams, and functions for
loading, processing, and combining continuous predictions.

Classes
-------
LivePhaseNet : Live phase detection predictor class
Detection : Seismic detection representation

Functions
---------
phase_detection_station : Run phase detection for station(s)
phase_detection : Main continuous phase detection workflow
beam_phase_detection : Continuous beam-based phase detection for arrays
combine_output_from_sliding_windows : Merge overlapping window predictions
batch_feed_padding : Parallel processing helper for window combination
load_cont_beams : Load continuous beam predictions from miniseed
load_cont_detections : Load continuous station predictions from NPZ
load_manual_picks : Load manual picks from text file

Beam Detection Functions
------------------------
get_array_stations : Get station list for an array
load_array_waveforms : Load waveforms for array stations
preprocess_stream : Preprocess waveforms for beamforming
create_detection_beams : Create P and S detection beams
normalize_data : Normalize waveform data
"""

import os
import re
import glob
import json
import sys
import logging

# Suppress TensorFlow warnings (must be set before importing TensorFlow)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no INFO, 2=no WARNING, 3=no ERROR
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
from multiprocessing import Process, Manager
from scipy.signal.windows import tukey
from obspy import UTCDateTime, Stream, Trace, read
from obspy.core.event import ResourceIdentifier
import tensorflow as tf


class Detection:
    """
    Represents a seismic detection on a beam or station.
    
    Parameters
    ----------
    id_ : ResourceIdentifier
        Unique detection identifier
    time : UTCDateTime
        Detection time
    duration_sec : float
        Detection duration in seconds
    beam_id : str
        ID of beam/phase that was detected (e.g., 'P', 'S')
    snr : float
        Signal-to-noise ratio (or probability for ML detections)
    sta : float
        Short-term average value
    lta : float
        Long-term average value
    threshold : float, optional
        Detection threshold used
    num_beams_detecting : int, optional
        Number of beams contributing to this detection
    """
    
    def __init__(self, id_, time, duration_sec, beam_id, snr, sta, lta, 
                 threshold=None, num_beams_detecting=1):
        self.id = id_
        self.time = time
        self.duration_sec = duration_sec
        self.beam_id = beam_id
        self.snr = snr
        self.sta = sta
        self.lta = lta
        self.threshold = threshold
        self.num_beams_detecting = num_beams_detecting

    def __str__(self):
        if self.incomplete:
            return f"[{self.id}] {self.beam_id} - {self.time} + ->, SNR: {self.snr:.2f}"
        return f"[{self.id}] {self.beam_id} - {self.time} + {self.duration_sec:.2f}, SNR: {self.snr:.2f}"

    def __eq__(self, other):
        return (self.time == other.time and 
                self.duration_sec == other.duration_sec and
                self.beam_id == other.beam_id and 
                self.snr == other.snr)

    @property
    def incomplete(self):
        """Check if detection duration is not set."""
        return self.duration_sec is None

    @property
    def end_time(self):
        """Calculate detection end time."""
        if self.incomplete:
            return None
        return self.time + self.duration_sec

    @classmethod
    def create(cls, time, duration_sec, beam_id, snr, sta, lta, 
               threshold=None, num_beams_detecting=1):
        """Factory method to create a Detection with auto-generated ID."""
        return cls(ResourceIdentifier(), time, duration_sec, beam_id, 
                   snr, sta, lta, threshold, num_beams_detecting)
from tqdm import tqdm


def load_station_waveforms(client, station, channels, start, end, array_info=None, station_index=0):
    """
    Load waveforms for a single station with proper channel selection.
    
    Parameters
    ----------
    client : obspy.clients.fdsn.Client
        FDSN client for data retrieval
    station : str
        Station code
    channels : str
        Comma-separated channel codes (e.g., "BHZ,BHN,BHE")
    start : UTCDateTime
        Start time
    end : UTCDateTime
        End time
    array_info : list, optional
        Array configuration info, e.g., [('STA', '1c'), ...]. If station is '1c',
        only Z-component is loaded.
    station_index : int, optional
        Index of this station in array_info list (default 0)
    
    Returns
    -------
    Stream
        ObsPy Stream with loaded waveforms
    """
    # Determine channels based on array configuration
    if array_info and station_index < len(array_info):
        if array_info[station_index][1] == '1c':
            chan = ",".join([ch for ch in channels.split(",") if 'z' in ch.lower()])
        else:
            chan = channels
    else:
        chan = channels
    
    st = client.get_waveforms('*', station, '*', chan, start, end)
    
    # Handle duplicate channels (both BH and HH present)
    if len(st) > 3:
        st = st.select(channel='BH*')
        if len(st) > 3:
            st = st[:3]
    
    # For 1c stations, ensure only 1 trace
    if array_info and station_index < len(array_info):
        if array_info[station_index][1] == '1c' and len(st) > 1:
            st = st.select(channel='BH*')
            if len(st) > 1:
                st = st[:1]
    
    return st


def get_indices(proc_idx,iterations,processes,remainder):
    """
    Calculate start and end indices for a specific process.

    Distributes remainder items across first 'remainder' processes to balance load.

    Parameters
    ----------
    proc_idx : int
        Process index (0-based)
    iterations : int
        Base number of items per process
    processes : int
        Total number of processes
    remainder : int
        Extra items to distribute

    Returns
    -------
    index_start : int
        Starting index for this process
    index_end : int
        Ending index (exclusive) for this process
    """
    index_start = proc_idx * iterations + min(proc_idx, remainder)
    index_end = index_start + iterations + (1 if proc_idx < remainder else 0)
    return index_start,index_end



def split_processses(num_processes, num_items):
    """
    Calculate workload distribution for parallel processing.

    Parameters
    ----------
    num_processes : int
        Desired number of parallel processes
    num_items : int
        Total number of items to process

    Returns
    -------
    processes : int
        Actual number of processes to use (min of num_processes and num_items)
    threads : list
        List of None placeholders for thread objects
    iterations : int
        Base number of items per process
    remainder : int
        Extra items to distribute among first processes
    """
    processes = min(num_processes, num_items)
    threads = [None] * processes
    iterations = num_items // processes
    remainder = num_items % processes
    return processes,threads,iterations,remainder


def read_model(cfg_pred,cfg_model):
    """
    Load a trained TensorFlow/Keras model for prediction.

    Constructs model filename from configuration parameters and loads the model.

    Parameters
    ----------
    cfg_pred : namespace
        Prediction configuration (currently unused but kept for API consistency)
    cfg_model : namespace
        Model configuration with:
        - run.outputdir : Output directory path
        - data.input_dataset_name : Dataset identifier
        - data.extract_array_channels : Array channel extraction flag
        - data.setname : Array configuration name
        - model.type : Model architecture type
        - run.custom_outname : Optional custom output name

    Returns
    -------
    tensorflow.keras.Model
        Loaded Keras model ready for prediction

    Warns
    -----
    Prints warning if model and data sampling rates differ.
    """
    model_name = f'{cfg_model.run.outputdir}models/'
    if not hasattr(cfg_model.data, "extract_array_channels"): setattr(cfg_model.data, "extract_array_channels", False)
    if not hasattr(cfg_model.run, "custom_outname"): setattr(cfg_model.run, "custom_outname", False)

    if cfg_model.data.extract_array_channels :
        model_type = f'array{cfg_model.data.setname}_{cfg_model.model.type}'
    else :
        model_type = f'{cfg_model.data.input_dataset_name}_{cfg_model.model.type}'
    if cfg_model.run.custom_outname :
        model_type = f'{cfg_model.data.input_dataset_name}_{cfg_model.run.custom_outname}_{cfg_model.model.type}'
    if cfg_model.run.custom_outname and cfg_model.data.extract_array_channels :
        model_type = f'array{cfg_model.data.setname}_{cfg_model.run.custom_outname}_{cfg_model.model.type}'
    model_name += f'saved_model_{model_type}.tf'
    print("Predicting with model:",model_type)
    model = tf.keras.models.load_model(model_name, compile=False)

    return model



class LivePhaseNet:
    """
    Live phase detection from continuous seismic data streams.
    
    Handles data loading, preprocessing, normalization, and model prediction
    for continuous or windowed seismic phase detection. Supports both single
    station and array processing with parallel computation.
    
    Parameters
    ----------
    model : tensorflow.keras.Model
        Trained phase detection model
    client : obspy.Client or False, default=False
        Seismic data client for waveform retrieval
    station : list of station names or single array for array detections
    channels : str, default="HHZ,HHE,HHN,BHZ,BHE,BHN,sz,se,sn,bz,be,bn"
        Channel priority list (comma-separated)
    length : int, default=60
        Window length (seconds)
    step : int, default=10
        Step size for sliding windows (seconds)
    bandpass : tuple or None, default=None
        Frequency range (fmin, fmax) for bandpass filter
    delay : int, default=10
        Time delay from present (for live mode, seconds)
    return_raw : bool, default=False
        Return raw waveforms along with predictions
    verbose : bool, default=True
        Print progress information
    normalization : str, default='max'
        Normalization method: 'max' or 'std'
    normalization_mode : str, default='global'
        'global' or 'local' channel normalization
    stream : obspy.Stream or None, default=None
        Pre-loaded stream for faster processing
    taper : float, default=0.01
        Taper fraction for window edges
    sampling_rate : float, default=40
        Target sampling rate (Hz)
    array : list or False, default=False
        Array station configuration for array detection [[station, '1c'/'3c'], ...]
    num_processes : int, default=2
        Number of parallel processes
    
    Attributes
    ----------
    skipped_stations : list
        Indices of stations that failed processing
    batches_per_station : int
        Number of time windows processed per station
    
    Methods
    -------
    predict(start, end, step)
        Run prediction on specified time range
    
    Notes
    -----
    Either client or stream must be provided for data access.
    """
    def __init__(self,
                 model,
                 client=False,
                 station=['ARA0'],
                 channels="HHZ,HHE,HHN,BHZ,BHE,BHN,sz,se,sn,bz,be,bn",
                 length=60,
                 step=10,
                 bandpass=None,
                 delay=10,
                 return_raw=False,
                 verbose=True,
                 normalization='max',
                 normalization_mode='global',
                 stream=None, # to speed up we can pre-load stream and create windows by slicing
                 taper=0.01,
                 sampling_rate=40,
                 array=False,
                 num_processes=2):
        self.model = model
        if not client and stream is None :
            raise ValueError(
                f'Either client or stream has to be provided!')
            exit()
        self.client = client
        self.return_raw = return_raw
        self.station = station
        self.channels = channels
        self.length = length
        self.delay = delay
        self.step = step
        self.bandpass = bandpass
        self.normalization = normalization
        self.normalization_mode = normalization_mode
        self.verbose = verbose
        self.sampling_rate = sampling_rate
        self.taper = taper
        self.array = array
        self.num_processes = num_processes
        if stream is not None : self.stream = stream
        else : self.stream = None
        self.w = tukey(int(length*sampling_rate), taper)[np.newaxis,:,np.newaxis]

    def _normalize(self, X, mode='max', channel_mode='local'):
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

    def _load_data(self, start, end,cut=True, station=False):

        if self.array : statlist = [sta[0] for sta in self.array]
        else : statlist = self.station
        # for parralel processing I have only a single station when load_data is called
        if station : statlist = [station]

        length = int(self.length * self.sampling_rate)
        if start is None or end is None:
            start, end = UTCDateTime.now() - self.length - self.delay, UTCDateTime.now() - self.delay

        if self.stream is None :
            st_org = Stream()
            for i, stat in enumerate(statlist):
                st = load_station_waveforms(
                    self.client, stat, self.channels, start, end,
                    array_info=self.array, station_index=i
                )
                st_org += st
        else :
            st_sliced = self.stream.slice(start, end).copy()
            st_org = Stream()
            for stat in statlist:
                st_org += st_sliced.select(station=stat)
        st_org.sort()
        if len(st_org) == 0 : return
        if not self.array and len(st_org) > 3 : st_org = st_org[:3]
        st_org.detrend()
        st_org.taper(self.taper)
        if self.bandpass is not None:
            st_org.filter('bandpass', freqmin=self.bandpass[0], freqmax=self.bandpass[1])
        for trace in st_org:
            sr = trace.stats.sampling_rate
            break
        if sr != self.sampling_rate:
            st_org.resample(self.sampling_rate, no_filter=True)
        if cut :
            # Ensure all traces have exactly 'length' samples (pad with zeros if needed)
            trace_arrays = []
            for trace in st_org:
                if len(trace.data) >= length:
                    trace_arrays.append(trace.data[:length])
                else:
                    # Pad short traces with zeros
                    padded = np.zeros(length)
                    padded[:len(trace.data)] = trace.data
                    trace_arrays.append(padded)
            data = np.stack(trace_arrays, axis=-1)
            assert len(data) == length
        else:
            # For saving waveforms - return list of trace data (may have different lengths)
            data = [trace.data for trace in st_org]
        return data

    def _batch_feed(self,x, index_start, index_end, start,end,step,proc_idx):
        """
        Run station detection in batches
        """
        delta = end - start
        for index in range(index_start, index_end):
            try:
                print(f"[PID {os.getpid()}] Processing index {index}")
                raw_data = [self._load_data(start + s, start + s + self.length,station=self.station[index]) for s in np.arange(0, delta, step)]
                x[index] = [self._normalize(r, self.normalization, self.normalization_mode) for r in raw_data]
                x[index] = np.stack(x[index], axis=0) * self.w
            except Exception as e:
                print(f"[PID {os.getpid()}] Error for station {self.station[index]}: {e} . Setting data to None.")
                x[index] = None

    def predict(self, start=None, end=None, step=None):
        
        #for s in np.arange(0, delta, step) :
        #    print(start + s, start + s + self.length)
        print('Loading and preparing data ...')
        #if isinstance(self.station, list)  :
        if not self.array :
            self.return_raw = False
            processes,threads,iterations,remainder = split_processses(self.num_processes,len(self.station))
            with Manager() as manager:
                x = manager.list([None] * len(self.station))
                for proc_idx in range(processes):
                    index_start,index_end = get_indices(proc_idx,iterations,processes,remainder)
                    args = (x, index_start, index_end, start,end,step,proc_idx)
                    threads[proc_idx] = Process(target=self._batch_feed, args=args)
                    threads[proc_idx].start()
                for thread in threads:
                    thread.join()
                x=list(x)
            valid_x = [r for r in x if r is not None]
            skipped_stations = [i for i in range(len(x)) if x[i] is None]
            if len(valid_x) == 0:
                print("No valid input data generated.")
                exit()
            # this could cause memory issues with too many stations -> would need to reduce hourly computation
            batches_per_station = valid_x[0].shape[0]
            x = np.concatenate(valid_x, axis=0)
        else :
            delta = end - start
            raw = [self._load_data(start + s, start + s + self.length) for s in np.arange(0, delta, step)]
            x = [self._normalize(r, self.normalization, self.normalization_mode) for r in raw]
            #w = tukey(int(length*self.sampling_rate), self.taper)[np.newaxis,:,np.newaxis]
            x = np.stack(x, axis=0) * self.w
            batches_per_station = x.shape[0]
            skipped_stations = []
        v = 1 if self.verbose else 0
        print('Data ready. Predicting ...')
        try :
            #prediction = self.model.predict(x, verbose=v)
            prediction = self.model.predict(x, verbose=v, batch_size=64)
            # no memory issue but also no progress bar so not easy to check if faster. Killed.
            #prediction = self.model.predict_on_batch(x)
        except Exception as e:
            print(f"[Prediction Error] {e} . Setting to None")
            prediction = None
        # this should be faster because it does not split into batches as defined in model
        # produces memory issues though
        #print('Predicting ...',UTCDateTime())
        #prediction = self.model.predict_on_batch(x)
        #print('Finished',UTCDateTime())

        prediction = np.squeeze(prediction)
        self.skipped_stations = skipped_stations
        self.batches_per_station = batches_per_station
        
        if self.return_raw:
            return np.expand_dims(raw, axis=0), prediction
        else:
            return x, prediction


def phase_detection_station(model, client, station, channels, length, start, end,
                           cfg_model, cfg_data, cfg_pred, array_stations):
    """
    Run phase detection for a single station or list of stations.
    
    Initializes LivePhaseNet predictor, processes data in sliding windows,
    and combines overlapping predictions.
    
    Parameters
    ----------
    model : tensorflow.keras.Model
        Trained phase detection model
    client : obspy.Client
        Seismic data client
    station : list
        Station name(s) to process or sinlge array name if array detection
    channels : str
        Channel priority list (comma-separated)
    length : int
        Window length (seconds)
    start, end : obspy.UTCDateTime
        Time range
    cfg_model : namespace
        Model configuration
    cfg_data : namespace
        Data configuration with frequency band
    cfg_pred : namespace
        Prediction configuration
    array_stations : list or False
        Array dection configuration
    
    Returns
    -------
    pred_padded : numpy.ndarray or list
        Combined predictions (single array for one station, list for multiple)
    prediction : LivePhaseNet
        Predictor instance (for accessing metadata)
    
    Notes
    -----
    For multiple stations, uses parallel processing to combine windows.
    """
    
    print('Initializing')
    try:
        # Pre-load waveforms for the full time range (much faster than per-window requests)
        print(f"Pre-loading waveforms for {start} to {end + length}...")
        # Use same logic as _load_data for determining station list
        if array_stations:
            statlist = [sta[0] for sta in array_stations]
        else:
            statlist = station
        
        preloaded_stream = Stream()
        for i, stat in enumerate(statlist):
            try:
                st = load_station_waveforms(
                    client, stat, channels, start, end + length,
                    array_info=array_stations, station_index=i
                )
                preloaded_stream += st
            except Exception as e:
                print(f"  Warning: Could not load {stat}: {e}")
        
        if len(preloaded_stream) == 0:
            print("  Preloading failed, will try window-wise loading")
            preloaded_stream = None
        else:
            print(f"  Loaded {len(preloaded_stream)} traces")
        
        prediction = LivePhaseNet(
            model,
            client,
            station=station,
            channels=channels,
            length=length,
            return_raw=True,
            normalization=cfg_model.normalization.mode,
            normalization_mode=cfg_model.normalization.channel_mode,
            verbose=True,
            sampling_rate=cfg_model.data.sampling_rate,
            taper=cfg_model.augment.taper,
            bandpass=(cfg_data.lower_frequency, cfg_data.upper_frequency),
            array=array_stations,
            num_processes=cfg_pred.num_processes,
            stream=preloaded_stream  # Use pre-loaded stream for fast slicing
        )

        print("Predicting for time window and station:", start, end, station)
        _, pred = prediction.predict(start, end, cfg_pred.step)
        
        print("Combining output ...")
        if not array_stations :
            processes, threads, iterations, remainder = split_processses(
                cfg_pred.num_processes, len(statlist)
            )
            pred_list = []
            for i in range(len(statlist)):
                if i in prediction.skipped_stations:
                    print(f"No prediction for station {statlist[i]}")
                    pred_list.append(None)
                else:
                    start_idx = i * prediction.batches_per_station
                    end_idx = (i + 1) * prediction.batches_per_station
                    pred_list.append(pred[start_idx:end_idx])
            
            with Manager() as manager:
                pred_padded = manager.list([None] * len(statlist))
                for proc_idx in range(processes):
                    index_start, index_end = get_indices(proc_idx, iterations, processes, remainder)
                    args = (pred_padded, pred_list, index_start, index_end, start, end,
                           proc_idx, cfg_model, cfg_pred)
                    threads[proc_idx] = Process(target=batch_feed_padding, args=args)
                    threads[proc_idx].start()
                for thread in threads:
                    thread.join()
                pred_padded = list(pred_padded)
        else:
            pred_padded = [combine_output_from_sliding_windows(pred, start, end, cfg_model, cfg_pred)]

            _, p, s = np.split(pred_padded[0], 3, axis=-1)
            # plotting for QC:
            pred_stream = Stream()
            pred_stream += Trace(data=np.array(list(np.squeeze(p))),
                                header={'station': station, 'channel': 'P',
                                       'sampling_rate': cfg_model.data.sampling_rate,
                                       'starttime': start
                                       })
            pred_stream += Trace(data=np.array(list(np.squeeze(s))),
                                header={'station': station, 'channel': 'S',
                                       'sampling_rate': cfg_model.data.sampling_rate,
                                       'starttime': start
                                       })
            pred_stream.plot()
        
        return pred_padded, prediction
    
    except Exception as e:
        print(f"[{station}] Error during prediction: {e}")
        return None, None


def phase_detection(client, cfg_pred, cfg_model, cfg_data):
    """
    Main function for continuous seismic phase detection.
    
    Loads model, processes waveforms in sliding windows, and saves predictions
    for specified time periods and stations.
    
    Parameters
    ----------
    client : obspy.Client
        Seismic data client for waveform retrieval
    cfg_pred : namespace
        Prediction configuration with processing parameters
    cfg_model : namespace
        Model configuration with architecture and training settings
    cfg_data : namespace
        Data configuration with array definitions and parameters
    
    Returns
    -------
    None
        Writes prediction files to cfg_pred.output_dir
    
    Notes
    -----
    - Processes data in hourly windows (cfg_pred.window_length)
    - Supports both single station and array processing
    - Can save probabilities, waveforms, or picks
    - Handles parallel processing for detect_only mode
    - Requires cfg_pred.arrays to define array station patterns
    """
    from tqdm import tqdm
    
    # Set default configuration attributes
    if not hasattr(cfg_model.data, "extract_array_channels"):
        setattr(cfg_model.data, "extract_array_channels", False)
    if not hasattr(cfg_model.run, "custom_outname"):
        setattr(cfg_model.run, "custom_outname", False)

    # Determine model name
    if cfg_model.data.extract_array_channels:
        modelname = f'array{cfg_model.data.setname}_{cfg_model.model.type}'
    else:
        modelname = f'{cfg_model.data.input_dataset_name}_{cfg_model.model.type}'
    if cfg_model.run.custom_outname:
        modelname = f'{cfg_model.data.input_dataset_name}_{cfg_model.run.custom_outname}_{cfg_model.model.type}'
    if cfg_model.run.custom_outname and cfg_model.data.extract_array_channels:
        modelname = f'array{cfg_model.data.setname}_{cfg_model.run.custom_outname}_{cfg_model.model.type}'

    # Load model
    model = read_model(cfg_pred, cfg_model)
    length = model.layers[0].input.shape[1] // cfg_model.data.sampling_rate

    # Generate time windows
    # Continuous mode: hour-wise predictions
    times = []
    start_cont = UTCDateTime(cfg_pred.start_time)
    end_cont = UTCDateTime(cfg_pred.end_time)
    window = [0, cfg_pred.window_length]
    t = start_cont
    while t + cfg_pred.window_length <= end_cont:
        times.append(t)
        t += cfg_pred.window_length

    # Process waveforms
    if cfg_model.data.input_datatype in ['single_station_waveforms', 'array_waveforms']:
        
        if cfg_model.data.input_datatype == 'array_waveforms':
            statlist = cfg_pred.stations
            if cfg_model.data.extract_array_channels:
                array_stations = [sta for sta in cfg_data.extract_array_channels]
            else:
                array_stations = [sta for sta in cfg_data.use_these_arraystations]
        else:
            statlist = []
            array_stations = False
            # Get station list from cfg_pred.arrays if station is an array name
            for station in cfg_pred.stations:
                if hasattr(cfg_pred, 'arrays') and station in cfg_pred.arrays:
                    statlist.extend(cfg_pred.arrays[station])
                    # use full array with wildcard is used to define array:
                    if '*' in statlist[-1] :
                        geometry_file = getattr(cfg_pred, 'geometry_file', None)
                        statlist = get_array_stations(station, geometry_file=None)
                else:
                    statlist.append(station)

        channels = 'HHZ,HHE,HHN,BHZ,BHE,BHN,SHZ,SHE,SHN,BH1,BH2,sz,se,sn,bz,be,bn'
        folder = cfg_pred.output_dir

        for event_time in tqdm(times):
            event_time_stripped = str(event_time.isoformat().split('.')[0].replace(':', '')).strip()
            start = event_time - window[0]
            end = event_time + window[1]

            # Single station waveform processing
            if cfg_model.data.input_datatype == 'single_station_waveforms':
                results, prediction = phase_detection_station(
                    model, client, statlist, channels, length, start, end,
                    cfg_model, cfg_data, cfg_pred, array_stations
                )
                
                if results is None or prediction is None:
                    print(f"  Skipping time window due to prediction error")
                    continue
                
                if not cfg_pred.detect_only:
                    print("Saving output ...")
                    times = np.arange(int((cfg_pred.window_length + prediction.length) * 
                                        cfg_model.data.sampling_rate))
                    times = [start + i for i in times / cfg_model.data.sampling_rate]
                    
                    for i, station in enumerate(statlist):
                        pred_padded = results[i]
                        if pred_padded is None:
                            continue
                        
                        if cfg_pred.save_waveforms:
                            np.savez(
                                f'{folder}/{modelname}_{station}_{cfg_pred.stacking}_{event_time_stripped}_wave.npz',
                                t=times,
                                x=prediction._load_data(start, end + prediction.length, cut=False)[:-1],
                                y=pred_padded
                            )
                        
                        if cfg_pred.save_prob:
                            np.savez(
                                f'{folder}/{modelname}_{station}_{cfg_pred.stacking}_{event_time_stripped}.npz',
                                t=times,
                                y=pred_padded
                            )
                else:
                    # Detect-only mode: combine and save picks
                    if cfg_pred.combine_array_stations:
                        if cfg_pred.combine_array_stations != 'stack':
                            print(f"{cfg_pred.combine_array_stations} is not implemented for parallel processing!")
                            exit()
                    
                    print("Stacking for ensemble or single prediction ...")
                    valid_results = [r for r in results if r is not None]
                    valid_results = np.array(valid_results)
                    
                    if len(valid_results) == 0:
                        print("No valid results to stack.")
                        exit()
                    
                    stack = np.mean(np.array(valid_results), axis=0)
                    _, p_pred_tmp, s_pred_tmp = np.split(stack, 3, axis=-1)
                    p_pred = np.array([p_pred_tmp])
                    s_pred = np.array([s_pred_tmp])
                    
                    print("Getting and writing picks ...")
                    times = np.arange(int((cfg_pred.window_length + prediction.length) * 
                                        cfg_model.data.sampling_rate))
                    times = [start + i for i in times / cfg_model.data.sampling_rate]
                    
                    suffix = '_stack'
                    if cfg_pred.stations != 'ARCES':
                        suffix += '_' + cfg_pred.stations[0] + '_cont'
                    else:
                        suffix += '_cont'

                    thr_opt_p = cfg_model.evaluation.p_threshold
                    thr_opt_s = cfg_model.evaluation.s_threshold
                    suffix += f'_{thr_opt_p}_{thr_opt_s}'
                    outdir = f'{folder}/{modelname}{suffix}'
                    os.makedirs(outdir, exist_ok=True)
                    times = np.array([times])
                    save_picks(p_pred, s_pred, times, cfg_model.data.sampling_rate,
                             thr_opt_p, thr_opt_s, cfg_pred.stations[0], outdir)
            
            else:
                # Array waveforms: statlist should only include a single array identifier
                station = statlist[0]
                pred_padded, prediction = phase_detection_station(
                    model, client, station, channels, length, start, end,
                    cfg_model, cfg_data, cfg_pred, array_stations
                )
                if pred_padded is None or prediction is None:
                    continue

                print("Saving output ...")
                times = np.arange(int((cfg_pred.window_length + prediction.length) * 
                                    cfg_model.data.sampling_rate))
                times = [start + i for i in times / cfg_model.data.sampling_rate]
                
                if cfg_pred.save_waveforms:
                    np.savez(
                        f'{folder}/{modelname}_{station}_{cfg_pred.stacking}_{event_time_stripped}_wave.npz',
                        t=times,
                        x=prediction._load_data(start, end + prediction.length, cut=False)[:-1],
                        y=pred_padded
                    )
                
                if cfg_pred.save_prob:
                    np.savez(
                        f'{folder}/{modelname}_{station}_{cfg_pred.stacking}_{event_time_stripped}.npz',
                        t=times,
                        y=pred_padded
                    )


def combine_output_from_sliding_windows(pred, start, end, cfg_model, cfg_pred):
    """
    Combine predictions from overlapping sliding windows.
    
    Uses specified stacking method (median, mean, std, p25) to merge overlaps.
    
    Parameters
    ----------
    pred : list
        List of prediction arrays from windows
    start, end : obspy.UTCDateTime
        Time range
    cfg_model : namespace
        Model configuration
    cfg_pred : namespace
        Prediction configuration with stacking method
    
    Returns
    -------
    numpy.ndarray
        Combined predictions
    
    Notes
    -----
    This function populates the pred_padded by using slicing instead of concatenating
    a lot of arrays together.
    """
    if cfg_model.model.type.startswith('splitoutput'):
        pred = np.moveaxis(pred, 0, -1)

    sr = cfg_model.data.sampling_rate
    total = ((end - start) * sr) + len(pred[0])
    padded_length = int(total)

    pred_shape = pred[0].shape[1]

    pred_padded = np.full((len(pred), padded_length, pred_shape), np.nan)

    for i, p in enumerate(pred):
        before = int(i * cfg_pred.step * sr)
        after = int(total) - len(p) - before
        if after < 0:
            pred_padded[i, before:before + len(p)] = p[:after]
        else:
            pred_padded[i, before:before + len(p)] = p

    pred_padded = np.ma.masked_invalid(pred_padded)

    if cfg_pred.stacking == 'std':
        pred_padded = np.ma.std(pred_padded, axis=0)
    if cfg_pred.stacking == 'median':
        pred_padded = np.ma.median(pred_padded, axis=0)
    if cfg_pred.stacking == 'p25':
        pred_padded = np.ma.masked_invalid(
            np.nanpercentile(np.ma.filled(pred_padded, np.nan), 25, axis=0))
    if cfg_pred.stacking == 'mean':
        pred_padded = np.ma.mean(pred_padded, axis=0)

    return pred_padded


def batch_feed_padding(pred_padded, pred, index_start, index_end, start, end, proc_idx, cfg_model, cfg_pred):
    """
    Process batch of stations for window combination in parallel.
    
    Helper function for parallel processing of sliding window predictions.
    
    Parameters
    ----------
    pred_padded : multiprocessing.Manager.list
        Shared list to store combined predictions
    pred : list
        List of prediction arrays for each station
    index_start, index_end : int
        Range of stations to process
    start, end : obspy.UTCDateTime
        Time range
    proc_idx : int
        Process index for logging
    cfg_model : namespace
        Model configuration
    cfg_pred : namespace
        Prediction configuration
    """
    for index in range(index_start, index_end):
        try:
            print(f"[PID {os.getpid()}] Processing index {index}")
            pred_padded[index] = combine_output_from_sliding_windows(pred[index], start, end, cfg_model, cfg_pred)
        except Exception as e:
            print(f"[PID {os.getpid()}] Error at index {index}: {e} . Setting to None")
            pred_padded[index] = None


def load_cont_beams(cfg_pred, cfg_model):
    """
    Load continuous beam predictions from NPZ files.
    
    Reads beam detector output files written by beam_phase_detection()
    and creates an ObsPy Stream of P and S prediction traces.
    
    Parameters
    ----------
    cfg_pred : namespace
        Prediction configuration with:
        - output_dir : Output directory path
        - start_time, end_time : Time range
        - stations : Station list (first element is array name)
        - stacking : Stacking method used
        - window_length : Window length (seconds)
    cfg_model : namespace
        Model configuration with:
        - data.input_dataset_name : Dataset name
        - data.sampling_rate : Sampling rate
        - model.type : Model architecture
    
    Returns
    -------
    pred_stream : obspy.Stream
        Stream of prediction traces (P and S channels per beam azimuth)
    array_name : str
        Array name
    """
    pred_stream = Stream()
    
    # Build model name pattern
    if not hasattr(cfg_model.run, "custom_outname"):
        setattr(cfg_model.run, "custom_outname", False)
    if cfg_model.data.extract_array_channels:
        inputdata = f'array{cfg_model.data.setname}'
    else:
        inputdata = f'{cfg_model.data.input_dataset_name}'
    if cfg_model.run.custom_outname:
        inputdata = f'{cfg_model.data.input_dataset_name}_{cfg_model.run.custom_outname}'
    
    model_type = f'{inputdata}_{cfg_model.model.type}'
    array_name = cfg_pred.stations[0]
    folder = cfg_pred.output_dir
    
    # Find beam npz files: pattern is {model}_{array}_beam_{azimuth}_{stacking}_{time}.npz
    filelist = glob.glob(f'{folder}/{model_type}_{array_name}_beam_*_{cfg_pred.stacking}_2*.npz')
    
    if len(filelist) == 0:
        print(f"No beam files found matching: {folder}/{model_type}_{array_name}_beam_*_{cfg_pred.stacking}_2*.npz")
        return pred_stream, array_name
    
    for dfile in tqdm(sorted(filelist), desc="Loading beam files"):
        pred = np.load(dfile, allow_pickle=True)
        
        # Extract times
        try:
            times = pred['t']
            test = times[0]
        except (IndexError, KeyError):
            print(f"Warning: Could not read times from {dfile}, skipping")
            continue
        
        # Check time range
        if times[0] + 60.0 < cfg_pred.start_time or times[-1] - 600.0 > cfg_pred.end_time:
            print(f'Skipping {dfile}. Adjust start and end time in predict_config.yaml if this is not intended!')
            continue
        
        # Extract azimuth from file or from npz
        if 'azimuth' in pred.files:
            azimuth = int(pred['azimuth'])
        else:
            # Try to parse from filename: ..._beam_XXX_...
            import re
            match = re.search(r'_beam_(\d+)_', dfile)
            azimuth = int(match.group(1)) if match else 0
        
        # Create station name from array and azimuth
        station_name = f'{array_name}_B{azimuth:03d}'
        
        # Cut to window length
        cut = int(cfg_pred.window_length * cfg_model.data.sampling_rate)
        
        # Split predictions into P and S
        if cfg_model.model.type.startswith('splitoutput'):
            p, s = np.split(pred['y'], 2, axis=-1)
        else:
            _, p, s = np.split(pred['y'], 3, axis=-1)
        
        pred_stream += Trace(data=np.array(list(np.squeeze(p[:cut]))), 
                            header={'station': station_name, 'channel': 'P', 
                                   'sampling_rate': cfg_model.data.sampling_rate,
                                   'starttime': times[0]})
        pred_stream += Trace(data=np.array(list(np.squeeze(s[:cut]))),
                            header={'station': station_name, 'channel': 'S',
                                   'sampling_rate': cfg_model.data.sampling_rate,
                                   'starttime': times[0]})
    
    pred_stream.merge(method=1, fill_value='interpolate')
    print(pred_stream)
    
    # QC plot of all beam predictions
    if len(pred_stream) > 0:
        print("Displaying QC plot of beam predictions...")
        pred_stream.plot(equal_scale=False)
    
    return pred_stream, array_name


def load_cont_detections(cfg_pred, cfg_model):
    """
    Load continuous station predictions from NPZ files.
    
    Reads prediction probability files from sliding window processing
    and creates an ObsPy Stream for further analysis or combination.
    
    Parameters
    ----------
    cfg_pred : namespace
        Prediction configuration with:
        - output_dir : Output directory path
        - start_time, end_time : Time range
        - stations : Station list
        - stacking : Stacking method used
        - window_length : Window length (seconds)
    cfg_model : namespace
        Model configuration with:
        - model.type : Model architecture
        - data.input_dataset_name : Dataset name
        - data.extract_array_channels : Array configuration
        - data.sampling_rate : Sampling rate
        - run.custom_outname : Custom output name
    
    Returns
    -------
    pred_stream : obspy.Stream
        Stream of prediction traces (P and S channels per station)
    array : str or False
        Array name if stations belong to array, False otherwise
    """

    if not hasattr(cfg_model.run, "custom_outname"):
        setattr(cfg_model.run, "custom_outname", False)
    if cfg_model.data.extract_array_channels:
        inputdata = f'array{cfg_model.data.setname}'
    else:
        inputdata = f'{cfg_model.data.input_dataset_name}'
    if cfg_model.run.custom_outname:
        inputdata = f'{cfg_model.data.input_dataset_name}_{cfg_model.run.custom_outname}'
    if cfg_model.run.custom_outname and cfg_model.data.extract_array_channels:
        inputdata = f'array{cfg_model.data.setname}_{cfg_model.run.custom_outname}'

    array = False
    statlist = []
    # Get station list from cfg_pred.arrays if station is an array name
    for station in cfg_pred.stations:
        if hasattr(cfg_pred, 'arrays') and station in cfg_pred.arrays and 'array' not in inputdata:
            statlist.extend(cfg_pred.arrays[station])
            if '*' in statlist[-1] :
                geometry_file = getattr(cfg_pred, 'geometry_file', None)
                statlist = get_array_stations(station, geometry_file=None)
            array = station
        else:
            statlist.append(station)

    folder = cfg_pred.output_dir
    model_type = f'{inputdata}_{cfg_model.model.type}'

    pred_stream = Stream()
    for station in statlist:
        filelist = glob.glob(f'{folder}/{model_type}_{station}_{cfg_pred.stacking}_2*.npz')
        if len(filelist) == 0:
            continue

        for i, dfile in tqdm(enumerate(sorted(filelist)), total=len(filelist)):
            if 'wave' not in dfile:
                pred = np.load(dfile, allow_pickle=True)
                # bug in writing times in some files : regenerate times again
                try:
                    times = pred['t']
                    test = times[0]
                except IndexError:
                    times = np.arange(int((cfg_pred.window_length) * cfg_model.data.sampling_rate)) / cfg_model.data.sampling_rate
                    times = [UTCDateTime(cfg_pred.start_time) + 3600 * i + j for j in times]
                if times[0] + 60.0 < cfg_pred.start_time or times[-1] - 600.0 > cfg_pred.end_time:
                    print(f'Skipping {dfile}. Adjust start and end time in predict_config.yaml if this is not intended!')
                    continue
                sampling_dt = times[1] - times[0]
                if int(1. / sampling_dt) != cfg_model.data.sampling_rate:
                    print("Something is wrong")
                    exit()
                cut = int(cfg_pred.window_length * cfg_model.data.sampling_rate)

                if cfg_model.model.type.startswith('splitoutput'):
                    p, s = np.split(pred['y'], 2, axis=-1)
                else:
                    _, p, s = np.split(pred['y'], 3, axis=-1)
                
                pred_stream += Trace(data=np.array(list(np.squeeze(p[:cut]))), 
                                    header={'station': station, 'channel': 'P', 
                                           'sampling_rate': cfg_model.data.sampling_rate,
                                           'starttime': times[0]})
                pred_stream += Trace(data=np.array(list(np.squeeze(s[:cut]))),
                                    header={'station': station, 'channel': 'S',
                                           'sampling_rate': cfg_model.data.sampling_rate,
                                           'starttime': times[0]})

    pred_stream.merge(fill_value='interpolate')
    print(pred_stream)
    pred_stream.plot()
    return pred_stream, array


def load_manual_picks(pshape, sshape, times, pickfile):
    """
    Load manual phase picks from text file and create label arrays.
    
    Parameters
    ----------
    pshape : tuple
        Shape for P-wave label array
    sshape : tuple
        Shape for S-wave label array
    times : list
        List of UTCDateTime objects for each sample
    pickfile : str
        Path to pick file
    
    Returns
    -------
    tuple
        (p_true, s_true, ap_true, as_true) label arrays
    """
    p_true = np.full(pshape, 0)
    s_true = np.full(sshape, 0)
    ap_true = np.full(pshape, 0)
    as_true = np.full(sshape, 0)
    with open(pickfile, 'r') as fp:
        x = len(fp.readlines())
    fp = open(pickfile, 'r')
    counter = 0
    maxtime = max(times)
    mintime = min(times)
    for line in tqdm(fp, total=x):
        line = line.strip().split()
        if line[0] != '#':
            pick = UTCDateTime(line[0] + 'T' + line[1])
            if pick < mintime - 1.0:
                continue
            if pick > maxtime + 1.0:
                break
            idx = int((pick - times[0]) / (times[1] - times[0]))
            if idx is not None and idx > -1:
                if re.search('A0\.\..HE', line[3]):
                    p_true[idx] = 1.
                if re.search('A0\.\..HN', line[3]):
                    s_true[idx] = 1.
                if re.search('A0\.\..HZ', line[3]) or re.search('A0\.\..HE', line[3]):
                    ap_true[idx] = 1.
                if re.search('A0\.\..HZ', line[3]) or re.search('A0\.\..HN', line[3]):
                    as_true[idx] = 1.
    fp.close()
    return p_true, s_true, ap_true, as_true


def save_picks(p_pred, s_pred, times, sampling, thr_opt_p, thr_opt_s, station, outdir):
    """
    Extract picks from predictions and save to JSON file.
    
    Parameters
    ----------
    p_pred, s_pred : numpy.ndarray
        P and S-wave prediction probabilities
    times : array-like
        Time array (UTCDateTime timestamps)
    sampling : float
        Sampling rate (Hz)
    thr_opt_p, thr_opt_s : float
        Detection thresholds
    station : str
        Station name
    outdir : str
        Output directory
    
    Returns
    -------
    list
        List of Detection objects
    
    Notes
    -----
    Saves detections to {outdir}/picks_{station}.json
    """
    from utils import find_peaks
    
    p_pred = np.array(p_pred)
    s_pred = np.array(s_pred)
    pdetect = find_peaks(p_pred, thr_opt_p, distance=int(sampling * 0.5))
    sdetect = find_peaks(s_pred, thr_opt_s, distance=int(sampling * 0.5))
    detections = []
    
    for i, (p, s, t) in enumerate(zip(p_pred, s_pred, times)):
        for pdet in pdetect[i]:
            if isinstance(p[pdet], float):
                snr = p[pdet]
            else:
                snr = p[pdet][0]
            dur = t[pdet + np.argmax(p[pdet:] < thr_opt_p)]
            dur -= t[pdet - np.argmax(np.flip(p[:pdet + 1]) < thr_opt_p)]
            detection = Detection.create(
                time=t[pdet],
                duration_sec=dur,
                sta=0.0,
                lta=0.0,
                snr=snr,
                beam_id='P',
                threshold=thr_opt_p
            )
            detections.append(detection)
        for sdet in sdetect[i]:
            if isinstance(s[sdet], float):
                snr = s[sdet]
            else:
                snr = s[sdet][0]
            dur = t[sdet + np.argmax(s[sdet:] < thr_opt_s)]
            dur -= t[sdet - np.argmax(np.flip(s[:sdet + 1]) < thr_opt_s)]
            detection = Detection.create(
                time=t[sdet],
                duration_sec=dur,
                sta=0.0,
                lta=0.0,
                snr=snr,
                beam_id='S',
                threshold=thr_opt_s
            )
            detections.append(detection)
    
    # Convert detections to JSON-serializable format
    picks_data = {
        'station': station,
        'thresholds': {'P': thr_opt_p, 'S': thr_opt_s},
        'sampling_rate': sampling,
        'detections': [
            {
                'time': str(d.time),
                'phase': d.beam_id,
                'probability': float(d.snr),
                'duration_sec': float(d.duration_sec) if d.duration_sec else None,
                'threshold': float(d.threshold) if d.threshold else None
            }
            for d in detections
        ]
    }
    
    # Save to JSON file
    output_file = os.path.join(outdir, f'picks_{station}.json')
    with open(output_file, 'w') as f:
        json.dump(picks_data, f, indent=2)
    
    print(f"Saved {len(detections)} picks to {output_file}")
    return detections


# =============================================================================
# BEAM DETECTION FUNCTIONS
# =============================================================================

def get_array_stations(array_name, geometry_file=None):
    """
    Get list of station codes for a given array.
    
    Parameters
    ----------
    array_name : str
        Array name (e.g., 'ARCES', 'FINES', 'NORES', 'SPITS')
    geometry_file : str, optional
        Path to geometry JSON file
    
    Returns
    -------
    list
        List of station codes
    """
    from beamforming import load_array_geometries
    
    geometries = load_array_geometries(geometry_file)
    if array_name not in geometries:
        raise ValueError(f"Unknown array: {array_name}. Available: {list(geometries.keys())}")
    return list(geometries[array_name].keys())


def load_array_waveforms(client, array_name, stations, start, end, cfg, channels="BHZ"):
    """
    Load waveforms for all stations in an array.
    
    Parameters
    ----------
    client : obspy.clients.fdsn.Client
        FDSN client for data retrieval
    array_name : str
        Array name for network lookup
    stations : list
        List of station codes
    start, end : UTCDateTime
        Time range
    channels : str
        Channel code(s) to retrieve
    
    Returns
    -------
    obspy.Stream
        Stream containing all array waveforms
    """
    stream = Stream()
    for station in stations:
        if not cfg.skip_stations or station not in cfg.skip_stations :
            try:
                st = client.get_waveforms(
                    network="*",
                    station=station,
                    location="*",
                    channel=channels,
                    starttime=start,
                    endtime=end
                )
                if len(st) > 0:
                    stream += st
            except Exception as e:
                print(f"  Warning: Could not retrieve {station}: {e}")
    
    return stream


def preprocess_stream(stream, sampling_rate, bandpass=None, taper=0.01):
    """
    Preprocess waveforms for beamforming and prediction.
    
    Parameters
    ----------
    stream : obspy.Stream
        Input waveforms
    sampling_rate : float
        Target sampling rate
    bandpass : tuple or None
        (fmin, fmax) for bandpass filter
    taper : float
        Taper fraction
    
    Returns
    -------
    obspy.Stream
        Preprocessed stream
    """
    stream = stream.copy()
    stream.detrend()
    stream.taper(taper)
    
    if bandpass is not None:
        stream.filter('bandpass', freqmin=bandpass[0], freqmax=bandpass[1])
    
    # Resample if needed
    for tr in stream:
        if tr.stats.sampling_rate != sampling_rate:
            stream.resample(sampling_rate, no_filter=True)
            break
    
    return stream


def create_detection_beams(stream, geometry, azimuths, p_velocities, s_velocities, three_component=False):
    """
    Create P and S detection beams at specified azimuths and velocities.
    
    P-beam is always formed from vertical (Z) component.
    S-beam depends on three_component flag:
    - False (2-channel model): S-beam from vertical (Z) component
    - True (3-channel model): S-beam from horizontal components rotated to T and R
    
    Parameters
    ----------
    stream : obspy.Stream
        Preprocessed array waveforms with Z (and optionally N, E) components
    geometry : dict
        Station geometry {station: {'dx': km, 'dy': km}}
    azimuths : list
        List of back-azimuths [degrees]
    p_velocities : float or list
        P-wave apparent velocity(ies) [km/s]. Can be single float or list.
    s_velocities : float or list
        S-wave apparent velocity(ies) [km/s]. Can be single float or list.
    three_component : bool, optional
        If True, create S beams from rotated horizontal components (T, R).
        If False, create S beam from vertical component. Default False.
    
    Returns
    -------
    dict
        Beams organized as {(azimuth, p_vel, s_vel): {'P': Trace, 'S': Trace}}
        For three_component=True: {'P': Trace, 'S-T': Trace, 'S-R': Trace}
        Channel order for 3-component: [Z, R, T] = [P beam, S-R beam, S-T beam]
        If single velocities provided, keys are just azimuth for backwards compatibility.
    """
    from beamforming import compute_beam_time_delays, create_beam, select_component, rotate_to_rt
    
    # Normalize velocities to lists for uniform handling
    if isinstance(p_velocities, (int, float)):
        p_vel_list = [p_velocities]
    else:
        p_vel_list = list(p_velocities)
    
    if isinstance(s_velocities, (int, float)):
        s_vel_list = [s_velocities]
    else:
        s_vel_list = list(s_velocities)
    
    # Check if we have single velocities (for backwards compatible key format)
    single_velocity = len(p_vel_list) == 1 and len(s_vel_list) == 1
    
    beams = {}
    
    # Ensure all traces have same start time
    start_time = max(tr.stats.starttime for tr in stream)
    end_time = min(tr.stats.endtime for tr in stream)
    stream = stream.slice(start_time, end_time)
    
    # Force same start time for beamforming
    for tr in stream:
        tr.stats.starttime = start_time
    
    # Select vertical component for P-beam (and S-beam if not three_component)
    stream_z = select_component(stream, 'Z')
    if len(stream_z) == 0:
        print("  Warning: No Z component traces found, cannot create beams")
        return beams
    
    # For three-component, verify horizontal components exist
    if three_component:
        stream_n = select_component(stream, 'N')
        stream_e = select_component(stream, 'E')
        # Also try '1' and '2' for stations using numeric component codes
        if len(stream_n) == 0:
            stream_n = select_component(stream, '1')
        if len(stream_e) == 0:
            stream_e = select_component(stream, '2')
        
        if len(stream_n) == 0 or len(stream_e) == 0:
            print("  Warning: Missing horizontal components for 3-component beams, falling back to Z")
            three_component = False
    
    for azimuth in azimuths:
        # For 3-component, rotate horizontals for this azimuth using ObsPy
        if three_component:
            stream_r, stream_t = rotate_to_rt(stream, azimuth)
        
        for p_vel in p_vel_list:
            for s_vel in s_vel_list:
                # Use simple key for single velocity (backwards compatible)
                if single_velocity:
                    key = azimuth
                else:
                    key = (azimuth, p_vel, s_vel)
                
                beams[key] = {}
                
                # P-wave beam (always from Z component)
                p_delays = compute_beam_time_delays(geometry, azimuth, p_vel)
                try:
                    p_beam = create_beam(stream_z, p_delays, station_name=f'BEAM_{int(azimuth):03d}')
                    p_beam.stats.channel = 'P'
                    beams[key]['P'] = p_beam
                except ValueError as e:
                    print(f"  Warning: Could not create P beam at {azimuth}°, vel={p_vel}: {e}")
                    beams[key]['P'] = None
                
                # S-wave beam(s)
                s_delays = compute_beam_time_delays(geometry, azimuth, s_vel)
                
                if three_component:
                    # S-beam from Transverse component
                    try:
                        s_beam_t = create_beam(stream_t, s_delays, station_name=f'BEAM_{int(azimuth):03d}')
                        s_beam_t.stats.channel = 'S-T'
                        beams[key]['S-T'] = s_beam_t
                    except ValueError as e:
                        print(f"  Warning: Could not create S-T beam at {azimuth}°, vel={s_vel}: {e}")
                        beams[key]['S-T'] = None
                    
                    # S-beam from Radial component
                    try:
                        s_beam_r = create_beam(stream_r, s_delays, station_name=f'BEAM_{int(azimuth):03d}')
                        s_beam_r.stats.channel = 'S-R'
                        beams[key]['S-R'] = s_beam_r
                    except ValueError as e:
                        print(f"  Warning: Could not create S-R beam at {azimuth}°, vel={s_vel}: {e}")
                        beams[key]['S-R'] = None
                else:
                    # S-beam from Z component (2-channel model)
                    try:
                        s_beam = create_beam(stream_z, s_delays, station_name=f'BEAM_{int(azimuth):03d}')
                        s_beam.stats.channel = 'S'
                        beams[key]['S'] = s_beam
                    except ValueError as e:
                        print(f"  Warning: Could not create S beam at {azimuth}°, vel={s_vel}: {e}")
                        beams[key]['S'] = None
                
                # QC plot: show beams and input streams for this configuration
                #if three_component:
                #    plot_stream = Stream()
                #    # Add beams for this key
                #    if beams[key].get('P') is not None:
                #        plot_stream += beams[key]['P']
                #    if beams[key].get('S-T') is not None:
                #        plot_stream += beams[key]['S-T']
                #    if beams[key].get('S-R') is not None:
                #        plot_stream += beams[key]['S-R']
                #    plot_stream.plot(equal_scale=True) 
                #    # Add input streams
                #    plot_stream += stream_z
                #    plot_stream += stream_r
                #    plot_stream += stream_t
                #else:
                #    plot_stream = Stream()
                #    if beams[key].get('P') is not None:
                #        plot_stream += beams[key]['P']
                #    if beams[key].get('S') is not None:
                #        plot_stream += beams[key]['S']
                #    plot_stream.plot(equal_scale=True)
                #    plot_stream += stream_z
                #print(f"Plotting beams for {key}")
                #plot_stream.plot(equal_scale=False)
    
    return beams


def normalize_beam_data(data, mode='std', channel_mode='global'):
    """
    Normalize waveform data for beam processing.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input data array
    mode : str
        Normalization mode ('max' or 'std')
    channel_mode : str
        'local' or 'global' channel normalization
    
    Returns
    -------
    numpy.ndarray
        Normalized data
    """
    data = data - np.mean(data, axis=0, keepdims=True)
    
    if mode == 'max':
        if channel_mode == 'local':
            m = np.max(np.abs(data), axis=0, keepdims=True)
        else:
            m = np.max(np.abs(data), keepdims=True)
    elif mode == 'std':
        if channel_mode == 'local':
            m = np.std(data, axis=0, keepdims=True)
        else:
            m = np.std(data, keepdims=True)
    else:
        m = 1.0
    
    m = np.where(m == 0, 1, m)
    return data / m


def _prepare_beam_windows(results, beam_keys, beams, index_start, index_end, 
                          window_samples, sampling_rate, step, delta, 
                          n_channels, norm_mode, norm_channel_mode, taper_window):
    """
    Helper function to prepare beam windows in parallel.
    
    Called by multiprocessing workers to create normalized, tapered windows
    for a subset of beam configurations.
    
    For 2-channel models: uses P beam (Z) and S beam (Z)
    For 3-channel models: uses P beam (Z), S-R beam, S-T beam in order [Z, R, T]
    """
    for i in range(index_start, index_end):
        beam_key = beam_keys[i]
        try:
            beam_dict = beams[beam_key]
            
            if beam_dict.get('P') is None:
                results[i] = None
                continue
            
            p_beam = beam_dict['P']
            
            # Determine which S beams to use based on what's available and n_channels
            if n_channels >= 3 and 'S-T' in beam_dict and 'S-R' in beam_dict:
                # 3-channel: P (Z), S-T (transverse), S-R (radial)
                if beam_dict.get('S-T') is None or beam_dict.get('S-R') is None:
                    results[i] = None
                    continue
                s_beam_1 = beam_dict['S-T']
                s_beam_2 = beam_dict['S-R']
                use_three_component = True
            else:
                # 2-channel: P (Z), S (Z)
                if beam_dict.get('S') is None:
                    results[i] = None
                    continue
                s_beam = beam_dict['S']
                use_three_component = False
            
            # Get common length
            if use_three_component:
                min_len = min(len(p_beam.data), len(s_beam_1.data), len(s_beam_2.data))
            else:
                min_len = min(len(p_beam.data), len(s_beam.data))
            
            windows = []
            for s in np.arange(0, delta, step):
                start_idx = int(s * sampling_rate)
                end_idx = start_idx + window_samples
                if end_idx <= min_len:
                    p_window = p_beam.data[start_idx:end_idx]
                    
                    if use_three_component:
                        # Stack Z, R, T (P-Z beam, S-R beam, S-T beam)
                        st_window = s_beam_1.data[start_idx:end_idx]  # S-T (transverse)
                        sr_window = s_beam_2.data[start_idx:end_idx]  # S-R (radial)
                        window = np.stack([p_window, sr_window, st_window], axis=-1)  # Z, R, T order
                    else:
                        # Stack P, S for 2-channel
                        s_window = s_beam.data[start_idx:end_idx]
                        window = np.stack([p_window, s_window], axis=-1)
                    
                    windows.append(window)
            
            if len(windows) == 0:
                results[i] = None
                continue
            
            # Taper then normalize (same order as training)
            x = np.array(windows)
            x = x * taper_window
            x = np.array([normalize_beam_data(w, norm_mode, norm_channel_mode) for w in x])
            results[i] = x
            
        except Exception as e:
            print(f"[PID {os.getpid()}] Error preparing beam {beam_key}: {e}")
            results[i] = None


def beam_phase_detection(client, cfg_pred, cfg_model):
    """
    Main function for continuous beam-based phase detection.
    
    Applies beamforming to continuous array data, then runs a trained phase 
    detection model on the beams. Supports multiple beam configurations 
    (different azimuths and velocities for P and S waves).
    
    Parameters
    ----------
    client : obspy.clients.fdsn.Client
        FDSN client for data retrieval
    cfg_pred : namespace
        Prediction configuration with:
        - stations : List with array name as first element (e.g., ['ARCES'])
        - start_time, end_time : Time range
        - window_length : Processing window in seconds
        - step : Sliding window step in seconds
        - stacking : Combination method
        - output_dir : Output directory
        - save_prob : Save probability traces
        - save_picks : Save discrete picks
        - azimuths : List of back-azimuths or False (optional)
        - p_beam_vel : P-wave apparent velocity(ies) [km/s]. Float or list.
        - s_beam_vel : S-wave apparent velocity(ies) [km/s]. Float or list.
        - geometry_file : Path to geometry JSON (optional)
    cfg_model : namespace
        Model configuration
    
    Returns
    -------
    None
        Writes prediction files to cfg_pred.output_dir
    """
    from beamforming import load_array_geometries
    from scipy.signal.windows import tukey
    
    # Load model
    print("Loading model...")
    model = read_model(cfg_pred, cfg_model)
    window_samples = model.layers[0].input.shape[1]  # Model input size in samples
    model_length = window_samples / cfg_model.data.sampling_rate  # Model input size in seconds (e.g., 300s = 5 min)
    print(f"  Model input length: {model_length}s ({window_samples} samples at {cfg_model.data.sampling_rate} Hz)")
    print(f"  Processing window: {cfg_pred.window_length}s")
    
    # Load array geometry from JSON file
    print("Loading array geometry...")
    geometry_file = getattr(cfg_pred, 'geometry_file', None)
    geometries = load_array_geometries(geometry_file)
    
    if geometry_file:
        print(f"  Loaded from: {geometry_file}")
    else:
        import os
        default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'array_geometries.json')
        print(f"  Loaded from default: {default_path}")
    print(f"  Available arrays: {list(geometries.keys())}")
    
    # Get array name from stations list (first element)
    array_name = cfg_pred.stations[0]
    if array_name not in geometries:
        raise ValueError(f"Array '{array_name}' not found in geometry file. "
                        f"Available arrays: {list(geometries.keys())}")
    
    geometry = geometries[array_name]
    stations = list(geometry.keys())
    print(f"  Array {array_name}: {len(stations)} stations")
    
    # Get beam configuration
    if hasattr(cfg_pred, 'azimuths') and cfg_pred.azimuths:
        azimuths = cfg_pred.azimuths
    else:
        # Default: grid search every 30 degrees
        azimuths = list(range(0, 360, 30))
    
    p_velocities = getattr(cfg_pred, 'p_beam_vel', 8.0)
    s_velocities = getattr(cfg_pred, 's_beam_vel', 4.5)
    
    # Normalize to lists for uniform handling
    if isinstance(p_velocities, (int, float)):
        p_vel_list = [p_velocities]
    else:
        p_vel_list = list(p_velocities)
    
    if isinstance(s_velocities, (int, float)):
        s_vel_list = [s_velocities]
    else:
        s_vel_list = list(s_velocities)
    
    single_velocity = len(p_vel_list) == 1 and len(s_vel_list) == 1
    
    print(f"Beam configuration: {len(azimuths)} azimuths, P vel={p_vel_list} km/s, S vel={s_vel_list} km/s")
    
    # Get number of parallel processes
    num_processes = getattr(cfg_pred, 'num_processes', 1)
    print(f"Using {num_processes} parallel processes")
    
    # Get bandpass filter settings from cfg_model.data
    bandpass = None
    if hasattr(cfg_model.data, 'lower_frequency') and hasattr(cfg_model.data, 'upper_frequency'):
        bandpass = (cfg_model.data.lower_frequency, cfg_model.data.upper_frequency)
        print(f"Bandpass filter: {bandpass[0]}-{bandpass[1]} Hz")
    
    # Time windows
    start_cont = UTCDateTime(cfg_pred.start_time)
    end_cont = UTCDateTime(cfg_pred.end_time)
    
    # Generate processing windows
    times = []
    t = start_cont
    while t + cfg_pred.window_length <= end_cont:
        times.append(t)
        t += cfg_pred.window_length
    
    print(f"Processing {len(times)} time windows...")
    
    # Prepare output
    folder = cfg_pred.output_dir
    os.makedirs(folder, exist_ok=True)
    
    # Model name for output files
    if hasattr(cfg_model.data, 'extract_array_channels') and cfg_model.data.extract_array_channels:
        modelname = f'array{cfg_model.data.setname}_{cfg_model.model.type}'
    else:
        modelname = f'{cfg_model.data.input_dataset_name}_{cfg_model.model.type}'
    
    # Tukey window for tapering predictions
    taper_window = tukey(window_samples, cfg_model.augment.taper)[np.newaxis, :, np.newaxis]
    
    # Determine if three-component from model input shape or config
    n_channels = model.layers[0].input.shape[-1]
    three_component = n_channels >= 3  # 3+ channels means P + S-T + S-R beams
    
    if hasattr(cfg_model, 'data') and hasattr(cfg_model.data, 'three_component'):
        three_component = cfg_model.data.three_component
    
    # Select channels based on three_component flag
    if three_component:
        channels = "BHZ,BHN,BHE,HHZ,HHN,HHE"  # Load all components
    else:
        channels = "BHZ,HHZ"  # Only vertical for 2-channel
    
    beam_type = "3-component (P-Z, S-R, S-T)" if three_component else "2-component (P-Z, S-Z)"
    print(f"Beam type: {beam_type}")
    
    # Process each time window
    for event_time in tqdm(times, desc="Processing windows"):
        event_time_str = str(event_time.isoformat().split('.')[0].replace(':', '')).strip()
        
        # Load waveforms with overlap for sliding windows
        # Need extra model_length at the end for the last sliding window
        start = event_time
        end = event_time + cfg_pred.window_length + model_length
        
        print(f"\n  Loading waveforms: {start} to {end}")
        stream = load_array_waveforms(client, array_name, stations, start, end, cfg_pred, channels=channels)
        
        if len(stream) < 2:
            print(f"  Insufficient data ({len(stream)} traces), skipping window")
            continue
        
        # Preprocess
        stream = preprocess_stream(
            stream, 
            cfg_model.data.sampling_rate, 
            bandpass=bandpass,
            taper=cfg_model.augment.taper
        )
        
        n_beams = len(azimuths) * len(p_vel_list) * len(s_vel_list)
        print(f"  Creating {n_beams} beams ({len(azimuths)} azimuths x {len(p_vel_list)} P-vel x {len(s_vel_list)} S-vel)")
        beams = create_detection_beams(stream, geometry, azimuths, p_velocities, s_velocities, three_component)
        
        beam_keys = list(beams.keys())
        delta = cfg_pred.window_length
        step = cfg_pred.step
        
        # Prepare all beam windows in parallel
        print(f"  Preparing windows for {len(beam_keys)} beams using {num_processes} processes...")
        if num_processes > 1 and len(beam_keys) > 1:
            # Parallel preparation
            processes_to_use, threads, iterations, remainder = split_processses(
                num_processes, len(beam_keys)
            )
            with Manager() as manager:
                prepared_windows = manager.list([None] * len(beam_keys))
                for proc_idx in range(processes_to_use):
                    index_start, index_end = get_indices(proc_idx, iterations, processes_to_use, remainder)
                    args = (prepared_windows, beam_keys, beams, index_start, index_end,
                           window_samples, cfg_model.data.sampling_rate, step, delta,
                           n_channels, cfg_model.normalization.mode, 
                           cfg_model.normalization.channel_mode, taper_window)
                    threads[proc_idx] = Process(target=_prepare_beam_windows, args=args)
                    threads[proc_idx].start()
                for thread in threads:
                    thread.join()
                prepared_windows = list(prepared_windows)
        else:
            # Sequential preparation
            prepared_windows = [None] * len(beam_keys)
            _prepare_beam_windows(prepared_windows, beam_keys, beams, 0, len(beam_keys),
                                 window_samples, cfg_model.data.sampling_rate, step, delta,
                                 n_channels, cfg_model.normalization.mode,
                                 cfg_model.normalization.channel_mode, taper_window)
        
        # Count valid beams and windows per beam
        valid_indices = [i for i, w in enumerate(prepared_windows) if w is not None]
        if len(valid_indices) == 0:
            print("  No valid beam windows, skipping")
            continue
        
        windows_per_beam = prepared_windows[valid_indices[0]].shape[0]
        
        # Concatenate all windows for batch prediction
        all_windows = np.concatenate([prepared_windows[i] for i in valid_indices], axis=0)
        print(f"  Running prediction on {len(all_windows)} windows ({len(valid_indices)} beams x {windows_per_beam} windows)...")
        
        # Single batch prediction (much faster than per-beam prediction)
        all_preds = model.predict(all_windows, verbose=1, batch_size=64)
        all_preds = np.squeeze(all_preds)
        
        # Split predictions back to individual beams and save
        pred_results = {}  # Store for picks combination
        for idx, beam_idx in enumerate(valid_indices):
            beam_key = beam_keys[beam_idx]
            
            # Extract this beam's predictions
            pred_start = idx * windows_per_beam
            pred_end = (idx + 1) * windows_per_beam
            pred = all_preds[pred_start:pred_end]
            
            # Extract azimuth and velocities from key
            if single_velocity:
                azimuth = beam_key
                p_vel = p_vel_list[0]
                s_vel = s_vel_list[0]
            else:
                azimuth, p_vel, s_vel = beam_key
            
            # Get beam's actual starttime (accounts for time delay adjustment)
            beam_starttime = beams[beam_key]['P'].stats.starttime
            
            # Combine overlapping windows
            # Use beam_starttime for proper timing alignment
            pred_padded = combine_output_from_sliding_windows(
                pred, beam_starttime, beam_starttime + delta, cfg_model, cfg_pred
            )
            pred_results[beam_key] = pred_padded
            
            # Save results
            if cfg_pred.save_prob:
                output_times = np.arange(len(pred_padded)) / cfg_model.data.sampling_rate
                output_times = [beam_starttime + t for t in output_times]
                
                # Include velocities in filename if multiple
                if single_velocity:
                    fname = f'{folder}/{modelname}_{array_name}_beam_{int(azimuth):03d}_{cfg_pred.stacking}_{event_time_str}.npz'
                else:
                    fname = f'{folder}/{modelname}_{array_name}_beam_{int(azimuth):03d}_pv{p_vel}_sv{s_vel}_{cfg_pred.stacking}_{event_time_str}.npz'
                
                np.savez(
                    fname,
                    t=output_times,
                    y=pred_padded,
                    azimuth=azimuth,
                    p_velocity=p_vel,
                    s_velocity=s_vel
                )
        
        # Save picks if requested (combine all beam configurations)
        if cfg_pred.save_picks:
            # Collect all predictions from in-memory results
            all_preds = list(pred_results.values())
            
            if len(all_preds) > 0:
                # Take maximum across all azimuths
                combined = np.max(np.array(all_preds), axis=0)
                
                if cfg_model.model.type.startswith('splitoutput'):
                    p_pred, s_pred = np.split(combined, 2, axis=-1)
                else:
                    _, p_pred, s_pred = np.split(combined, 3, axis=-1)
                
                times_arr = np.arange(len(combined)) / cfg_model.data.sampling_rate
                times_arr = np.array([[start + t for t in times_arr]])
                
                p_pred = np.array([p_pred.squeeze()])
                s_pred = np.array([s_pred.squeeze()])
                
                outdir = f'{folder}/picks'
                os.makedirs(outdir, exist_ok=True)
                
                save_picks(
                    p_pred, s_pred, times_arr,
                    cfg_model.data.sampling_rate,
                    cfg_model.evaluation.p_threshold,
                    cfg_model.evaluation.s_threshold,
                    f'{array_name}_beam_{event_time_str}',
                    outdir
                )
    
    print("\nBeam phase detection complete!")
