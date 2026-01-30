"""
Beamforming Utilities for Seismic Array Processing
===================================================

Standalone beamforming functions that work with standard ObsPy Stream/Trace objects.
This module provides array processing capabilities without external dependencies
beyond ObsPy and NumPy.

Functions
---------
load_array_geometries
    Load array geometries from JSON file.
compute_beam_time_delays
    Calculate time delays for each station based on plane wave propagation.
create_beam
    Stack traces with time delays to form a beam (linear stack).

Example
-------
>>> from obspy import read
>>> from beamforming import compute_beam_time_delays, create_beam
>>> 
>>> # Define array geometry (station offsets in km from reference point)
>>> geometry = {
...     'ARA0': {'dx': 0.0, 'dy': 0.0},
...     'ARA1': {'dx': 0.5, 'dy': 0.3},
...     'ARA2': {'dx': -0.2, 'dy': 0.8},
... }
>>> 
>>> # Compute time delays for plane wave from azimuth 45°, velocity 8 km/s
>>> time_delays = compute_beam_time_delays(geometry, azimuth_deg=45.0, velocity_km_sec=8.0)
>>> 
>>> # Create beam from stream
>>> stream = read('data.mseed')
>>> beam_trace = create_beam(stream, time_delays)

Notes
-----
- Time delays are computed for a plane wave arriving from a given back-azimuth
- Positive time delay means the station records the wave later than reference
- Geometry offsets (dx, dy) are in kilometers relative to array reference point
  - dx: East-West offset (positive = East)
  - dy: North-South offset (positive = North)

Author: Integrated from seismonpy for array_tphasenet
"""

import json
import math
import os
from typing import Dict, Optional

import numpy as np
from obspy import Stream, Trace


def load_array_geometries(filepath: str = None) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Load array geometries from a JSON file.
    
    Parameters
    ----------
    filepath : str, optional
        Path to the JSON geometry file. If None, looks for 'array_geometries.json'
        in the same directory as this module.
    
    Returns
    -------
    dict
        Array geometries as {array_name: {station: {'dx': km, 'dy': km}}}.
    
    Example
    -------
    >>> geometries = load_array_geometries()
    >>> arces_geometry = geometries['ARCES']
    >>> delays = compute_beam_time_delays(arces_geometry, azimuth_deg=45, velocity_km_sec=8.0)
    """
    if filepath is None:
        # Default to array_geometries.json in same directory as this module
        module_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(module_dir, 'array_geometries.json')
    
    with open(filepath, 'r') as f:
        return json.load(f)


def wave_time_delay(
    dx_km: float, 
    dy_km: float, 
    back_azimuth_deg: float, 
    slowness_sec_per_km: float
) -> float:
    """
    Compute time delay for a plane wave at a station offset from reference.
    
    Parameters
    ----------
    dx_km : float
        East-West offset from reference point [km]. Positive = East.
    dy_km : float
        North-South offset from reference point [km]. Positive = North.
    back_azimuth_deg : float
        Back-azimuth of incoming wave [degrees]. 0° = North, 90° = East.
    slowness_sec_per_km : float
        Horizontal slowness [s/km]. Equal to 1/velocity.
    
    Returns
    -------
    float
        Time delay in seconds. Positive = wave arrives later at this station.
    """
    baz_rad = math.radians(back_azimuth_deg)
    
    # Direction cosines for wave propagation (opposite to back-azimuth)
    cx = -math.sin(baz_rad)
    cy = -math.cos(baz_rad)
    
    # Project station offset onto wave direction
    distance = dx_km * cx + dy_km * cy
    return distance * slowness_sec_per_km


def compute_beam_time_delays(
    geometry: Dict[str, Dict[str, float]],
    azimuth_deg: float,
    velocity_km_sec: float,
    reference_station: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute time delays for each station in an array.
    
    Parameters
    ----------
    geometry : dict
        Station geometry as {station_code: {'dx': float, 'dy': float}}.
        Offsets are in km from the array reference point.
    azimuth_deg : float
        Back-azimuth of incoming wave [degrees]. 0° = North, 90° = East.
    velocity_km_sec : float
        Apparent velocity of the wave [km/s].
    reference_station : str, optional
        Station code to use as reference (zero delay). If None, uses the
        array reference point (dx=0, dy=0).
    
    Returns
    -------
    dict
        Time delays as {station_code: delay_seconds}.
    
    Example
    -------
    >>> geometry = {
    ...     'STA1': {'dx': 0.0, 'dy': 0.0},
    ...     'STA2': {'dx': 1.0, 'dy': 0.0},
    ...     'STA3': {'dx': 0.0, 'dy': 1.0},
    ... }
    >>> delays = compute_beam_time_delays(geometry, azimuth_deg=90, velocity_km_sec=8.0)
    >>> # Wave from East: STA2 will have negative delay (arrives first)
    """
    # Handle reference station offset
    if reference_station is not None and reference_station in geometry:
        ref = geometry[reference_station]
        ref_dx = ref.get('dx', 0.0)
        ref_dy = ref.get('dy', 0.0)
    else:
        ref_dx, ref_dy = 0.0, 0.0
    
    # Compute slowness (handle zero velocity)
    slowness = 0.0 if velocity_km_sec <= 0 else 1.0 / velocity_km_sec
    
    time_delays = {}
    for station, offsets in geometry.items():
        dx = offsets.get('dx', 0.0) - ref_dx
        dy = offsets.get('dy', 0.0) - ref_dy
        time_delays[station] = wave_time_delay(dx, dy, azimuth_deg, slowness)
    
    return time_delays


def create_beam(
    stream: Stream,
    time_delays: Dict[str, float],
    slice_ends: bool = False,
    station_name: Optional[str] = None
) -> Trace:
    """
    Create a beam by stacking traces with time delays (linear stack).
    
    Parameters
    ----------
    stream : obspy.Stream
        Stream containing traces to stack. Each trace must have a station code
        that matches a key in time_delays.
    time_delays : dict
        Time delays as {station_code: delay_seconds} from compute_beam_time_delays().
    slice_ends : bool, optional
        If True, trim beam to only include samples where all traces contributed.
    station_name : str, optional
        Station name to assign to the beam trace.
    
    Returns
    -------
    obspy.Trace
        The stacked beam trace with stats:
        - beam_components: list of contributing station codes
        - beam_weights: array of contribution counts per sample
    
    Raises
    ------
    ValueError
        If stream has fewer than 2 traces, traces have inconsistent sampling rates
        or start times, or time delays are missing for a station.
    
    Example
    -------
    >>> from obspy import read
    >>> stream = read('array_data.mseed')
    >>> time_delays = {'STA1': 0.0, 'STA2': -0.05, 'STA3': 0.03}
    >>> beam = create_beam(stream, time_delays)
    """
    if len(stream) < 2:
        raise ValueError("At least 2 traces are required to create a beam")
    
    # Validate trace consistency
    sampling_rate = None
    channel = None
    start_time = None
    
    for trace in stream:
        if sampling_rate is None:
            sampling_rate = trace.stats.sampling_rate
        if channel is None:
            channel = trace.stats.channel
        if start_time is None:
            start_time = trace.stats.starttime
        
        if trace.stats.sampling_rate != sampling_rate:
            raise ValueError("All traces must have equal sampling rates")
        if trace.stats.starttime != start_time:
            raise ValueError("All traces must have equal start times")
    
    # Convert time delays to sample offsets
    sample_offsets = {}
    for trace in stream:
        station = trace.stats.station
        if station not in time_delays:
            raise ValueError(f"Time delay not provided for station '{station}'")
        sample_offsets[station] = int(round(time_delays[station] * sampling_rate))
    
    # Calculate beam dimensions
    max_offset = max(sample_offsets.values()) - min(sample_offsets.values())
    max_trace_length = max(len(trace) for trace in stream)
    beam_length = max_offset + max_trace_length
    
    # Initialize arrays
    beam_data = np.zeros(beam_length, dtype=np.float64)
    weights = np.zeros(beam_length, dtype=np.int32)
    
    # Stack traces (linear stack)
    max_sample_offset = max(sample_offsets.values())
    components = []
    
    for trace in stream:
        station = trace.stats.station
        components.append(station)
        
        sample_offset = max_sample_offset - sample_offsets[station]
        current_slice = slice(sample_offset, sample_offset + len(trace))
        
        # Handle masked arrays (gaps in data)
        if isinstance(trace.data, np.ma.MaskedArray):
            valid = ~trace.data.mask
            weights[current_slice][valid] += 1
            beam_data[current_slice][valid] += trace.data[valid]
        else:
            weights[current_slice] += 1
            beam_data[current_slice] += trace.data
    
    # Normalize by number of contributing traces
    valid_samples = weights > 0
    beam_data[valid_samples] /= weights[valid_samples]
    
    # Mask samples with no contributions
    if np.any(weights == 0):
        beam_data = np.ma.masked_where(weights == 0, beam_data, copy=False)
    
    # Calculate beam start time (adjusted for max delay)
    beam_start_time = start_time - max_sample_offset / sampling_rate
    
    # Create output trace
    beam = Trace(data=beam_data)
    beam.stats.sampling_rate = sampling_rate
    beam.stats.starttime = beam_start_time
    beam.stats.channel = channel
    beam.stats.station = station_name if station_name else 'BEAM'
    
    # Add beam-specific metadata
    beam.stats.beam_components = components
    beam.stats.beam_weights = weights
    
    # Optionally trim to region where all traces contributed
    if slice_ends:
        trim_start = beam_start_time + max_offset * beam.stats.delta
        trim_end = beam.stats.endtime - max_offset * beam.stats.delta
        beam.trim(starttime=trim_start, endtime=trim_end, nearest_sample=False)
    
    return beam


def rotate_to_rt(stream: Stream, back_azimuth_deg: float, inventory=None) -> tuple:
    """
    Rotate horizontal components (N, E) to Radial (R) and Transverse (T) using ObsPy.
    
    Parameters
    ----------
    stream : obspy.Stream
        Stream containing Z, N, E components for each station
    back_azimuth_deg : float
        Back-azimuth of incoming wave [degrees]. 0° = North, 90° = East.
    inventory : obspy.Inventory, optional
        Station inventory for proper ZNE orientation. If provided, will first
        rotate to ZNE using sensor orientations from inventory.
    
    Returns
    -------
    tuple
        (stream_r, stream_t) - Radial and Transverse component streams
    
    Notes
    -----
    Uses ObsPy's stream.rotate() method which handles:
    - Proper sensor orientations via inventory
    - Standard seismological rotation conventions
    """
    # Work on a copy to avoid modifying the original
    st = stream.copy()
    
    # First ensure proper ZNE orientation if inventory provided
    if inventory is not None:
        try:
            st.rotate(method="->ZNE", inventory=inventory)
        except Exception as e:
            print(f"  Warning: Could not rotate to ZNE: {e}")
    
    # Rotate NE to RT
    try:
        st.rotate(method="NE->RT", back_azimuth=back_azimuth_deg)
    except Exception as e:
        print(f"  Warning: Could not rotate NE->RT: {e}")
        return Stream(), Stream()
    
    # Select R and T components
    stream_r = select_component(st, 'R')
    stream_t = select_component(st, 'T')
    
    return stream_r, stream_t


def select_component(stream: Stream, component: str) -> Stream:
    """
    Select traces by component (last character of channel code).
    
    Parameters
    ----------
    stream : obspy.Stream
        Input stream
    component : str
        Component letter ('Z', 'N', 'E', 'R', 'T', '1', '2', etc.)
    
    Returns
    -------
    obspy.Stream
        Stream with only matching components
    """
    return Stream([tr for tr in stream if tr.stats.channel[-1].upper() == component.upper()])
