"""
Utility Functions for MINEM Array Detection
============================================

This module provides utility functions for seismic phase detection, evaluation,
and prediction tasks. It combines functionality previously split between eval_utils.py
and predict_utils.py into a unified module.

Sections
--------
1. Evaluation Utilities
   - Performance metrics (precision, recall, F1 score)
   - Residual calculations
   - Peak finding and filtering
   - Visualization (L-curves, grids)
   
2. Prediction Utilities
   - Model loading and prediction
   - Array processing and beamforming
   - Detection grouping and combination

Usage
-----
    from utils import precision, recall, f1_score, calculate_residual
    from utils import LivePhaseNet, group_and_combine_phase_detections

Dependencies
------------
    - numpy, scipy: Numerical operations
    - tensorflow: Model inference
    - obspy: Seismological data handling
    - matplotlib: Visualization
    - beamforming: Array beamforming utilities (local module)

"""

# Standard library imports
import sys
import os

# Third-party imports
import numpy as np
import scipy as sp
from scipy.spatial import distance
from scipy.signal import find_peaks as find_peaks_scipy
from tqdm import tqdm
from obspy import UTCDateTime, Stream, Trace
import matplotlib.pyplot as plt
import tensorflow as tf

# =============================================================================
# SECTION 1: EVALUATION UTILITIES
# =============================================================================
# Functions for evaluating model performance, calculating metrics, and
# visualizing results on test data.
# =============================================================================

def residuals(y_true, y_pred, cfg, dt):
    """
    Calculate temporal residuals between true and predicted arrival times.
    
    Parameters
    ----------
    y_true : list of arrays
        True arrival time indices for each event
    y_pred : list of arrays
        Predicted arrival time indices for each event
    cfg : namespace
        Configuration with cfg.data.sampling_rate
    dt : float or None
        Maximum residual (seconds) to include. If None, include all.
    
    Returns
    -------
    numpy.ndarray
        Residuals in seconds (mean for events with multiple peaks)
    """
    if dt :
        r = np.array([np.nanmean(a-b) / cfg.data.sampling_rate for a,b in zip(y_true, y_pred) if np.nanmean(np.abs(a-b)) / cfg.data.sampling_rate < dt])
    else :
        r = np.array([np.nanmean(a-b) / cfg.data.sampling_rate for a,b in zip(y_true, y_pred)])
    return r

def calculate_residual(y_true, y_pred, cfg, dt=0.5, th=0.5, filter=False, dt_limit=False):
    """
    Calculate temporal residuals with peak finding and filtering.
    
    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth probability arrays (events, samples, channels)
    y_pred : numpy.ndarray
        Predicted probability arrays (events, samples, channels)
    cfg : namespace
        Configuration with cfg.data.sampling_rate
    dt : float, default=0.5
        Time window (seconds) for peak separation
    th : float, default=0.5
        Probability threshold for peak detection
    filter : bool, default=False
        Filter to events with detections above threshold and single true peaks
    dt_limit : float or False, default=False
        Maximum residual to include in output
    
    Returns
    -------
    numpy.ndarray
        Temporal residuals in seconds
    """
    true_peaks = find_peaks(y_true, height=th, distance=dt*cfg.data.sampling_rate)
    pred_peaks = find_peaks(y_pred, height=th, distance=dt*cfg.data.sampling_rate)

    if filter:
        th_idx = np.any(y_pred > th, axis=(1,2))
        num_idx = np.asarray([len(tp) < 2 for tp in true_peaks])
        idx = np.where(np.logical_and(th_idx, num_idx))[0]
        true_peaks = [true_peaks[i] for i in idx]
        pred_peaks = [pred_peaks[i] for i in idx]

    pred_peaks = filter_pred(true_peaks, pred_peaks)
    if dt_limit : dt_limti = dt
    return residuals(true_peaks, pred_peaks,cfg,dt=dt_limit)


def find_peaks(y,
               height=0.5,
               threshold=None,
               distance=int(40.0*0.5),
               prominence=None):
    """
    Find peaks in probability arrays for multiple events.
    
    Parameters
    ----------
    y : array-like
        Probability arrays (events, samples, channels)
    height : float, default=0.5
        Minimum peak height
    threshold : float, optional
        Vertical distance to neighboring samples
    distance : int, default=20
        Minimum horizontal distance between peaks (samples)
    prominence : float, optional
        Required peak prominence
    
    Returns
    -------
    list
        List of peak indices for each event
    """
    return list(map(lambda x: find_peaks_scipy(np.squeeze(x),
                                                   height,
                                                   threshold,
                                                   distance,
                                                   prominence)[0], y))

def filter_pred(y_true, y_pred):
    """
    Match each true peak to its closest predicted peak.
    
    Parameters
    ----------
    y_true : list
        List of true peak arrays for each event
    y_pred : list
        List of predicted peak arrays for each event
    
    Returns
    -------
    numpy.ndarray
        Filtered predictions with closest match for each true peak
    """
    out = []
    for a, b in zip(y_true, y_pred):
        try:
            A = np.asarray([[abs(c-v) for c in a] for v in b]).T
            i = np.argmin(A, axis=1)
            x = b[i]
        except IndexError:
            x = np.array([np.nan]*len(a))
        out.append(x)
    return np.array(out,dtype=object)

def within_distance(t, p, cfg, dt = 1):
    """
    Check if true peaks have predictions within time tolerance.
    
    Parameters
    ----------
    t : array-like
        True peak indices
    p : array-like
        Predicted peak indices
    cfg : namespace
        Configuration with cfg.data.sampling_rate
    dt : float, default=1
        Time tolerance in seconds
    
    Returns
    -------
    list of bool
        True for each peak with a prediction within dt seconds
    """
    out = []
    for a in t:
        p = [b if not np.isnan(b) else a for i, b in enumerate(p)]
        out.append(any([abs(a-b) < dt * cfg.data.sampling_rate for b in p]))
    return out

def not_within_distance(t, p, cfg, dt = 1):
    """
    Find true peaks without predictions within time tolerance (missed detections).
    
    Parameters
    ----------
    t : array-like
        True peak indices
    p : array-like
        Predicted peak indices
    cfg : namespace
        Configuration with cfg.data.sampling_rate
    dt : float, default=1
        Time tolerance in seconds
    
    Returns
    -------
    list
        True peaks that have no predictions within dt seconds
    """
    out = []
    for a in t:
        p = [b if not np.isnan(b) else a for i, b in enumerate(p)]
        if all([abs(a-b) >= dt * cfg.data.sampling_rate for b in p]): out.append(a)
    return out

def js_divergence(true, pred):
    """
    Calculate mean Jensen-Shannon divergence between probability distributions.
    
    Measures similarity between true and predicted probability distributions,
    only for samples where both have non-zero probability mass.
    
    Parameters
    ----------
    true : numpy.ndarray
        True probability distributions (samples, classes)
    pred : numpy.ndarray
        Predicted probability distributions (samples, classes)
    
    Returns
    -------
    float
        Mean squared Jensen-Shannon divergence
    """
    true_idx = np.where(np.sum(true, axis=1)>1e-7)[0]
    pred_idx = np.where(np.sum(pred, axis=1)>1e-7)[0]
    idx = np.asarray(list(set(true_idx).intersection(set(pred_idx))))

    return np.mean([distance.jensenshannon(t,p)**2 for t,p in zip(true[idx], pred[idx])])

def recall(y_true, y_pred, cfg, dt=1.0, th=0.5, livemode=False):
    """
    Calculate recall (sensitivity) for phase detection.
    
    Recall = True Positives / (True Positives + False Negatives)
    
    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth probability arrays
    y_pred : numpy.ndarray
        Predicted probability arrays
    cfg : namespace
        Configuration with cfg.data.sampling_rate
    dt : float, default=1.0
        Time tolerance (seconds) for matching detections
    th : float, default=0.5
        Probability threshold for peak detection
    livemode : bool, default=False
        If True, compute per-pick recall; if False, per-event recall
    
    Returns
    -------
    float
        Recall score between 0 and 1
    """
    peaks_true = find_peaks(y_true, th ) #, distance=dt*cfg.data.sampling_rate )
    peaks_pred = find_peaks(y_pred, th ) #, distance=dt*cfg.data.sampling_rate)
    r = 0
    c = 0
    for a, b in zip(peaks_true, peaks_pred):
        if livemode : 
            r += np.array(within_distance(a, b, cfg, dt=dt)).sum()
            # all relevant elements
            c += len(a)
        else :
            if any(within_distance(a, b, cfg, dt=dt)):
                r += 1
    if livemode : return r / c
    else : return r / len(peaks_true)

def missed_detections(y_true, y_pred, cfg, dt=1.0, th=0.5):
    """
    Find missed detections (false negatives) in predictions.
    
    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth probability arrays
    y_pred : numpy.ndarray
        Predicted probability arrays
    cfg : namespace
        Configuration with cfg.data.sampling_rate
    dt : float, default=1.0
        Time tolerance (seconds) for matching
    th : float, default=0.5
        Probability threshold for peak detection
    
    Returns
    -------
    list
        List of missed peak indices for each event
    """
    peaks_true = find_peaks(y_true, th ) #, distance=dt*cfg.data.sampling_rate )
    peaks_pred = find_peaks(y_pred, th ) #, distance=dt*cfg.data.sampling_rate)
    out=[]
    for a, b in zip(peaks_true, peaks_pred):
        out.append(not_within_distance(a, b, cfg, dt=dt))
    return out


def precision(y_true, y_pred, cfg, dt=1.0, th=0.5, livemode=False):
    """
    Calculate precision (positive predictive value) for phase detection.
    
    Precision = True Positives / (True Positives + False Positives)
    
    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth probability arrays
    y_pred : numpy.ndarray
        Predicted probability arrays
    cfg : namespace
        Configuration with cfg.data.sampling_rate
    dt : float, default=1.0
        Time tolerance (seconds) for matching detections
    th : float, default=0.5
        Probability threshold for peak detection
    livemode : bool, default=False
        If True, compute per-pick precision; if False, per-event precision
    
    Returns
    -------
    float
        Precision score between 0 and 1
    """
    peaks_true = find_peaks(y_true, th )
    peaks_pred = find_peaks(y_pred, th )
    r = 0
    c = 0
    for a, b in zip(peaks_true, peaks_pred):
        if livemode :
            # count with in distance items in predictions to take into account mutiple peaks
            r += np.array(within_distance(b, a, cfg, dt=dt)).sum()
            c += len(b)
        else :
            if all(within_distance(a, b, cfg, dt=dt)):
                r += 1
    if livemode : return r / c
    else : return r / len(peaks_true)

def f1_score(y_true, y_pred, cfg, dt=1.0, th=0.5, livemode=False):
    """
    Calculate F1 score (harmonic mean of precision and recall).
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth probability arrays
    y_pred : numpy.ndarray
        Predicted probability arrays
    cfg : namespace
        Configuration with cfg.data.sampling_rate
    dt : float, default=1.0
        Time tolerance (seconds) for matching detections
    th : float, default=0.5
        Probability threshold for peak detection
    livemode : bool, default=False
        If True, compute per-pick F1; if False, per-event F1
    
    Returns
    -------
    float
        F1 score between 0 and 1
    """
    p = precision(y_true, y_pred, cfg, dt, th, livemode)
    r = recall(y_true, y_pred, cfg, dt, th, livemode)
    return 2 * ((p * r) / (p + r + 1e-7))

def initialize_tree(tree, utctime):
    """
    Initialize nested dictionary tree structure for temporal event indexing.
    
    Creates hierarchical structure: year->month->day->hour->minute->list
    
    Parameters
    ----------
    tree : dict
        Dictionary to populate with time hierarchy
    utctime : obspy.UTCDateTime
        Time to create branches for
    
    Returns
    -------
    None
        Modifies tree in-place
    """
    if not utctime.year in tree:
        tree[utctime.year] = {}

    if not utctime.month in tree[utctime.year]:
        tree[utctime.year][utctime.month] = {}

    if not utctime.day in tree[utctime.year][utctime.month]:
        tree[utctime.year][utctime.month][utctime.day] = {}

    if not utctime.hour in tree[utctime.year][utctime.month][utctime.day]:
        tree[utctime.year][utctime.month][utctime.day][utctime.hour] = {}

    if not utctime.minute in tree[utctime.year][utctime.month][utctime.day][utctime.hour]:
        tree[utctime.year][utctime.month][utctime.day][utctime.hour][utctime.minute] = []

def add_to_tree_withindex(tree, t, index):
    """
    Add event time and index to temporal search tree.
    
    Also adds to adjacent minutes if near boundary (for efficient searching).
    
    Parameters
    ----------
    tree : dict
        Temporal search tree (created by initialize_tree)
    t : obspy.UTCDateTime
        Event time
    index : int
        Event index
    
    Returns
    -------
    None
        Modifies tree in-place
    """
    initialize_tree(tree, t)

    tree[t.year][t.month][t.day][t.hour][t.minute].append([t,index])

    # If on 'the edge' to previous or next minute, add it there too
    if t.second < 5:
        prev_t = UTCDateTime(t) - 60
        initialize_tree(tree, prev_t)
        tree[prev_t.year][prev_t.month][prev_t.day][prev_t.hour][prev_t.minute].append([t,index])

    if t.minute > 55:
        next_t = UTCDateTime(t) + 60
        initialize_tree(tree, next_t)
        tree[next_t.year][next_t.month][next_t.day][next_t.hour][next_t.minute].append([t,index])

def find_best_match_withindex(tree,t,match_threshold):
    """
    Find closest temporal match in search tree within threshold.
    
    Parameters
    ----------
    tree : dict
        Temporal search tree
    t : obspy.UTCDateTime
        Query time
    match_threshold : float
        Maximum time difference (seconds) to accept
    
    Returns
    -------
    best_match : obspy.UTCDateTime or None
        Closest matching time within threshold
    best_ind : int or None
        Index of closest match
    """
    best_match = None
    best_ind = None
    try : potential_matches = tree[t.year][t.month][t.day][t.hour][t.minute]
    except :
        potential_matches = []
    for pm,i in potential_matches:
        diff = abs(t - pm)
        if diff < match_threshold:
            if best_match is None:
                best_match = pm
                best_ind = i
            else:
                if diff < (t - best_match):
                    best_match = pm
                    best_ind = i
    return best_match,best_ind


def plot_L_curve(p_yte, p_pred,s_yte, s_pred, model, cfg, cont=False,stat=None,useforprec=False,eval_output_dir='evaluation_output'):
    """
    Generate precision-recall curves (L-curves) and find optimal thresholds.
    
    Computes precision and recall at multiple probability thresholds to create
    L-curves, and identifies optimal thresholds that maximize F1 score.
    
    Parameters
    ----------
    p_yte : numpy.ndarray
        Ground truth P-wave labels
    p_pred : numpy.ndarray
        Predicted P-wave probabilities
    s_yte : numpy.ndarray
        Ground truth S-wave labels
    s_pred : numpy.ndarray
        Predicted S-wave probabilities
    model : str
        Model name for output files
    cfg : namespace
        Configuration with evaluation parameters
    cont : bool, default=False
        Continuous mode flag (affects output filename)
    stat : str, optional
        Station name for continuous mode output
    useforprec : tuple, optional
        Alternative labels (p_yte_alt, s_yte_alt) for precision calculation
    eval_output_dir : str, default='evaluation_output'
        Output directory for results
    
    Returns
    -------
    thr_opt_p : float
        Optimal P-wave probability threshold
    thr_opt_s : float
        Optimal S-wave probability threshold
    fig : matplotlib.figure.Figure
        Generated figure object
    """
    if useforprec :
        p_yte_pre = useforprec[0]
        s_yte_pre = useforprec[1]
    else :
        p_yte_pre = p_yte
        s_yte_pre = s_yte
    if cont : dt = cfg.evaluation.dt_cont
    else : dt = cfg.evaluation.dt

    # Don't think this is needed
    #if cfg.model.type.startswith('splitoutput'):
    #    # Combine multi-head outputs to 3-channel format
    #    combined_pred = combine_split_heads_to_3channel(p_pred, s_pred)
    #    p_pred = combined_pred[..., 1]  # Extract P channel
    #    s_pred = combined_pred[..., 2]  # Extract S channel

    precision_p=[]
    recall_p = []
    precision_s=[]
    recall_s = []
    #threshold_list = np.concatenate(([0.0,0.05,0.01,0.025],np.arange(0,1,0.05),[0.98]))
    threshold_list = np.concatenate(([0.0,0.0005,0.001,0.002,0.005,0.01,0.025],np.arange(0.05,1,0.05),[0.98]))
    f1_max_p = 0.0
    f1_max_s = 0.0
    thr_opt_p = False
    thr_opt_s = False
    for thresh in threshold_list :
        precision_p.append(round(precision(p_yte_pre, p_pred, cfg, dt=dt, th=thresh, livemode=True), 2))
        recall_p.append(round(recall(p_yte, p_pred, cfg, dt=dt, th=thresh, livemode=True), 2))
        precision_s.append(round(precision(s_yte_pre, s_pred, cfg, dt=dt, th=thresh, livemode=True), 2))
        recall_s.append(round(recall(s_yte, s_pred, cfg, dt=dt, th=thresh, livemode=True), 2))
        f1_p=round(f1_score(p_yte, p_pred, cfg, dt=dt, th=thresh, livemode=True), 2)
        f1_s=round(f1_score(s_yte, s_pred, cfg, dt=dt, th=thresh, livemode=True), 2)
        print('P threshold:',round(thresh, 2),'F1 P:',round(f1_p, 2))
        print('S threshold:',round(thresh, 2),'F1 S:',round(f1_s, 2))
        if f1_p > f1_max_p  :
            f1_max_p = f1_p
            thr_opt_p = thresh
        if f1_s > f1_max_s :
            f1_max_s = f1_s
            thr_opt_s = thresh
    plt.plot(precision_p,recall_p)
    plt.plot(precision_s,recall_s)
    if not cont :
        with open(f'{cfg.run.outputdir}/{eval_output_dir}/L_curve_{model}.txt', 'w+') as f:
            for th,pp,pr,sp,sr in zip(threshold_list,precision_p,recall_p,precision_s,recall_s):
                f.write('%s %s %s %s %s\n' % (th,pp,pr,sp,sr))
    else :
        with open(f'{cfg.run.outputdir}/{eval_output_dir}/L_curve_{model}_{stat}_cont.txt', 'w+') as f:
            for th,pp,pr,sp,sr in zip(threshold_list,precision_p,recall_p,precision_s,recall_s):
                f.write('%s %s %s %s %s\n' % (th,pp,pr,sp,sr))
    plt.xlabel('Precision',fontsize=15)
    plt.ylabel('Recall',fontsize=15)
    for p,r,word in zip(precision_p[1::2],recall_p[1::2], threshold_list[1::2]) :
        plt.text(p+.01, r+.03, "%1.2f" % (word), fontsize=12)
    for p,r,word in zip(precision_s[1::2],recall_s[1::2], threshold_list[1::2]) :
        plt.text(p-.06, r-.06, "%1.2f" % (word), fontsize=12)
    plt.text(0.1, 0.85, "Thresholds", fontsize=15)
    if not cont : plt.legend(['P waves','S waves'],fontsize=15,loc='lower left')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim(0,1.1)
    plt.ylim(0,1.1)
    if cfg.evaluation.save_fig and not cont :
        plt.savefig(f'{cfg.run.outputdir}/{eval_output_dir}/L_curve_{model}.png')
    return thr_opt_p,thr_opt_s,plt.gcf()

def create_grid(pred_dict,dist_bins,nsamples,ids,distances,stations,phase,cfg,
                pick=False,smask=False,pvel=7.0,svel=3.5,redvel=9999.):
    """
    Create distance-time grid of predictions for visualization.
    
    Aligns predictions by distance and theoretical travel time to create
    stacked grid visualization. Used for distance-time plots.
    
    Parameters
    ----------
    pred_dict : dict
        Dictionary of predictions keyed by '{event_id}_{station}'
    dist_bins : list of tuples
        Distance bins [(min1,max1), (min2,max2), ...] in km
    nsamples : int
        Number of time samples in grid
    ids : array-like
        Event IDs
    distances : array-like
        Event distances (km)
    stations : array-like
        Station names
    phase : str
        Phase type: 'P', 'S', or 'PS'
    cfg : namespace
        Configuration with data.sampling_rate
    pick : bool, default=False
        Use true picks instead of predictions
    smask : bool, default=False
        Apply S-wave mask to P grid
    pvel : float, default=7.0
        P-wave velocity (km/s) for alignment
    svel : float, default=3.5
        S-wave velocity (km/s) for alignment
    redvel : float, default=9999.0
        Reduction velocity (km/s)
    
    Returns
    -------
    grid : numpy.ndarray
        P-wave prediction grid (distance_bins, time_samples)
    grid_s : numpy.ndarray
        S-wave prediction grid (distance_bins, time_samples)
    """
    grid = np.zeros((len(dist_bins),nsamples))
    grid_s = np.zeros((len(dist_bins),nsamples))
    counter = np.zeros(len(dist_bins))
    counter_s = np.zeros(len(dist_bins))
    p_counter = 0
    s_counter = 0
    npredictions = int(5*60*cfg.data.sampling_rate)
    for di,dist_bin in enumerate(dist_bins) :
        for eventid,dist,stat in zip(ids,distances,stations):
            eventid=eventid.decode("utf-8")+'_'+stat
            ptraveltime = dist / pvel - dist / redvel 
            straveltime = dist / svel - dist / redvel 
            if dist > dist_bin[0] and dist <= dist_bin[1]:
                try :
                    p_true = np.array(pred_dict[eventid]['y_p'])
                    s_true = np.array(pred_dict[eventid]['y_s'])
                    if not pick :
                        #_, p_pred, s_pred = np.split(np.array(pred_dict[eventid]['yhat']), 3, axis=-1)
                        p_pred = np.array(pred_dict[eventid]['yhat_p'])
                        s_pred = np.array(pred_dict[eventid]['yhat_s'])
                    else :
                        p_pred = p_true
                        s_pred = s_true
                    # if only one P pick
                    norm=27.572912
                    if phase == 'P' and int(0.1+np.sum(np.squeeze(p_true)/norm)) == 1 and np.sum(np.squeeze(s_true)) == 0 :
                        # due to SNR-based removal of arrivals, some single picks are not aligned
                        # these need extra shift
                        shift = int(ptraveltime*cfg.data.sampling_rate)
                        if np.argmax(np.squeeze(p_true)) != 6000 :
                            shift += 6000-np.argmax(np.squeeze(p_true))
                        p_counter += 1
                    # if only one S pick
                    elif (phase == 'S' and np.sum(np.squeeze(p_true)) == 0 and
                          int(0.1+np.sum(np.squeeze(s_true)/norm)) == 1) :
                        shift = int(straveltime*cfg.data.sampling_rate)
                        if np.argmax(np.squeeze(s_true)) != 6000 :
                            shift += 6000-np.argmax(np.squeeze(s_true))
                        s_counter += 1
                    # if one P and one S pick
                    elif ( phase =='PS' and int(0.1+np.sum(np.squeeze(p_true)/norm)) == 1
                            and int(0.1+np.sum(np.squeeze(s_true)/norm)) == 1):
                        start_first_phase = int((np.argmax(np.squeeze(s_true)) - np.argmax(np.squeeze(p_true)))/2)
                        shift = int(ptraveltime*cfg.data.sampling_rate) + start_first_phase
                    # ignore other cases for now
                    else : continue
                    pred_shifted = np.zeros(nsamples)
                    pred_shifted_s = np.zeros(nsamples)
                    if shift>0 and shift < nsamples and (shift+npredictions) < nsamples :
                        pred_shifted[shift:(shift+npredictions)] = np.squeeze(p_pred)
                        pred_shifted_s[shift:(shift+npredictions)] = np.squeeze(s_pred)
                    elif shift>0 and shift < nsamples and (shift+npredictions) > nsamples :
                        pred_shifted[shift:] = np.squeeze(p_pred)[:(nsamples-shift)]
                        pred_shifted_s[shift:] = np.squeeze(s_pred)[:(nsamples-shift)]
                    else :
                        pred_shifted = np.zeros(nsamples)
                        pred_shifted_s = np.zeros(nsamples)
                    grid[di] += pred_shifted
                    grid_s[di] += pred_shifted_s
                    counter_s[di] += 1
                    counter[di] += 1
                except KeyError:
                    continue
    # scale and mask grid
    for i,dist_bin in enumerate(dist_bins) :
        if counter[i]>0 :grid[i] /= counter[i]
        if counter_s[i]>0 : grid_s[i] /= counter_s[i]

        ptraveltime = dist_bin[0] / pvel - dist_bin[0] / redvel
        straveltime = dist_bin[0] / svel - dist_bin[0] / redvel
        shift = int(ptraveltime*cfg.data.sampling_rate)
        if shift < 0 : continue
        end = nsamples-npredictions-shift
        if end < 0 :  mask = np.concatenate((np.ones(shift) + 10.,np.zeros(nsamples-shift)))
        else : mask = np.concatenate((np.ones(shift) + 10.,np.zeros(npredictions),np.ones(end) + 10.))
        if not smask : grid[i] += mask
        if phase == 'S' :
            shift = int(straveltime*cfg.data.sampling_rate)
            if shift < 0 : continue
            end = nsamples-npredictions-shift
            if end < 0 :
                if nsamples-shift > 0 :  mask = np.concatenate((np.ones(shift) + 10.,np.zeros(nsamples-shift)))
                else : mask = np.ones(nsamples) + 10.
            else : mask = np.concatenate((np.ones(shift) + 10.,np.zeros(npredictions),np.ones(end) + 10.))
        grid_s[i] += mask
        if smask : grid[i] += mask

    return grid,grid_s


class EventWindowEval():
    """
    Container for event metadata and evaluation data.
    
    Stores event information including IDs, stations, distances, times,
    and provides methods for filtering events.
    
    Attributes
    ----------
    ids : array-like
        Event identifiers
    station : array-like
        Station codes
    distance : array-like
        Event-station distances (km)
    orig_time : array-like
        Event origin times
    baz : array-like
        Back-azimuths (degrees)
    arrival_id : array-like
        Arrival identifiers
    catalog : array-like
        Catalog/bulletin names
    first_arrival : array-like
        First arrival times
    last_arrival : array-like
        Last arrival times
    
    Methods
    -------
    remove_events(idx)
        Filter all attributes to specified indices
    """
    def __init__(self, ids = [], station = [], distance = [],
                 orig_time = [], baz = [], arrival_id = [] ,catalog = [],
                 last_arrival = [], first_arrival = []):
        self.ids = ids
        self.station = station
        self.distance = distance
        self.orig_time = orig_time
        self.baz = baz
        self.arrival_id = arrival_id
        self.catalog = catalog
        self.first_arrival = first_arrival
        self.last_arrival = last_arrival
    
    def remove_events(self, idx):
        """
        Filter all event attributes to specified indices.
        
        Parameters
        ----------
        idx : array-like
            Indices to keep
        
        Returns
        -------
        None
            Modifies object in-place
        """
        if len(self.ids)>0 : self.ids = self.ids[idx]
        if len(self.station)>0 : self.station = self.station[idx]
        if len(self.distance)>0 : self.distance = self.distance[idx]
        if len(self.orig_time)>0 : self.orig_time = self.orig_time[idx]
        if self.baz is not None and self.baz is not False and len(self.baz)>0 : self.baz = self.baz[idx]
        if len(self.arrival_id)>0 : self.arrival_id = self.arrival_id[idx]
        if len(self.catalog)>0 : self.catalog = self.catalog[idx]
        if len(self.first_arrival)>0 : self.first_arrival = self.first_arrival[idx]
        if len(self.last_arrival)>0 : self.last_arrival = self.last_arrival[idx]


def combine_split_heads_to_3channel(p_out, s_out, 
                                    noise_mode='min', 
                                    eps=1e-8,
                                    renormalize=True):
    """
    Combine two 2-channel heads ([noise,P], [noise,S]) into a
    single 3-channel array: [noise, P, S].

    Parameters
    ----------
    p_out : np.ndarray, shape=(B, T, 2)
        Output from the P-head, channels are [noise_p, p_pred].
    s_out : np.ndarray, shape=(B, T, 2)
        Output from the S-head, channels are [noise_s, s_pred].
    noise_mode : str
        How to combine the two noise channels.
        Options:
          - 'avg' => (noise_p + noise_s) / 2
          - 'min' => min(noise_p, noise_s)
          - 'prod' => noise_p * noise_s
          - 'max' => max(noise_p, noise_s)
          - 'none' => 0, and rely on 1 - P - S (see comments below).
    eps : float
        Small value to avoid division by zero if renormalizing.
    renormalize : bool
        If True, ensures that (noise + P + S) sums to 1 at each step.

    Returns
    -------
    combined : np.ndarray, shape=(B, T, 3)
        The unified array: combined[...,0] = noise,
                           combined[...,1] = P,
                           combined[...,2] = S.
    """
    # 1) Extract the pick channels
    p_prob = p_out[..., 1]  # Probability of P
    s_prob = s_out[..., 1]  # Probability of S

    # 2) Merge the noise channels depending on your chosen strategy
    noise_p = p_out[..., 0]
    noise_s = s_out[..., 0]

    if noise_mode == 'avg':
        noise = 0.5 * (noise_p + noise_s)
    elif noise_mode == 'min':
        noise = np.minimum(noise_p, noise_s)
    elif noise_mode == 'max':
        noise = np.maximum(noise_p, noise_s)
    elif noise_mode == 'prod':
        noise = noise_p * noise_s
    elif noise_mode == 'none':
        # Optionally set noise = 1 - p - s (post-hoc), but be careful
        noise = np.zeros_like(p_prob)
    else:
        raise ValueError(f"Unrecognized noise_mode='{noise_mode}'.")

    # 3) If you prefer "noise = 1 - P - S":
    #    just skip the average entirely, do:
    # noise = 1.0 - p_prob - s_prob
    #    # and possibly clip at zero
    # noise = np.clip(noise, 0.0, 1.0)

    # 4) Optionally re-normalize so noise + P + S sums to 1
    if renormalize:
        total = noise + p_prob + s_prob + eps
        noise /= total
        p_prob /= total
        s_prob /= total

    # 5) Stack channels to shape (B, T, 3)
    combined = np.stack([noise, p_prob, s_prob], axis=-1)
    return combined


# =============================================================================
# SECTION 2: PROCEDD PREDICTION UTILITIES
# =============================================================================
# Functions for processing predictions with phase detection models
# =============================================================================

def make_trace(station, label, prob, starttime, samp_rate):
    """
    Create ObsPy Trace object from prediction probabilities.
    
    Parameters
    ----------
    station : str
        Station name
    label : str
        Channel label ('P' or 'S')
    prob : numpy.ndarray
        Probability array
    starttime : obspy.UTCDateTime
        Start time
    samp_rate : float
        Sampling rate (Hz)
    
    Returns
    -------
    obspy.Trace
        Trace with probability data
    """
    tr = Trace()
    tr.stats.station = station
    tr.stats.channel = label
    tr.data = np.array(list(np.squeeze(prob)))
    tr.stats.sampling_rate = samp_rate
    tr.stats.starttime = starttime
    return tr


def voting(st, threshold=0.4):
    """
    Combine predictions by voting (fraction of stations detecting).
    
    Parameters
    ----------
    st : obspy.Stream
        Stream of prediction traces
    threshold : float, default=0.4
        Detection threshold for individual stations
    
    Returns
    -------
    obspy.Trace
        Combined trace with voting result
    """
    for tr in st:
        tr.data[tr.data < threshold] = 0.0
        tr.data[tr.data >= threshold] = 1.0
    data = np.sum(np.array([tr.data for tr in st]), axis=0) / len(st)
    st[0].data = data
    return st[0]


def get_max_beam(pred_stream, *, prefer='P_on_tie', return_meta=False):
    """
    Build a two-trace Stream (P, S) by collapsing multiple P- and S-traces
    column-wise (time-sample-wise) using a coupled max:

      For each sample j:
        - Let p_max = max over P-traces at j, with index i_p
        - Let s_max = max over S-traces at j, with index i_s
        - If P wins (according to `prefer`), set:
            outP[j] = p_max
            outS[j] = S[i_p, j]
          else S wins:
            outP[j] = P[i_s, j]
            outS[j] = s_max

    Parameters
    ----------
    pred_stream : obspy.Stream
        Must contain >=1 trace with channel='P' and >=1 trace with channel='S'.
    prefer : {'P_on_tie', 'S_on_tie'}, optional
        Tie-breaking rule when p_max == s_max.
    return_meta : bool, optional
        If True, also return (idx_used, p_wins) for inspection.

    Returns
    -------
    st : obspy.Stream
        Two traces: one P and one S, carrying the coupled outputs.
    (optional) idx_used : np.ndarray (n,)
        Row indices used per sample (i_p where P won, i_s where S won).
    (optional) p_wins : np.ndarray (n,) of bool
        True where P won, False where S won.
    """
    if not isinstance(pred_stream, Stream):
        raise TypeError("pred_stream must be an obspy.Stream")

    P_trs = pred_stream.select(channel='P')
    S_trs = pred_stream.select(channel='S')
    if len(P_trs) == 0 or len(S_trs) == 0:
        raise ValueError("Need at least one 'P' trace and one 'S' trace in pred_stream.")

    # Use the shortest length among all P and S traces to ensure alignment
    length = min(min(tr.stats.npts for tr in P_trs),
                 min(tr.stats.npts for tr in S_trs))

    # Stack to (n_traces, length): columns are time samples, axis=0 is across traces
    P_arr = np.vstack([tr.data[:length] for tr in P_trs])
    S_arr = np.vstack([tr.data[:length] for tr in S_trs])

    n = length
    cols = np.arange(n)

    # Column-wise argmax / max for P and S
    i_p = np.argmax(P_arr, axis=0)         # (n,)
    p_max = P_arr[i_p, cols]               # (n,)

    i_s = np.argmax(S_arr, axis=0)         # (n,)
    s_max = S_arr[i_s, cols]               # (n,)

    # Decide winner per column
    if prefer == 'P_on_tie':
        p_wins = p_max >= s_max
    elif prefer == 'S_on_tie':
        p_wins = p_max > s_max
    else:
        raise ValueError("prefer must be 'P_on_tie' or 'S_on_tie'")

    # Coupled outputs
    outP = np.empty(n, dtype=P_arr.dtype)
    outS = np.empty(n, dtype=S_arr.dtype)

    # Where P wins: take p_max and S at P's argmax row
    outP[p_wins] = p_max[p_wins]
    outS[p_wins] = S_arr[i_p[p_wins], cols[p_wins]]

    # Where S wins: take P at S's argmax row and s_max
    outP[~p_wins] = P_arr[i_s[~p_wins], cols[~p_wins]]
    outS[~p_wins] = s_max[~p_wins]

    # Build output stream using first P/S traces as templates (keeps metadata)
    st = Stream()
    p_template = P_trs[0].copy()
    s_template = S_trs[0].copy()
    p_template.data = outP
    s_template.data = outS

    st += p_template
    st += s_template

    if return_meta:
        idx_used = np.where(p_wins, i_p, i_s)
        return st, idx_used, p_wins
    return st


def combine_phase_detections(pred_stream,baz,cfg_pred,cfg,geometry=None,cont=False):
    """
    Combine phase detections from multiple array elements.
    
    Uses array processing techniques (stacking, voting, beamforming) to
    combine individual station predictions into array-level detections.
    
    Parameters
    ----------
    pred_stream : obspy.Stream
        Stream of prediction traces from array stations
    baz : float or None
        Back-azimuth (degrees) for beamforming, None for grid search
    cfg_pred : namespace
        Prediction configuration with:
        - combine_array_stations : Method string ('stack', 'vote', 'beam') or False
        - combine_beams : Use max beam combination
        - p_beam_vel, s_beam_vel : Beamforming velocities (float or list of floats)
        - azimuths : List of back-azimuths [deg] (optional, default 0-360 step 30)
        - vote_threshold_p, vote_threshold_s : Voting thresholds
        - start_time, end_time : Time range for trimming
    cfg : namespace
        Model configuration
    geometry : dict or None, default=None
        Station geometry for beamforming as {station_code: {'dx': km, 'dy': km}}.
        Required if combine_array_stations is 'beam'.
    cont : True if evaluation on contineous data
    
    Returns
    -------
    st_comb : obspy.Stream
        Combined prediction traces (P and S channels)
        Station name indicates combination method (STACK, VOTE, BEAM, MAXBEAM)
    """
    st_comb = Stream()

    if cfg_pred.skip_stations :
        for stat in cfg_pred.skip_stations :
            for tr in pred_stream.select(station=stat):
                print(f'Removing station {tr.stats.station}')
                pred_stream.remove(tr)  

    if cfg_pred.combine_beams :
        st_comb = get_max_beam(pred_stream)
        for tr in st_comb : tr.stats.station = 'MAXBEAM'
        st_comb = st_comb.trim(UTCDateTime(cfg_pred.start_time),UTCDateTime(cfg_pred.end_time))
        return st_comb

    comb = cfg_pred.combine_array_stations
    if comb == 'stack' or \
       (comb == 'beam' and \
       (len(pred_stream.select(channel='P')) < 2 or \
        len(pred_stream.select(channel='S')) < 2)) :

        st = pred_stream.copy().select(channel='P').stack()
        st += pred_stream.copy().select(channel='S').stack()
        for tr in st : tr.stats.station = 'STACK'
        st_comb += st
        bazlist=[None]
    elif comb == 'vote' :

        st = Stream()
        st += voting(pred_stream.copy().select(channel='P'),cfg_pred.vote_threshold_p)
        st += voting(pred_stream.copy().select(channel='S'),cfg_pred.vote_threshold_s)
        for tr in st : tr.stats.station = 'VOTE'
        st_comb += st

    elif comb == 'beam' :
        from beamforming import compute_beam_time_delays, create_beam
        if geometry is None:
            raise ValueError("geometry dict required for beamforming. "
                             "Provide station offsets as {station: {'dx': km, 'dy': km}}")
        # Get azimuth list: use provided baz, or config azimuths, or default grid
        if baz:
            bazlist = [baz]
        elif hasattr(cfg_pred, 'azimuths') and cfg_pred.azimuths:
            bazlist = list(cfg_pred.azimuths)
        else:
            bazlist = list(range(0, 360, 30))
        
        # Normalize velocities to lists (backwards compatible with single float)
        p_beam_vel = cfg_pred.p_beam_vel
        s_beam_vel = cfg_pred.s_beam_vel
        if isinstance(p_beam_vel, (int, float)):
            p_vel_list = [p_beam_vel]
        else:
            p_vel_list = list(p_beam_vel)
        if isinstance(s_beam_vel, (int, float)):
            s_vel_list = [s_beam_vel]
        else:
            s_vel_list = list(s_beam_vel)
        
        # force same start time
        for tr in pred_stream : tr.stats.starttime = pred_stream[0].stats.starttime
        st = Stream()
        # Create beams for all azimuth and velocity combinations
        for baz_val in bazlist :
            for p_vel in p_vel_list:
                time_delays = compute_beam_time_delays(geometry, azimuth_deg=baz_val, velocity_km_sec=p_vel)
                p_beam = create_beam(pred_stream.select(channel='P'), time_delays=time_delays, station_name='BEAM')
                st += p_beam
            for s_vel in s_vel_list:
                time_delays = compute_beam_time_delays(geometry, azimuth_deg=baz_val, velocity_km_sec=s_vel)
                s_beam = create_beam(pred_stream.select(channel='S'), time_delays=time_delays, station_name='BEAM')
                st += s_beam
            for tr in st :
                tr.data = tr.data[:len(pred_stream[0].data)]
        if len(bazlist)>1 or len(p_vel_list)>1 or len(s_vel_list)>1: 
            st_comb += get_max_beam(st)
        else : 
            st_comb += st

    else : 
        print("Not implemented")
        exit()

    # Trimming and alignment for continuous data
    if cont:
        # Initial trim to requested time range
        st_comb = st_comb.trim(UTCDateTime(cfg_pred.start_time), UTCDateTime(cfg_pred.end_time))

        # Ensure all traces have exactly the same start and end times
        if len(st_comb) > 0:
            # Find common time window (latest start, earliest end)
            common_start = max([tr.stats.starttime for tr in st_comb])
            common_end = min([tr.stats.endtime for tr in st_comb])

            # Trim all traces to common time window
            st_comb = st_comb.trim(common_start, common_end)

            # Verify all traces now have the same length
            lengths = [tr.stats.npts for tr in st_comb]
            if len(set(lengths)) > 1:
                print(f"WARNING: Traces have different lengths after alignment: {set(lengths)}")
                print(f"         Common window: {common_start} to {common_end}")
        st_comb.plot()

    return st_comb

def group_and_combine_phase_detections(p_pred,s_pred,ids,ids_stat,stations,cfg,cfg_pred,array_geometries,baz):
    """
    Group predictions by array and combine using array processing.
    
    For each event, groups predictions by array (ARCES, FINES, etc.),
    applies array combination methods, and returns combined predictions.
    
    Parameters
    ----------
    p_pred : numpy.ndarray
        P-wave predictions (events, samples)
    s_pred : numpy.ndarray
        S-wave predictions (events, samples)
    ids : array-like
        Event IDs (may include multiple arrays per event)
    ids_stat : array-like
        Event IDs with station suffixes (event_id + array code)
    stations : array-like
        Full station names
    cfg : namespace
        Model configuration with data.sampling_rate
    cfg_pred : namespace
        Prediction configuration with combine_array_stations settings
    array_geometries : dict
        Array geometries for beamforming as {array_name: {station: {'dx': km, 'dy': km}}}.
        Required if combine_array_stations is 'beam'.
        Example: {'ARCES': {'ARA0': {'dx': 0.0, 'dy': 0.0}, 'ARA1': {'dx': 0.5, 'dy': 0.3}}}
    baz : array-like
        Back-azimuths for each event

    
    Returns
    -------
    p_pred_new : numpy.ndarray
        Combined P-wave predictions
    s_pred_new : numpy.ndarray
        Combined S-wave predictions
    idx_new : list
        Indices of original events used (for metadata mapping)
    
    Notes
    -----
    Groups by arrays: ARCES (AR), FINES (FI), SPITS (SP), NORES (NR)
    """
    p_pred_new = []
    s_pred_new = []
    idx_new = []
    for evid in tqdm(sorted(list(set(ids)))) :
        for array in cfg_pred.stations :
            array_stations = cfg_pred.arrays[array] 
            idx = []
            for ar in array_stations :
                if ar[2] == '*':
                    idx += [i for i,evid_stat in enumerate(ids_stat)
                       if evid.decode("utf-8") == '_'.join((evid_stat.split('_')[0],evid_stat.split('_')[1],evid_stat.split('_')[2])) 
                              and evid_stat.split('_')[3][:2] == ar[:2] and i < len(p_pred)]
                else :
                    idx += [i for i,evid_stat in enumerate(ids_stat)
                       if evid.decode("utf-8") == '_'.join((evid_stat.split('_')[0],evid_stat.split('_')[1],evid_stat.split('_')[2]))  
                              and evid_stat.split('_')[3] == ar and i < len(p_pred)]
            if len(idx) == 0 : continue
            pred_stream = Stream()
            for i in idx :
                pred_stream += make_trace(stations[i],'P',np.squeeze(p_pred[i]),UTCDateTime(),cfg.data.sampling_rate)
                pred_stream += make_trace(stations[i],'S',np.squeeze(s_pred[i]),UTCDateTime(),cfg.data.sampling_rate)
            if cfg_pred.combine_array_stations == "beam" :
                geometry = array_geometries.get(array) if array_geometries else None
                st_comb=combine_phase_detections(pred_stream,baz[idx[0]],cfg_pred,cfg,geometry=geometry)
            else :
                st_comb=combine_phase_detections(pred_stream,False,cfg_pred,cfg)
            if len(st_comb.select(station=cfg_pred.combine_array_stations.upper()).select(channel='P')) == 0 : continue
            p_pred_new.append(st_comb.select(station=cfg_pred.combine_array_stations.upper()).select(channel='P')[0].data)
            s_pred_new.append(st_comb.select(station=cfg_pred.combine_array_stations.upper()).select(channel='S')[0].data)
            idx_new.append(idx[0]) # <- good if this is central element
    return np.array(p_pred_new),np.array(s_pred_new),idx_new
