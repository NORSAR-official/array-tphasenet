import numpy as np
import matplotlib.pyplot as plt
import argparse
from omegaconf import OmegaConf
from setup_config import get_config_dir,dict_to_namespace
from utils import (
    calculate_residual, js_divergence, recall, precision, f1_score, missed_detections,
    plot_L_curve, make_trace
)
from contineous_prediction_utils import (
    load_manual_picks,
    load_cont_detections,
    save_picks
)
from collections import defaultdict
from obspy import UTCDateTime, Stream
import warnings
import glob
import os

warnings.filterwarnings("ignore")

plt.rcParams.update({'font.size': 15})

def add_ref_to_steam(pred_stream,stat,components,times,cfg):
    # adding reference for keeping same scale 0->1 in snuffler
    ref = np.ones(len(pred_stream[0].data))
    ref[::2] = np.zeros(len(ref[::2]))
    for comp in components :
        pred_stream += make_trace(stat,comp,ref,times[0],cfg.data.sampling_rate)
    return pred_stream

if __name__ == '__main__':

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Evaluate continuous phase detection predictions.',
        epilog='Example: python evaluate_contineous.py --config config.yaml'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Main configuration file (contains model, data, and prediction settings)'
    )
    cmd_args = parser.parse_args()

    print('Reading config ...')
    config_dir = get_config_dir()
    # Load main configuration file (model + prediction settings)
    args = OmegaConf.load(f'{config_dir}/{cmd_args.config}')
    args_dict = OmegaConf.to_container(args, resolve=True)
    args = OmegaConf.create(args_dict)
    OmegaConf.set_struct(args, False)
    cfg = dict_to_namespace(args)
    
    # Extract prediction settings from main config
    cfg_pred = cfg.prediction
    
    print('Config read.')

    #model = 'risinghope_transphasenet'
    if not hasattr(cfg.data, "extract_array_channels"): setattr(cfg.data, "extract_array_channels", False)
    if not hasattr(cfg.run, "custom_outname"): setattr(cfg.run, "custom_outname", False)
    if cfg.data.extract_array_channels :
        model = f'array{cfg.data.setname}_{cfg.model.type}'
    else :
        model = f'{cfg.data.input_dataset_name}_{cfg.model.type}'
    if cfg.run.custom_outname :
        model = f'{cfg.data.input_dataset_name}_{cfg.run.custom_outname}_{cfg.model.type}'
    if cfg.run.custom_outname and cfg.data.extract_array_channels :
        model = f'array{cfg.data.setname}_{cfg.run.custom_outname}_{cfg.model.type}'

    dt = cfg.evaluation.dt_cont
    batchlen = int(3600*cfg.data.sampling_rate)
    stationstr = cfg_pred.stations[0]

    if not cfg_pred.combine_array_stations and not cfg_pred.combine_beams :
        print("Evaluate station by station")
        print("Reading detection output ...")
        pred_stream,_ = load_cont_detections(cfg_pred,cfg)
        p_pred = pred_stream[0].data
        times = pred_stream[0].times("utcdatetime")
        times_split=[times]

        # adding reference for keeping same scale 0->1 in snuffler
        pred_stream =  add_ref_to_steam(pred_stream,'X',['P','S'],times,cfg)
        # used for snuffler:
        if cfg_pred.save_prob : pred_stream.write(f'{cfg_pred.output_dir}/predictions_{model}_{stationstr}.msd',format='MSEED')

        print("Reading hand-picked arrivals ...")
        p_true,s_true,ap_true,as_true = load_manual_picks(p_pred.shape,p_pred.shape,times,cfg.evaluation.picks)
        p_true = np.array_split(p_true[:int(len(p_true)/batchlen)*batchlen],int(len(p_true)/batchlen))
        s_true = np.array_split(s_true[:int(len(s_true)/batchlen)*batchlen],int(len(s_true)/batchlen))
        ap_true = np.array_split(ap_true[:int(len(ap_true)/batchlen)*batchlen],int(len(ap_true)/batchlen))
        as_true = np.array_split(as_true[:int(len(as_true)/batchlen)*batchlen],int(len(as_true)/batchlen))
        for stat in list(set([tr.stats.station for tr in pred_stream])) :
            if stat != 'X' and stat in cfg_pred.stations :
                last_stat = stat
                p_pred = pred_stream.select(station=stat).select(channel='P')[0].data
                s_pred = pred_stream.select(station=stat).select(channel='S')[0].data
                p_pred = np.array_split(p_pred[:int(len(p_pred)/batchlen)*batchlen],int(len(p_pred)/batchlen))
                s_pred = np.array_split(s_pred[:int(len(s_pred)/batchlen)*batchlen],int(len(s_pred)/batchlen))
                if cfg.evaluation.optimal_threshold :
                    print("Generating recall-precision curves for getting optimal threshold")
                    thr_opt_p,thr_opt_s,fig = plot_L_curve(p_true, p_pred, s_true, s_pred,
                                           model, cfg, True, stat , useforprec=(ap_true,as_true))
                    fig.show()
                else :
                    thr_opt_p=cfg.evaluation.p_threshold
                    thr_opt_s=cfg.evaluation.s_threshold
                if cfg.evaluation.overall_performance :
                    text=f'Thresholds: {thr_opt_p} {thr_opt_s}\n'
                    print(text)

    else :
        print("Evaluate combined array output")
        print("Reading detection output ...")
        if cfg_pred.combine_array_stations :
            if len(cfg_pred.stations[0])>1:
                print(f'No array identifier found. Using station list: {cfg_pred.stations} for combined output file name')
                stationstr = "-".join(cfg_pred.stations)
            dfiles = glob.glob(f'{cfg_pred.output_dir}/{model}_{stationstr}_{cfg_pred.stacking}_combined_{cfg_pred.combine_array_stations}*.npz')
            comb_method = cfg_pred.combine_array_stations
            comb_method_org = comb_method
        if cfg_pred.combine_beams :
            comb_method_org = 'maxbeam'
            comb_method = 'maxbeam'
            dfiles = glob.glob(f'{cfg_pred.output_dir}/{model}_{stationstr}_{comb_method}*.npz')
        times_split=[]
        for i,dfile in enumerate(dfiles) :
            pred = np.load(dfile,allow_pickle=True)
            times = np.array([UTCDateTime(t) for t in pred['t']])
            print("Reading hand-picked arrivals ...")
            shap = np.transpose(pred['y'])[0].shape
            p_true,s_true,ap_true,as_true = load_manual_picks(shap,shap,times,cfg.evaluation.picks)
            idx_p=list(pred['label']).index(comb_method_org.upper()+'_P')
            idx_s=list(pred['label']).index(comb_method_org.upper()+'_S')
            pred = np.transpose(pred['y'])
            p_pred = pred[idx_p]
            s_pred = pred[idx_s]
            # save combined output as mseed for evaluation
            if cfg_pred.save_prob and ( cfg_pred.combine_array_stations or cfg_pred.combine_beams ) :
                pred_stream = Stream()
                station = comb_method_org
                if cfg_pred.combine_array_stations :
                    # stack can be only from single station as a work-around for time periods with long gaps (FINES)
                    if cfg_pred.stations[0] not in ['ARCES','FINES','SPITS','NORES'] : station = cfg_pred.stations[0]
                pred_stream += make_trace(station,'P',p_pred,times[0],cfg.data.sampling_rate)
                pred_stream += make_trace(station,'S',s_pred,times[0],cfg.data.sampling_rate)
                pred_stream = add_ref_to_steam(pred_stream,'X',['P','S'],times,cfg)
                if len(dfiles)==1 : pred_stream.write(f'{cfg_pred.output_dir}/predictions_{model}_{comb_method}_{stationstr}.msd',format='MSEED')
                else: pred_stream.write(f'{cfg_pred.output_dir}/predictions_{model}_{comb_method}_{stationstr}-{i}.msd',format='MSEED')

            if i == 0 :
                p_pred_tmp = p_pred.copy()
                s_pred_tmp = s_pred.copy()
                p_pred_tmp = p_pred.copy()
                s_pred_tmp = s_pred.copy()
                p_true_tmp = p_true.copy()
                s_true_tmp = s_true.copy()
                ap_true_tmp = ap_true.copy()
                as_true_tmp = as_true.copy()
                times_tmp = times.copy()
            else :
                p_pred_tmp = np.concatenate((p_pred_tmp,p_pred))
                s_pred_tmp = np.concatenate((s_pred_tmp,s_pred))
                p_true_tmp= np.concatenate((p_true_tmp,p_true))
                s_true_tmp = np.concatenate((s_true_tmp,s_true))
                ap_true_tmp = np.concatenate((ap_true_tmp,ap_true))
                as_true_tmp = np.concatenate((as_true_tmp,as_true))
                times_tmp = np.concatenate((times_tmp,times))
            times_split.append(times)
        p_pred = np.array_split(p_pred_tmp[:int(len(p_pred_tmp)/batchlen)*batchlen],int(len(p_pred_tmp)/batchlen))
        s_pred = np.array_split(s_pred_tmp[:int(len(s_pred_tmp)/batchlen)*batchlen],int(len(s_pred_tmp)/batchlen))
        p_true = np.array_split(p_true_tmp[:int(len(p_true_tmp)/batchlen)*batchlen],int(len(p_true_tmp)/batchlen))
        s_true = np.array_split(s_true_tmp[:int(len(s_true_tmp)/batchlen)*batchlen],int(len(s_true_tmp)/batchlen))
        ap_true = np.array_split(ap_true_tmp[:int(len(ap_true_tmp)/batchlen)*batchlen],int(len(ap_true_tmp)/batchlen))
        as_true = np.array_split(as_true_tmp[:int(len(as_true_tmp)/batchlen)*batchlen],int(len(as_true_tmp)/batchlen))
        times = times_tmp.copy()

        if cfg.evaluation.optimal_threshold :
            suf = f'{comb_method}'
            suf += '_'+stationstr
            print("Plotting recall-precision curves for optimal thresholds")
            thr_opt_p,thr_opt_s,fig = plot_L_curve(p_true, p_pred, s_true, s_pred,
                                       model, cfg, True, suf , useforprec=(ap_true,as_true))
            #fig.show()
        else :
            thr_opt_p=cfg.evaluation.p_threshold
            thr_opt_s=cfg.evaluation.s_threshold
        if cfg.evaluation.overall_performance :
            text=f'Thresholds: {comb_method} {thr_opt_p} {thr_opt_s}\n'
            print(text)


        if cfg.evaluation.optimal_threshold :
            plt.legend(['P MLArray','S MLArray','P FKX','S FKX'],fontsize=15,loc='lower left')

    if cfg_pred.combine_array_stations or cfg_pred.combine_beams :
        suffix = '_'+comb_method
        suffix += '_'+stationstr
    else : suffix = '_'+last_stat
    if cfg.evaluation.save_fig and cfg.evaluation.optimal_threshold :
        plt.savefig(f'{cfg_pred.output_dir}/L_curve_{model}{suffix}_cont.png')
    if cfg.evaluation.optimal_threshold : plt.show()

    # only last station or last combination method in case of single station picker ...
    if cfg.evaluation.overall_performance :
        #text=f'P samples {np.count_nonzero(p_true)}\n'
        text+=f'P Precision ({dt}s) {round(precision(p_true,p_pred,cfg,dt=dt,th=thr_opt_p, livemode=True), 2)}\n'
        text+=f'P Precision all P ({dt}s) {round(precision(ap_true,p_pred,cfg,dt=dt,th=thr_opt_p, livemode=True), 2)}\n'
        text+=f'P Recall ({dt}s) {round(recall(p_true, p_pred, cfg, dt=dt, th=thr_opt_p, livemode=True), 2)}\n'
        text+=f'P F1 ({dt}s) {round(f1_score(p_true, p_pred, cfg, dt=dt, th=thr_opt_p, livemode=True), 2)}\n'
        text+='\n'
        #text+=f'S samples {np.count_nonzero(s_true)}\n'
        text+=f'S Precision ({dt}s) {round(precision(s_true,s_pred,cfg,dt=dt,th=thr_opt_s, livemode=True), 2)}\n'
        text+=f'S Precision all S ({dt}s) {round(precision(as_true,s_pred,cfg,dt=dt,th=thr_opt_s, livemode=True), 2)}\n'
        text+=f'S Recall ({dt}s) {round(recall(s_true, s_pred,cfg,dt=dt,th=thr_opt_s, livemode=True), 2)}\n'
        text+=f'S F1 ({dt}s) {round(f1_score(s_true, s_pred,cfg,dt=dt,th=thr_opt_s, livemode=True), 2)}\n'

        text+=f'Ar Precision ({dt}s) {round(precision(np.add(p_true,s_true),np.add(p_pred,s_pred),cfg,dt=dt,th=thr_opt_s, livemode=True), 2)}\n'
        text+=f'Ar Recall ({dt}s) {round(recall(np.add(p_true,s_true),np.add(p_pred,s_pred),cfg,dt=dt,th=thr_opt_s, livemode=True), 2)}\n'
        text+=f'Ar F1 ({dt}s) {round(f1_score(np.add(p_true,s_true),np.add(p_pred,s_pred),cfg,dt=dt,th=thr_opt_s, livemode=True), 2)}\n'

        print(text)
        suffix += '_cont'

        fout=open(f'{cfg_pred.output_dir}/performance_{model}{suffix}.txt','w')
        fout.writelines(text)
        fout.close()

    # only last station or last combination method in case of single station picker ...
    if cfg_pred.save_picks :
        suffix += f'_{thr_opt_p}_{thr_opt_s}'
        outdir = f'{cfg_pred.output_dir}/{model}{suffix}'
        os.makedirs(outdir, exist_ok=True)
        batchlen = int(3600*cfg.data.sampling_rate)
        times = np.array_split(times[:int(len(times)/batchlen)*batchlen],int(len(times)/batchlen))
        save_picks(p_pred,s_pred,times,cfg.data.sampling_rate,thr_opt_p,thr_opt_s,stationstr,outdir)
