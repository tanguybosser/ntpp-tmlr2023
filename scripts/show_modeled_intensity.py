
import numpy as np 
import os, sys
import torch as th 
from pathlib import Path
import json
import matplotlib.pyplot as plt 
sys.path.append(os.path.abspath(os.path.join('..', 'tmlr')))
from plots.acronyms import get_acronym, map_datasets_name
from typing import Dict, Tuple, Optional

from tpp.processes.multi_class_dataset import MultiClassDataset as Dataset
from tpp.utils.data import get_loader
from tpp.utils.events import get_events, get_window 
from tpp.models import get_model


from argparse import ArgumentParser, Namespace
from distutils.util import strtobool


def parse_args_intensity():
    parser = ArgumentParser(allow_abbrev=False)
    # Model dir
    parser.add_argument("--model", type=str, required=True, 
                        help="Path of the saved model checkpoint.")
    # Run configuration
    parser.add_argument("--seed", type=int, default=0, help="The random seed.")
    parser.add_argument("--padding-id", type=float, default=-1.,
                        help="The value used in the temporal sequences to "
                             "indicate a non-event.")
    # Common model hyperparameters
    parser.add_argument("--batch-size", type=int, default=1,
                        help="The batch size to use for parametric model"
                             " training and evaluation.")
    parser.add_argument("--time-scale", type=float, default=1.,
                        help='Time scale used to prevent overflow')
    parser.add_argument("--multi-labels",
                        type=lambda x: bool(strtobool(x)), default=False,
                        help="Whether the likelihood is computed on "
                             "multi-labels events or not")
    parser.add_argument("--window", type=int, default=100,
                        help="The window of the simulated process.py. Also "
                             "taken as the window of any parametric Hawkes "
                             "model if chosen.")
    # Dirs
    parser.add_argument("--load-from-dir", type=str, required=True,
                        help="If not None, load data from a directory")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory where model's checkpoint is stored.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="If not None, load data from a directory")
    parser.add_argument("--split", type=str, default='split_0',
                        help="If not None, load data from a directory")
    parser.add_argument("--plots-dir", type=str,
                        default="~/neural-tpps/plots",
                        help="Directory to save the plots")
    parser.add_argument("--save-fig-dir", type=str, default=None,
                        help="Directory to save the preprocessed data")
                
    #plot
    parser.add_argument("--seq-idx", type=int, default=1,
                        help="Sequence index on which to compute the intensity")
    parser.add_argument("--ground-intensity", default=True,
                        type=lambda x: bool(strtobool(x)),
                        help="If True, shows the ground intensity.")
    parser.add_argument("--x-lims", default=[0,10], nargs='+', 
                        type=float,
                        help="Limits on x-axis")                    
    args, _ = parser.parse_known_args()
    cwd = Path.cwd()
    data_args_path = os.path.join(args.load_from_dir, args.dataset)
    data_args_path = os.path.join(data_args_path, args.split)
    data_args_path = os.path.join(data_args_path, 'args.json')
    data_args_path = os.path.join(cwd, data_args_path)
    #model_path = os.path.join('checkpoints/model_selection/marked_filtered', args.dataset)
    #model_dir = os.path.join(model_path, 'best')
    
    model_dir = os.path.join(cwd, args.model_dir)
    with open(data_args_path, 'r') as fp:
        args_dict_json = json.load(fp)
    args_dict = vars(args)
    args_dict.update(args_dict_json)
    args_dict["model_dir"] = model_dir
    args = Namespace(**args_dict)
    return args

def update_args(args:Namespace, model_file:str) -> Tuple[Namespace, str]:
    cwd = Path.cwd()
    model_name = model_file.split('/')[-1][:-3]
    model_args_path = os.path.join(Path(model_file).parent, 'args')
    model_args_path = os.path.join(model_args_path, model_name + 'json')
    model_args_path = os.path.join(cwd, model_args_path)
    args_dict = vars(args)
    with open(model_args_path, 'r') as f:
        model_args_dic = json.load(f)
    args_dict.update(model_args_dic)
    args = Namespace(**args_dict)
    args.device = th.device('cpu')
    args.verbose = False
    args.batch_size = 1
    return args, model_name


def reformat(process:list) -> list:
    labelled_process = [[[time, label] for label, dimension in enumerate(seq) for time in dimension] for seq in process]
    sorted_process = [sorted(seq, key= lambda seq: seq[0]) for seq in labelled_process]
    dic_process = [[{'time':event[0], 'labels':[event[1]]} for event in seq] for seq in sorted_process]
    return dic_process 

def plot_modelled_intensity(times, intensity, axes, args, model_name):
    if args.ground_intensity is False:
        for i, ax in enumerate(axes):
            ax.plot(times, intensity[:,i], label=model_name, color='darkorange')
    else:
        axes.plot(times, intensity, label=model_name, color='darkorange')

def plot_modelled_density(times, density, axes, args, model_name):
    if args.ground_intensity is False:
        for i, ax in enumerate(axes):
            ax.plot(times, density[:,i], label=model_name, color='forestgreen')
    else:
        axes.plot(times, density, label=model_name, color='forestgreen')

def plot_process(multi_timestamps, ax):
    n_marks = len(multi_timestamps)
    for i, timestamps in enumerate(multi_timestamps):
        y_point_pos = np.array([0] * len(timestamps))
        if n_marks > 1:
            ax[i].scatter(timestamps, y_point_pos)
        else:
            ax.scatter(timestamps, y_point_pos)
    return ax

def reformat_timestamps(sequence):
    marks = np.unique([event['labels'][0] for event in sequence])
    timestamps = [[event['time'] for event in sequence if event['labels'][0] == mark] for mark in marks]
    return np.array(timestamps)

def plot_sequence_intensity(args:Namespace):
    dataset_path = os.path.join(args.load_from_dir, args.dataset)
    dataset_path = os.path.join(dataset_path, 'split_0/test.json') 
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)  
    sequence = dataset[args.seq_idx]
    if args.ground_intensity is False:
        timestamps = reformat_timestamps(sequence)
        n_marks = len(np.unique([event['labels'][0] for event in sequence]))
        fig, ax = plt.subplots(n_marks,1)
    else:
        timestamps =  [[event['time'] for event in sequence]]
        fig, ax = plt.subplots(1,1)
    sequence = [sequence]
    ax = plot_process(timestamps, ax)
    t_min, t_max = args.x_lims[0], args.x_lims[1]
    step = (t_max-t_min)/100000 
    all_models_files = os.listdir(args.model_dir)
    if 'base' in args.model:
        file_to_find = 'poisson_' + args.model.split('_base')[0] 
    else:
        file_to_find = args.model 
    model_file = None
    for file in all_models_files:
        if file.startswith(file_to_find):
            model_file = os.path.join(args.model_dir, file)
            break
    if model_file is None:
        raise ValueError('Checkpoint not found')
    args, model_name = update_args(args, model_file)
    model_name = model_file.split('/')[-1][:-3]
    data_sequence = Dataset(
    args=args, size=1, seed=args.seed, data=sequence)
    loader = get_loader(data_sequence, args=args, shuffle=False)
    model = get_model(args)
    model.load_state_dict(th.load(model_file, map_location=args.device))
    for batch in loader:
        times, labels = batch["times"], batch["labels"]
        labels = (labels != 0).type(labels.dtype) 
        mask = (times != args.padding_id).type(times.dtype)
        times = times * args.time_scale 
        window_start, window_end = get_window(times=times, window=args.window)
        events = get_events(
        times=times, mask=mask, labels=labels,
        window_start=window_start, window_end=window_end)
        query_times = th.arange(t_min, t_max, step=step, dtype=th.float32).unsqueeze(0)        
        ground_intensity, log_ground_intensity, ground_intensity_integral, _, _ = get_intensity(model, query_times, events) 
        ground_intensity_events, log_ground_intensity_events,ground_intensity_integral_events, _, _ = get_intensity(model, times, events, is_event=True)

        ground_density, log_ground_density = get_density(model, query_times, events) 
        ground_density_events, log_ground_density_events = get_density(model, times, events, is_event=True)
        log_ground_intensity_events_sum = np.sum(log_ground_intensity_events)
        log_ground_density_events_sum = np.sum(log_ground_density_events)

        print('TOTAL SEQ LOG-INTENSITY', log_ground_intensity_events_sum)
        print('TOTAL SEQ LOG-DENSITY', log_ground_density_events_sum)
        
        query_times = query_times.squeeze().cpu().numpy()
        plot_modelled_density(query_times, ground_density, ax, args, model_name)
        plot_modelled_intensity(query_times, ground_intensity, ax, args, model_name)
        
        title = fig_title(model_name, args.dataset)
        ax.set_title(title, fontsize=24)

        #ax.legend()
        ax.set_xlim(t_min, t_max)
        ax.grid(True)
        ax.set_ylabel(r'$\lambda^\ast(t)~\backslash~f(t|\mathcal{H}_t)$', fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=24 )
        ax.set_xticklabels('')
        ax.set_xlabel(r'$t$', fontsize=24) 
        if not os.path.exists(args.save_fig_dir):
            os.makedirs(args.save_fig_dir)
        fig_name = os.path.join(args.save_fig_dir, title.replace('/', '_'))
        fig.savefig(fig_name, bbox_inches='tight')
        print('DONE')
    

def get_intensity(model, query_times, events, poisson=False, is_event=False):
    log_intensity, marked_intensity_integral , intensity_mask, artifacts = model.artifacts(query=query_times, events=events)
    if is_event:
        log_intensity = log_intensity * intensity_mask.unsqueeze(-1)
    ground_intensity = th.sum(th.exp(log_intensity), dim=-1).squeeze(-1)
    log_ground_intensity = th.logsumexp(log_intensity, dim=-1).squeeze(-1)
    ground_intensity_integral = th.sum(marked_intensity_integral, dim=-1).squeeze(-1)
    if is_event:
        ground_intensity = ground_intensity * intensity_mask
        log_ground_intensity = log_ground_intensity * intensity_mask
        ground_intensity_integral = ground_intensity_integral * intensity_mask
    ground_intensity = ground_intensity.squeeze(0)
    log_ground_intensity = log_ground_intensity.squeeze()
    if poisson:
        log_intensity_0 = artifacts["log_intensity_0"]
        if is_event:
            log_intensity_0 = log_intensity_0 * intensity_mask.unsqueeze(-1)
        intensity_0 = th.sum(th.exp(log_intensity_0),dim=-1).squeeze(-1)
        if is_event:
            intensity_0 = intensity_0 * intensity_mask
        intensity_0 = intensity_0.squeeze()
        log_intensity_1 = artifacts["log_intensity_1"]
        if is_event:
            log_intensity_1 = log_intensity_1 * intensity_mask.unsqueeze(-1)
        intensity_1 = th.sum(th.exp(log_intensity_1),dim=-1).squeeze(-1)
        if is_event:
            intensity_1 = intensity_1 * intensity_mask
        intensity_1 = intensity_1.squeeze()
        alpha = th.tensor(artifacts["alpha"])     
        intensity_0 = alpha[1] * intensity_0 
        intensity_1 = alpha[0] * intensity_1 
        intensity_0 = intensity_0.detach().cpu().numpy()
        intensity_1 = intensity_1.detach().cpu().numpy()
    else:
        intensity_0, intensity_1 = None, None
    ground_intensity = ground_intensity.detach().cpu().numpy()
    log_ground_intensity = log_ground_intensity.detach().cpu().numpy()
    ground_intensity_integral = ground_intensity_integral.detach().cpu().numpy()
    artifacts = {}
    artifacts['intensity_0'] = intensity_0
    artifacts['intensity_1'] = intensity_1
    return ground_intensity, log_ground_intensity, ground_intensity_integral, intensity_mask, artifacts

def get_density(model, query_times, events, is_event=False):
    log_density, log_mark_pmf, density_mask = model.log_density(query=query_times, events=events)
    if is_event:
        log_density = log_density * density_mask.unsqueeze(-1)
    ground_density = th.sum(th.exp(log_density), dim=-1).squeeze(-1)
    log_ground_density = th.logsumexp(log_density, dim=-1).squeeze(-1)
    if is_event:
        ground_density = ground_density * density_mask
        log_ground_density = log_ground_density * density_mask
    ground_density = ground_density.squeeze(0)
    log_ground_density = log_ground_density.squeeze(0)
    ground_density = ground_density.detach().cpu().numpy()
    log_ground_density = log_ground_density.detach().cpu().numpy()
    return ground_density, log_ground_density

def fig_title(model_name:str, dataset:str) -> str:
    model_short = model_name.split(dataset + '_')[1].split('_split')[0]
    if 'lnmk1' in model_name:
        model_short += '_lnmk1'
    model_acr = get_acronym([model_short])[0]
    if model_acr != 'Poisson' and model_acr != 'Hawkes':
        model_acr_short = model_acr.split('-')[0] + '-' + model_acr.split('-')[1]
    else:
        model_acr_short = model_acr
    dataset_acr = map_datasets_name(dataset)
    title = dataset_acr + '-' + model_acr_short
    return title

if __name__ == "__main__":
    parsed_args = parse_args_intensity()
    plot_sequence_intensity(parsed_args)
