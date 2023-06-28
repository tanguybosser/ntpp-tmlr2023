from argparse import ArgumentParser, Namespace
import json
import os, sys
sys.path.remove(os.path.abspath(os.path.join('..', 'neuralTPPs')))
sys.path.append(os.path.abspath(os.path.join('..', 'tmlr')))

from pathlib import Path
from typing import Optional

from data_processing.simu_hawkes import simulate_process
from data_processing.prepare_datasets import make_splits


def parse_args():
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument("--kernel-name", type=str, required=True,
                        help="Type of Process to simulate.")
    parser.add_argument("--window", type=float, default=10,
                        help="End window of the simulated process.")
    parser.add_argument("--n-seq", type=int, default=1000,
                        help="Number of independent sequences to simulate")
    parser.add_argument("--marks", type=int, default=5,
                        help="Number of dimensions to simulate.")
    parser.add_argument("--self-decays", type=float, nargs="+", default=[],
                        help="Self decays of the Hawkes process.")
    parser.add_argument("--mutual-decays", type=float, default=1,
                        help="Mutual decays of the Hawkes process.")
    parser.add_argument("--baselines", type=float, nargs="+", default=[],
                        help="Baselines of the Hawkes process.")
    parser.add_argument("--self-adjacency", type=float, nargs="+", default=[],
                        help="Self adjacencies of the Hawkes process.")
    parser.add_argument("--mutual-adjacency", type=float, default=.5,
                        help="Mutual adjacencies of the Hawkes process.")
    parser.add_argument("--noise", type=float, default=None,
                        help="Upper bound of noise to add to the adjacency matrix. Only for Hawkes sum of exponentials.")
    parser.add_argument("--save-dir", type=str, required=True,
                        help="Directory in which to save the simulated dataset.")
    parser.add_argument("--num-splits", type=int, default=None,
                        help="Number of splits to make out of the simulated dataset. If None, no splits get created.")
    parser.add_argument("--seed", type=int, default=0)
    args, _ = parser.parse_known_args()
    
    return args


def reformat(process):
    labelled_process = [[[time, label] for label, dimension in enumerate(seq) for time in dimension] for seq in process]
    sorted_process = [sorted(seq, key= lambda seq: seq[0]) for seq in labelled_process]
    dic_process = [[{'time':event[0], 'labels':[event[1]]} for event in seq] for seq in sorted_process]
    return dic_process 



def simulate_dataset(args:Namespace):
    if 'independent' in args.kernel_name:
        assert(args.mutual_adjacency == 0.), 'Independent Hawkes must have mutual adjacency set to 0'
        assert(args.mutual_decays == 0.), 'Independent Hawkes must have mutual decays set to 0'
    if 'mutual' in args.kernel_name:
        assert(args.mutual_adjacency > 0.), 'Dependent Hawkes must have mutual adjacency different of 0'
        assert(args.mutual_decays > 0.), 'Dependent Hawkes must have mutual decays different of 0'        
    process, artifacts = simulate_process(kernel_name=args.kernel_name, window=args.window, 
                               n_seq=args.n_seq, marks=args.marks, self_decays=args.self_decays, mutual_decays=args.mutual_decays, 
                               baselines=args.baselines, self_adjacency=args.self_adjacency, mutual_adjacency=args.mutual_adjacency, track_intensity=False)
    process_reformat = reformat(process.timestamps)
    return process_reformat, artifacts    


def make_splits(process, 
                artifacts:dict, 
                args:Namespace, 
                split_idx:str, 
                train_prop:Optional[float]=0.6, 
                val_prop:Optional[float]=0.2):
    assert(train_prop + val_prop < 1)
    end_train_idx = int(len(process)*train_prop)
    end_val_idx = end_train_idx + int(len(process)*val_prop)
    train = process[0:end_train_idx]
    val = process[end_train_idx:end_val_idx]
    test = process[end_val_idx:]
    data_args = {'seed':args.seed, 'window':args.window, 'marks':args.marks, 'train_size':len(train), 'val_size':len(val), 'test_size':len(test),
            'decays':artifacts['decays'], 'baselines':args.baselines, 'adjacency':artifacts['adjacency'], 'kernel_name':args.kernel_name}
    save_dir = os.path.join(Path.cwd(), args.save_dir)
    save_dir = os.path.join(save_dir, args.kernel_name)
    save_dir = os.path.join(save_dir, 'split_{}'.format(split_idx))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)   
    save_train_path = os.path.join(save_dir, 'train.json')
    save_val_path = os.path.join(save_dir, 'val.json')
    save_test_path = os.path.join(save_dir, 'test.json')
    save_args_path = os.path.join(save_dir, 'args.json')
    paths = [save_train_path, save_val_path, save_test_path, save_args_path]
    data = [train, val, test, data_args]
    for i, path in enumerate(paths):
        with open(path, 'w') as f:
            json.dump(data[i], f)
    print('Successfully created split {} for simulated dataset'.format(split_idx))

if __name__ == "__main__":
    args = parse_args()
    print('Simulating {} dataset'.format(args.kernel_name))
    for split in range(args.num_splits):
        process, artifacts = simulate_dataset(args)
        make_splits(process, artifacts, args, split)
    print("Successfully saved simulated dataset")