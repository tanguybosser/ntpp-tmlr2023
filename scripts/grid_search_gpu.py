import pickle 
import subprocess
import numpy as np
import os 
import random
from argparse import ArgumentParser, Namespace
from typing import Dict, Tuple, Optional

import sys 
#sys.path.remove(os.path.abspath(os.path.join('..', 'neuralTPPs')))
sys.path.append(os.path.abspath(os.path.join('..', 'tmlr')))

from distutils.util import strtobool
import time

from tpp.utils.commands import get_command


parser = ArgumentParser(allow_abbrev=False)
#Model parameters
parser.add_argument("--encoding", type=str, default='times_only', help="type of event encoding")
parser.add_argument("--encoder", type=str, help="encoder type")
parser.add_argument("--decoder", type=str, help="decoder type")
parser.add_argument("--val-emb-dim", type=int, default=[4], nargs="+", metavar='N', help="Dimension of the event encoding")
parser.add_argument("--val-encoder-units-rnn", type=int, default=[], nargs="+", metavar='N', help="Hidden RNN/SA encoder units")
parser.add_argument("--val-encoder-layers-rnn", type=int, default=[], nargs="+", metavar='N', help="Num. layers of RNN/SA encoder")
parser.add_argument("--val-units-mlp", type=int, default=[], nargs="+", metavar='N', help="Hidden MLP units")
parser.add_argument("--val-decoder-n-mixture", type=int, default=[], nargs="+", metavar='N', help="Num. mixtures for LogNormMix")
parser.add_argument("--decoder-mc-prop-est", type=int, default=10, help="Num. of Monte Carlo samples when estimating the Cum. MCIF")
parser.add_argument("--val-decoder-layers-rnn", default=[], nargs="+", metavar='N', help="Num. layers of SA decoders")
parser.add_argument("--val-decoder-units-rnn", default=[], nargs="+", metavar='N', help="Hidden units SA decoders")
parser.add_argument("--val-n-heads", default=[], nargs="+", metavar='N', help="Num. heads SA encoders/decoders")
parser.add_argument("--include-poisson", default=False, type=lambda x: bool(strtobool(x)), 
                    help='Whether or not to include a Poisson term')
parser.add_argument("--use-coefficients", default=True, type=lambda x: bool(strtobool(x)),
                    help="If True, modular process will be trained with coefficients. Only if include-poisson is True")
parser.add_argument("--coefficients", default=[], type=float, nargs='+',
                    help="Fix the intensity coefficients if a Poisson term is included. Only if use-coefficients is True.")
parser.add_argument("--use-softmax", default=True, type=lambda x: strtobool(x), 
                    help="If True, applies softmax to intensity coefficients.")

#Dataset parameters 
parser.add_argument("--datasets", default=[], nargs="+", help="Datasets on which to train")
parser.add_argument("--batch-size", default=[], nargs="+", metavar='N')
parser.add_argument("--num-config", type=int, default=5, help="Num. of configurations to try in the random search")
parser.add_argument("--splits", default=[1,2,3,4,5], nargs="+", type=int, help="Num. of dataset splits on which to train")

#Directories
parser.add_argument("--load-from-dir", default='data/baseline3', type=str,
                    help="Directory from where to load data")
parser.add_argument("--save-results-dir", default='results/model_selection/marked_filtered', type=str,
                    help="Directory where to store the results")
parser.add_argument("--save-checkpoints-dir", default='checkpoints/model_selection/marked_filtered', type=str,
                    help="Directory where to store model checkpoints")
parser.add_argument("--exp-name", default=None, type=str, 
                    help="Additional name to give the experiment. If None experiment will be saved under dataset + encoding + encoder + decoder")

#Training parameters 
parser.add_argument("--train-epochs", default=501, type=int, 
                    help="Num. of epochs to train the model")
parser.add_argument("--lr", default=0.001, type=float, 
                    help="Learning rate")

def random_search(param_dic:dict, num_config:Optional[int] = 5) -> dict:
    if len(param_dic) == 0:
        return [dict()]
    configs= []
    for _ in range(num_config):
        config = dict.fromkeys(param_dic.keys())
        while True:
            for param, values in param_dic.items():
                j = random.randint(0, len(values)-1)
                config[param] = values[j]
            if config not in configs:
                configs.append(config)  
                break
    return configs


def get_param_dic(args:Namespace) -> dict:
    param_dic = {}
    encoder_dic = {}
    decoder_dic = {}
    if args.encoder == 'gru':
        encoder_dic = {'encoder_emb_dim':args.val_emb_dim,
                    'encoder_units_rnn':args.val_encoder_units_rnn,
                    'encoder_layers_rnn':args.val_encoder_layers_rnn,
                    'encoder_units_mlp':args.val_units_mlp
                    }
    if args.encoder == 'selfattention':
        encoder_dic = {'encoder_emb_dim':args.val_emb_dim,
                    'encoder_units_rnn':args.val_encoder_units_rnn,
                    'encoder_layers_rnn':args.val_encoder_layers_rnn,
                    'encoder_n_heads':args.val_n_heads,
                    'encoder_units_mlp':args.val_units_mlp
                    }
    if args.decoder in ['conditional-poisson', 'mlp-cm', 'mlp-mc', 'rmtpp']:
        decoder_dic = {
                'decoder_units_mlp':args.val_units_mlp,
                'decoder_emb_dim':args.val_emb_dim
                    }
    if args.decoder == 'log-normal-mixture':
        decoder_dic = {
                'decoder_units_mlp':args.val_units_mlp,
                'decoder_n_mixture':args.val_decoder_n_mixture
                    }
    if args.decoder == 'selfattention-cm' or args.decoder == 'selfattention-mc':
        decoder_dic = {
                'decoder_units_mlp':args.val_units_mlp,
                'decoder_layers_rnn':args.val_decoder_layers_rnn,
                'decoder_units_rnn': args.val_decoder_units_rnn,
                'decoder_n_heads': args.val_n_heads, 
                'decoder_emb_dim':args.val_emb_dim
                }
    if args.decoder == 'neural-hawkes':
        decoder_dic = {
                'decoder_units_mlp':args.val_units_mlp,
                'decoder_layers_rnn':args.val_decoder_layers_rnn,
                'decoder_units_rnn': args.val_decoder_units_rnn
                }

    param_dic.update(encoder_dic)
    param_dic.update(decoder_dic)
    return param_dic

def make_dir(dir_path:str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

if __name__ == '__main__':

    t_start = time.time()
    args = parser.parse_args()
    args.coefficients = [None, None] if len(args.coefficients) == 0 else args.coefficients
    param_dic = get_param_dic(args)
    for i, dataset in enumerate(args.datasets):
        for split in args.splits:
            split = split-1
            save_results_dir = os.path.join(args.save_results_dir, dataset)
            save_checkpoints_dir = os.path.join(args.save_checkpoints_dir, dataset)
            save_results_dir_can = os.path.join(save_results_dir, 'candidates')
            save_checkpoints_dir_can = os.path.join(save_checkpoints_dir, 'candidates')
            save_results_dir_best = os.path.join(save_results_dir, 'best')
            save_checkpoints_dir_best = os.path.join(save_checkpoints_dir, 'best')
            make_dir(save_results_dir_can)
            make_dir(save_results_dir_best)
            make_dir(save_checkpoints_dir_can)
            make_dir(save_checkpoints_dir_best)
            configs = random_search(param_dic, num_config=args.num_config)
            config_val_losses = []
            config_file_res_paths, config_file_check_paths = [], []
            for j, config in enumerate(configs):
                print('\n\nDATASET: {}, CONFIG : {}/{}, SPLIT : {}/{}\n\n'.format(dataset, j+1, len(configs), split+1, len(args.splits)))
                print('MODEL : {}'.format(args.encoding + '_' + args.encoder + '_' + args.decoder))
                cmd = get_command(dataset, save_results_dir_can, args.batch_size[i], split, j, args, eval_metrics=True, config=config, 
                save_check_dir=save_checkpoints_dir_can)
                while True:
                    try:  
                        out = subprocess.run([cmd], shell=True, check=True)
                        break
                    except subprocess.CalledProcessError as e:
                            print('error occured')
                            break 
                file_name = dataset + '_' + args.encoder + '_' + args.decoder + '_' + args.encoding + '_split' + str(split)
                if args.exp_name is not None:
                    file_name = file_name + '_' + args.exp_name
                file_name = file_name + '_config' + str(j)
                if args.include_poisson:
                    if args.coefficients[0] is not None and args.coefficients[1] is not None:
                        file_name = 'poisson_coefficients_' + file_name
                    else:
                        file_name = 'poisson_' + file_name
                file_path_res_can = os.path.join(save_results_dir_can, file_name + '.txt')
                file_path_check_can = os.path.join(save_checkpoints_dir_can, file_name + '.pth')
                config_file_res_paths.append(file_path_res_can)
                config_file_check_paths.append(file_path_check_can)
                with open(file_path_res_can, 'rb') as fp:
                    while True:
                        try:
                            e = pickle.load(fp)
                            loss = e['val'][-1]['loss']
                            config_val_losses.append(loss)
                        except EOFError:
                            break             
            best_config_idx = np.argmin(config_val_losses)
            best_config = configs[best_config_idx]
            best_file_res_path = config_file_res_paths[best_config_idx]
            best_file_check_path = config_file_check_paths[best_config_idx]
            print('BEST CONFIGURATION FOR SPLIT {} : {}'.format(split, best_config))
            save_args_dir_best = os.path.join(save_checkpoints_dir_best, 'args')
            make_dir(save_args_dir_best)
            best_file_args_name = os.path.basename(best_file_check_path).rstrip('pth') + 'json'
            can_args_dir = os.path.join(save_checkpoints_dir_can, 'args')
            best_file_args_path = os.path.join(can_args_dir, best_file_args_name)
            while True:
                    try:                    
                        cmd_1 = "mv {} {}".format(best_file_res_path, save_results_dir_best)
                        cmd_2 = "mv {} {}".format(best_file_check_path, save_checkpoints_dir_best)
                        cmd_3 = "mv {} {}".format(best_file_args_path, save_args_dir_best)
                        out = subprocess.run([cmd_1], shell=True, check=True)
                        out = subprocess.run([cmd_2], shell=True, check=True)
                        out = subprocess.run([cmd_3], shell=True, check=True)
                        break
                    except subprocess.CalledProcessError as e:
                            break
            print('GRID SEARCH DONE')
    print('\nRUN FINISHED\n')
    delta_t = time.time() - t_start
    hours, mins, sec = int(delta_t/3600), int((delta_t%3600)/60), int((delta_t%3600)%60)
    print("Total run time : {}:{}:{}".format(hours,mins,sec))


