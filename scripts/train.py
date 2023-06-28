from ast import arg
from calendar import c
from cmath import exp, log
from logging.config import valid_ident
from unittest import result
import sys, os
#sys.path.remove(os.path.abspath(os.path.join('..', 'neuralTPPs')))
sys.path.append(os.path.abspath(os.path.join('..', 'tmlr')))
import pdb
import json
import numpy as np
import os
import stat
import time
import torchvision  
import pickle as pkl
import datetime 

import torch as th
from torch.optim import Adam
from torch.utils.data import DataLoader

from argparse import Namespace
from copy import deepcopy
from typing import Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from tpp.utils.events import get_events, get_window

from tpp.models import get_model
from tpp.models.base.process import Process
from tpp.utils.cli import parse_args
from tpp.utils.metrics import eval_metrics
from tpp.utils.data import get_loader, load_data
from tpp.utils.logging import get_status
from tpp.utils.lr_scheduler import create_lr_scheduler
from tpp.utils.run import make_deterministic
from tpp.utils.stability import check_tensor

print(th.cuda.is_available(), flush=True)

#torchvision.__version__ = '0.4.0'

def get_loss(
        model: Process,
        batch: Dict[str, th.Tensor],
        args: Namespace,
        eval_metrics: Optional[bool] = False,
        dynamic_batch_length: Optional[bool] = True,
        test: Optional[bool] = False
) -> Tuple[th.Tensor, th.Tensor, Dict]:
    """Returns the batch loss (NLL) of a model.

    Args:
        model: Model to be trained.
        batch: The batch on which to evaluate the loss. 
        args: Arguments for training.
        eval_metrics: If True, the predicted marks are stored in artifacts. 
        dynamic_batch_lenght: To be set to True if sequences are padded. 
        test: If False, only the loss (and eval_metrics) are returned. 
    Returns:
       loss, loss mask, and any desired quantity in artifacts.

    """
    times, labels = batch["times"], batch["labels"]
    labels = (labels != 0).type(labels.dtype) 
    if dynamic_batch_length:
        seq_lens = batch["seq_lens"]
        max_seq_len = seq_lens.max()
        times, labels = times[:, :max_seq_len], labels[:, :max_seq_len] 
    mask = (times != args.padding_id).type(times.dtype)
    times = times * args.time_scale 
    window_start, window_end = get_window(times=times, window=args.window) 
    events = get_events(
        times=times, mask=mask, labels=labels,
        window_start=window_start, window_end=window_end)
    loss, loss_mask, artifacts = model.neg_log_likelihood(events=events, test=test)  # [B]
    if eval_metrics:
        events_times = events.get_times()
        log_p, log_mark_density, y_pred_mask = model.log_density(
            query=events_times, events=events)  # [B,L,M], [B,L]
        proba = th.exp(log_mark_density)
        if args.multi_labels:
            y_pred = log_p  # [B,L,M]
            labels = events.labels
        else:
            y_pred = log_p.argmax(-1).type(log_p.dtype)  # [B,L]
            labels = events.labels.argmax(-1).type(events.labels.dtype)
            max_proba  = proba.max(-1).values.type(log_p.dtype)
        artifacts['y_pred'] = y_pred
        artifacts['y_true'] = labels
        artifacts['y_pred_mask'] = y_pred_mask
        artifacts['max proba'] = max_proba

    return loss, loss_mask, artifacts


def detach(x: th.Tensor):
    return x.cpu().detach().numpy()

def count_parameters(model: Process):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate(model: Process, args: Namespace, loader: DataLoader, test: Optional[bool] = False
             ) -> Dict[str, float]:
    """Evaluate a model on a specific dataset.

    Args:
        model: The model to evaluate.
        args: Arguments for evaluation
        loader: The loader corresponding to the dataset to evaluate on.
        test: If True, artifacts contains additional quantities evaluated at test time. 

    Returns:
        Dictionary containing all metrics evaluated and averaged over total
            sequences.

    """
    model.eval()

    t0, epoch_loss, epoch_loss_per_time, n_seqs = time.time(), 0., 0., 0.
    pred_labels, gold_labels, mask_labels, probas = [], [], [], []
    results = {}
    epoch_ground_density_loss, epoch_marks_loss, epoch_window_loss = 0, 0, 0
    epoch_intensity_0, epoch_intensity_1, alpha = 0, 0, 0
    epoch_mu, epoch_sigma, epoch_weight = [], [], []
    cumulative_density = []
    num_batch = 0
    for batch in tqdm(loader) if args.verbose else loader:
        batch['times'], batch['labels'], batch['seq_lens'] = batch['times'].to(args.device), batch['labels'].to(args.device), batch['seq_lens'].to(args.device)
        loss, loss_mask, artifacts = get_loss(  # [B]
            model, batch=batch, eval_metrics=args.eval_metrics, args=args,
            dynamic_batch_length=False, test=test) 
        loss = loss * loss_mask  # [B]
        epoch_loss += detach(th.sum(loss))
        ground_density_loss = artifacts["true log density"] * loss_mask
        marks_loss = artifacts["true mark density"] * loss_mask
        if test:
            cdf = artifacts['cumulative density'].cpu().numpy()
            valid_cdf = [cdf[i][cdf[i] >=0].tolist() for i in range(cdf.shape[0])]
            cumulative_density.extend(valid_cdf) 
        window_loss = artifacts["window integral"] * loss_mask
        epoch_ground_density_loss += detach(th.sum(ground_density_loss))
        epoch_marks_loss += detach(th.sum(marks_loss))
        epoch_window_loss += detach(th.sum(window_loss))
        loss_per_time = loss / artifacts["interval"] 
        epoch_loss_per_time += detach(th.sum(loss_per_time))
        #Only if include_poisson is True.
        if 'alpha' in artifacts:
            epoch_intensity_0 += detach(th.sum(artifacts['intensity_0'] * loss_mask)) #Baseline intensity.
            epoch_intensity_1 += detach(th.sum(artifacts['intensity_1'] * loss_mask)) #Model's intensity
            alpha += artifacts['alpha'] #Intensity weights.
        #Only if LogNormMix is used.
        if test and 'mu' in artifacts:
                epoch_mu.append(artifacts["mu"])
                epoch_sigma.append(artifacts["sigma"])
                epoch_weight.append(artifacts["w"])

        n_seqs_batch = detach(th.sum(loss_mask))
        n_seqs += n_seqs_batch
        num_batch += 1

        if args.eval_metrics:
            pred_labels.append(detach(artifacts['y_pred']))
            gold_labels.append(detach(artifacts['y_true']))
            mask_labels.append(detach(artifacts['y_pred_mask']))
            probas.append(detach(artifacts['max proba']))

    if args.eval_metrics: 
        results = eval_metrics(
            pred=pred_labels,
            gold=gold_labels,
            probas=probas,
            mask=mask_labels,
            results=results,
            n_class=args.marks,
            multi_labels=args.multi_labels,
            test=test)

    dur = time.time() - t0
    results["dur"] = dur

    results["loss"] = float(epoch_loss / n_seqs)
    results["loss_per_time"] = float(epoch_loss_per_time / n_seqs)
    results["log ground density"] = float(epoch_ground_density_loss/n_seqs)
    results["log mark density"] = float(epoch_marks_loss/n_seqs)
    results["window integral"] = float(epoch_window_loss/n_seqs)
    if test:
        results["cdf"] = cumulative_density
    results['intensity_0'] = float(epoch_intensity_0/n_seqs)
    results['intensity_1'] = float(epoch_intensity_1/n_seqs)
    if args.use_coefficients:
        results['alpha'] = alpha/num_batch
    results["mu"] = epoch_mu
    results["sigma"] = epoch_sigma
    results["mixture weights"] = epoch_weight
    return results


def train(
        model: Process,
        args: Namespace,
        loader: DataLoader,
        val_loader: DataLoader) -> Tuple[Process, list, list]:
    """Train a model.

    Args:
        model: Model to be trained.
        args: Arguments for training.
        loader: The dataset for training.
        val_loader: The dataset for evaluation.
        test_loader: The dataset for testing

    Returns:
        Best trained model from early stopping.

    """
    print("NEW TRAINING HAS BEGUN")
    train_metrics_list, val_metrics_list = [], []
    if args.include_poisson:
        processes = model.processes.keys()
        modules = []
        for p in processes:
            if p != 'poisson':
                modules.append(getattr(model, p))
        optimizer = Adam(
            [{'params': m.parameters()} for m in modules] + [
                {'params': model.alpha}] + [
                {'params': model.poisson.parameters(),
                 'lr': args.lr_poisson_rate_init}
            ], lr=args.lr_rate_init)
    else:
        optimizer = Adam(model.parameters(), lr=args.lr_rate_init)
    lr_scheduler = create_lr_scheduler(optimizer=optimizer, args=args)

    parameters = dict(model.named_parameters())
    lr_wait, cnt_wait, best_loss, best_epoch = 0, 0, 1e9, 0
    best_state = deepcopy(model.state_dict())
    train_dur, val_dur = list(), list()
    
    epochs = range(args.train_epochs)
    if args.verbose:
        epochs = tqdm(epochs)

    t_start = time.time()
    for epoch in epochs:
        t0, _ = time.time(), model.train()
        if args.lr_scheduler != 'plateau':
            lr_scheduler.step()
        train_metrics = {}
        epoch_loss, epoch_log_ground_density, epoch_log_mark_density, epoch_window_integral, n_seqs = 0, 0, 0, 0, 0
        for i, batch in enumerate((tqdm(loader)) if args.verbose else loader):
            batch['times'], batch['labels'], batch['seq_lens'] = batch['times'].to(args.device), batch['labels'].to(args.device), batch['seq_lens'].to(args.device)
            optimizer.zero_grad()
            loss, loss_mask, artifacts = get_loss(model, batch=batch, args=args, test=False)  # [B]
            loss = loss * loss_mask
            loss = th.sum(loss)
            check_tensor(loss)
            loss.backward()
            optimizer.step()
            epoch_loss += detach(loss)
            epoch_log_ground_density += detach(sum(artifacts['true log density'] * loss_mask))
            epoch_log_mark_density += detach(sum(artifacts['true mark density'] * loss_mask))
            epoch_window_integral += detach(sum(artifacts['window integral'] * loss_mask))
            n_seqs += detach(th.sum(loss_mask))
        train_dur.append(time.time() - t0)
        train_metrics['loss'] = epoch_loss/n_seqs
        train_metrics["log ground density"] = float(epoch_log_ground_density/n_seqs)
        train_metrics["log mark density"] = float(epoch_log_mark_density/n_seqs)
        train_metrics["window integral"] = float(epoch_window_integral/n_seqs)
        train_metrics_list.append(train_metrics)
        val_metrics = evaluate(model, args=args, loader=val_loader, test=False)
        val_dur.append(val_metrics["dur"])
        val_metrics_list.append(val_metrics)

        if args.lr_scheduler == 'plateau':
            lr_scheduler.step(metrics=val_metrics["loss"])

        new_best = val_metrics["loss"] < best_loss
        if args.loss_relative_tolerance is not None:
            abs_rel_loss_diff = (val_metrics["loss"] - best_loss) / best_loss
            abs_rel_loss_diff = abs(abs_rel_loss_diff)
            above_numerical_tolerance = (abs_rel_loss_diff >
                                         args.loss_relative_tolerance)
            new_best = new_best and above_numerical_tolerance

        if new_best:
            best_loss, best_t = val_metrics["loss"], epoch
            cnt_wait, lr_wait = 0, 0
            best_state = deepcopy(model.state_dict())
        else:
            cnt_wait, lr_wait = cnt_wait + 1, lr_wait + 1
            #print("Model didn't improve, patience at {}".format(str(cnt_wait)))
        if cnt_wait == args.patience:
            print("Early stopping! Stopping at epoch {}".format(str(epoch)), flush=True)
            break

        lr = optimizer.param_groups[0]['lr']
        train_metrics["lr"] = lr
        if args.include_poisson:
            lr_poisson = optimizer.param_groups[-1]['lr']
        else:
            lr_poisson = lr

        status = get_status(
            args=args, epoch=epoch, lr=lr, lr_poisson=lr_poisson,
            parameters=parameters, train_loss=train_metrics["loss"],
            val_metrics=val_metrics, cnt_wait=cnt_wait)
        if epoch % 1 == 0:
           print(status, flush=True)       

    model.load_state_dict(best_state)
    
    delta_t = time.time() - t_start
    hours, mins, sec = int(delta_t/3600), int((delta_t%3600)/60), int((delta_t%3600)%60)
    print("Total training time : {}:{}:{}".format(hours,mins,sec))
    train_metrics_list.append({'Training time': delta_t})
    return model, train_metrics_list, val_metrics_list


def main(args: Namespace):
    datasets = load_data(args=args) 
    loaders = dict()
    loaders["train"] = get_loader(datasets["train"], args=args, shuffle=True)
    loaders["val"] = get_loader(datasets["val"], args=args, shuffle=False)
    loaders["test"] = get_loader(datasets["test"], args=args, shuffle=False)
    exp_name = get_exp_name(args)
    if args.save_check_dir is not None:
        save_args(args, exp_name)
    model = get_model(args)
    num_params = count_parameters(model)
    print('NUM OF TRAINABLE PARAMETERS : {}'.format(num_params))
    print("INSTATIATED MODEL : {}/{}/{} on dataset {}".format(args.encoder_encoding, args.encoder, args.decoder, args.dataset))
    if args.mu_cheat and "poisson" in model.processes: 
        poisson = model.processes["poisson"].decoder
        mu = th.from_numpy(args.mu).type(
            poisson.mu.dtype).to(poisson.mu.device)
        poisson.mu.data = mu
    model, train_metrics, val_metrics = train(
        model, args=args, loader=loaders["train"],
        val_loader=loaders["val"]) 
    if args.save_check_dir is not None:
        file_path = os.path.join(args.save_check_dir, exp_name + '.pth')
        th.save(model.state_dict(), file_path)
        print('Model saved to disk')
    print("EVALUATING MODEL")
    metrics = {
        k: evaluate(model=model, args=args, loader=l, test=True)
        for k, l in loaders.items()}
    train_metrics.append(metrics['train']) 
    val_metrics.append(metrics['val'])
    test_metrics = metrics['test']
    if args.save_results_dir is not None:
        save_results(train_metrics, val_metrics, test_metrics,save_path=args.save_results_dir, exp_name=exp_name ,args=args)

def save_args(args:Namespace, exp_name:str):
    args_dic = vars(args).copy()
    args_dic.pop('device')
    for key, values in args_dic.items():
        if type(values) == np.ndarray:
            args_dic[key] = values.tolist()
    save_args_dir = os.path.join(args.save_check_dir, 'args')
    if not os.path.exists(save_args_dir):
        os.makedirs(save_args_dir)
    args_path = os.path.join(save_args_dir, exp_name + '.json')
    with open(args_path , 'w') as f:
        json.dump(args_dic, f)

def get_exp_name(args: Namespace) -> str:
    exp_name = args.encoder + '_' + args.decoder 
    if args.encoder_encoding is not None:
        exp_name += '_' + args.encoder_encoding
    if args.dataset is None:
        exp_name = 'Hawkes_' + exp_name
        if args.include_poisson:
            exp_name = 'poisson_' + exp_name
    else:
        exp_name = args.dataset + '_' + exp_name
        if args.include_poisson:
            if len(args.coefficients) == 0: 
                exp_name = 'poisson_' + exp_name
            elif args.coefficients[0] is None:
                exp_name = 'poisson_' + exp_name
            else:
                exp_name = 'poisson_coefficients_' + exp_name 
    if args.split is not None:
        exp_name = exp_name + '_split' + str(args.split)
    if args.exp_name is not None:
        exp_name += '_' + args.exp_name
    if args.config is not None:
        exp_name += '_config' + str(args.config)
    print('EXP_NAME ', exp_name)
    return exp_name

def save_results(train_metrics: Dict[str, list], val_metrics: Dict[str, list], test_metrics: Dict[str, list], 
                save_path: str, exp_name: str, args: Namespace):
    results = {'train':train_metrics, 'val':val_metrics, 'test':test_metrics, 'args':args}
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    exp_name = exp_name + '.txt'
    save_path = os.path.join(save_path, exp_name)
    with open(save_path, "wb") as fp: 
        pkl.dump(results, fp)
    print('Results saved to {}'.format(save_path, flush=True))

def get_state_dic(path:str):
    state_dic = th.load(path, map_location=th.device('cpu'))
    print(state_dic)


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.load_from_dir is not None:
        json_dir = os.path.join(os.getcwd(), parsed_args.load_from_dir)
        json_dir = os.path.join(json_dir, parsed_args.dataset)
        if parsed_args.split is not None:
            split = 'split_{}'.format(parsed_args.split)
            json_dir = os.path.join(json_dir, split)
        json_path = os.path.join(json_dir, 'args.json')
        with open(json_path, 'r') as fp:
            args_dict_json = json.load(fp)
        args_dict = vars(parsed_args)
        print("Warning: overriding some args from json:", flush=True)
        shared_keys = set(args_dict_json).intersection(set(args_dict))
        for k in shared_keys:
            v1, v2 = args_dict[k], args_dict_json[k]
            is_equal = np.allclose(v1, v2) if isinstance(
                v1, np.ndarray) else v1 == v2
            if not is_equal:
                print(f"    {k}: {v1} -> {v2}", flush=True)
        args_dict.update(args_dict_json)
        parsed_args = Namespace(**args_dict)
        parsed_args.mu = np.array(parsed_args.mu, dtype=np.float32)
        parsed_args.alpha = np.array(
            parsed_args.alpha, dtype=np.float32).reshape(
            parsed_args.mu.shape * 2)
        parsed_args.beta = np.array(
            parsed_args.beta, dtype=np.float32).reshape(
            parsed_args.mu.shape * 2)

    else:
        parsed_args.data_dir = os.path.expanduser(parsed_args.data_dir)
        parsed_args.save_dir = os.path.join(parsed_args.data_dir, "None")
        Path(parsed_args.save_dir).mkdir(parents=True, exist_ok=True)

    cuda = th.cuda.is_available() and not parsed_args.disable_cuda
    if cuda:
        parsed_args.device = th.device('cuda')
    else:
        parsed_args.device = th.device('cpu')
    make_deterministic(seed=parsed_args.seed)
    main(args=parsed_args)
