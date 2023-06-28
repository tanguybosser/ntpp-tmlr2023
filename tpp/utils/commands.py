from argparse import Namespace

def get_command(dataset:str, 
                save_dir:str, 
                batch_size:int, 
                split:int, 
                config_num:int, 
                args:Namespace, 
                eval_metrics:bool, 
                config:dict, 
                save_check_dir:str):
    if args.encoding == 'temporal_with_labels':
        decoder_encoding = 'temporal'
    if args.encoding == 'learnable_with_labels':
        decoder_encoding = 'learnable'
    elif args.encoding  == 'concatenate':
        decoder_encoding = 'times_only'
    elif args.encoding == 'log_concatenate':
        decoder_encoding = 'log_times_only'
    else:
        decoder_encoding = args.encoding
    if args.encoder in ['gru', 'constant'] and args.decoder == 'conditional-poisson':
        if args.encoder == 'constant':
            config["encoder_units_rnn"] = config["encoder_emb_dim"] = config ["encoder_layers_rnn"] = config["encoder_units_mlp"] = 0
        cmd = "python3 -u scripts/train.py --dataset={} --load-from-dir={} --save-results-dir={} --save-check-dir={} " \
                        "--eval-metrics={} --include-poisson={} --patience={} --batch-size={} --split={} --config={} --train-epochs={} " \
                        "--exp-name={} " \
                        "--use-coefficients={} --coefficients={} --coefficients={} --use-softmax={} " \
                        "--encoder={} --encoder-encoding={} --encoder-emb-dim={} " \
                        "--encoder-units-rnn={} --encoder-layers-rnn={} " \
                        "--encoder-units-mlp={} --encoder-activation-mlp={} " \
                        "--decoder={} --decoder-encoding={} --decoder-emb-dim={} " \
                        "--decoder-units-mlp={} --decoder-units-mlp={} --decoder-activation-mlp={} --decoder-activation-final-mlp={} " \
                        .format(dataset, args.load_from_dir, save_dir, save_check_dir,
                                eval_metrics, args.include_poisson, 20, batch_size, split, config_num, args.train_epochs,
                                args.exp_name,
                                args.use_coefficients, args.coefficients[0], args.coefficients[1], args.use_softmax, 
                                args.encoder, args.encoding, config["encoder_emb_dim"],
                                config["encoder_units_rnn"], config["encoder_layers_rnn"],
                                config["encoder_units_mlp"], "relu",
                                args.decoder, args.encoding, config["decoder_emb_dim"],
                                config["decoder_units_mlp"], config["decoder_units_mlp"], 'relu', 'parametric_softplus')
    elif args.encoder in ['gru', 'constant'] and args.decoder == 'log-normal-mixture':
        if args.encoder == 'constant':
            config["encoder_units_rnn"] = config["encoder_emb_dim"] = config ["encoder_layers_rnn"] = config["encoder_units_mlp"] = 0
        cmd = "python3 -u scripts/train.py --dataset={} --load-from-dir={} --save-results-dir={} --save-check-dir={} " \
                        "--eval-metrics={} --include-poisson={} --patience={} --batch-size={} --split={} --config={} --train-epochs={} " \
                        "--exp-name={} " \
                        "--use-coefficients={} --coefficients={} --coefficients={} --use-softmax={} " \
                        "--encoder={} --encoder-encoding={} --encoder-emb-dim={} " \
                        "--encoder-units-rnn={} --encoder-layers-rnn={} " \
                        "--encoder-units-mlp={} --encoder-activation-mlp={} " \
                        "--decoder={} " \
                        "--decoder-units-mlp={} --decoder-units-mlp={} " \
                        "--decoder-n-mixture={} " \
                        .format(dataset, args.load_from_dir, save_dir, save_check_dir,
                                eval_metrics, args.include_poisson, 20, batch_size, split, config_num, args.train_epochs,
                                args.exp_name,
                                args.use_coefficients, args.coefficients[0], args.coefficients[1], args.use_softmax, 
                                args.encoder, args.encoding, config["encoder_emb_dim"],
                                config["encoder_units_rnn"], config["encoder_layers_rnn"],
                                config["encoder_units_mlp"], "relu",
                                args.decoder,
                                config["decoder_units_mlp"], config["decoder_units_mlp"],
                                config["decoder_n_mixture"])
    elif args.encoder in ['gru', 'constant'] and args.decoder == 'mlp-cm':
        assert(args.encoding  not in ['temporal', 'temporal_with_labels']), 'Wrong encoding for cumulative decoder'
        if args.encoder == 'constant':
            config["encoder_units_rnn"] = config["encoder_emb_dim"] = config ["encoder_layers_rnn"] = config["encoder_units_mlp"] = 0
        cmd = "python3 -u scripts/train.py --dataset={} --load-from-dir={} --save-results-dir={} --save-check-dir={} " \
                        "--eval-metrics={} --include-poisson={} --patience={} --batch-size={} --split={} --config={} --train-epochs={} " \
                        "--exp-name={} " \
                        "--use-coefficients={} --coefficients={} --coefficients={} --use-softmax={} " \
                        "--encoder={} --encoder-encoding={} --encoder-emb-dim={} --encoder-embedding-constraint 'nonneg' " \
                        "--encoder-units-rnn={} --encoder-layers-rnn={} --encoder-constraint-rnn 'nonneg' " \
                        "--encoder-units-mlp={} --encoder-activation-mlp 'relu' --encoder-constraint-mlp 'nonneg' " \
                        "--decoder={} --decoder-encoding={} --decoder-emb-dim={} --decoder-embedding-constraint 'nonneg' " \
                        "--decoder-units-mlp={} --decoder-units-mlp={} --decoder-activation-mlp 'gumbel_softplus' " \
                        "--decoder-activation-final-mlp 'parametric_softplus' --decoder-constraint-mlp 'nonneg' " \
                        .format(dataset, args.load_from_dir, save_dir, save_check_dir, 
                                eval_metrics, args.include_poisson, 20, batch_size, split, config_num, args.train_epochs,
                                args.exp_name,
                                args.use_coefficients, args.coefficients[0], args.coefficients[1], args.use_softmax, 
                                args.encoder, args.encoding, config["encoder_emb_dim"],
                                config["encoder_units_rnn"], config["encoder_layers_rnn"],
                                config["encoder_units_mlp"],
                                args.decoder, decoder_encoding, config["decoder_emb_dim"], 
                                config["decoder_units_mlp"]+config["decoder_emb_dim"], config["decoder_units_mlp"])
    elif args.encoder in ['gru', 'constant'] and args.decoder == 'mlp-mc':
        if args.encoder == 'constant':
            config["encoder_units_rnn"] = config["encoder_emb_dim"] = config ["encoder_layers_rnn"] = config["encoder_units_mlp"] = 0
        cmd = "python3 -u scripts/train.py --dataset={} --load-from-dir={} --save-results-dir={} --save-check-dir={} " \
                        "--eval-metrics={} --include-poisson={} --patience={} --batch-size={} --split={} --config={} --train-epochs={} " \
                        "--exp-name={} " \
                        "--use-coefficients={} --coefficients={} --coefficients={} --use-softmax={} " \
                        "--encoder={} --encoder-encoding={} --encoder-emb-dim={} " \
                        "--encoder-units-rnn={} --encoder-layers-rnn={} " \
                        "--encoder-units-mlp={} --encoder-activation-mlp 'relu' " \
                        "--decoder={} --decoder-encoding={} --decoder-emb-dim={} " \
                        "--decoder-units-mlp={} --decoder-units-mlp={} --decoder-activation-mlp 'relu' --decoder-activation-final-mlp 'parametric_softplus' " \
                        "--decoder-mc-prop-est 10" \
                        .format(dataset, args.load_from_dir, save_dir, save_check_dir, 
                                eval_metrics, args.include_poisson, 20, batch_size, split, config_num, args.train_epochs,
                                args.exp_name,
                                args.use_coefficients, args.coefficients[0], args.coefficients[1], args.use_softmax, 
                                args.encoder, args.encoding, config["encoder_emb_dim"],
                                config["encoder_units_rnn"], config["encoder_layers_rnn"],
                                config["encoder_units_mlp"],
                                args.decoder, decoder_encoding, config["decoder_emb_dim"],
                                config["decoder_units_mlp"]+config["decoder_emb_dim"], config["decoder_units_mlp"])
    elif args.encoder in ['gru', 'constant'] and args.decoder == 'rmtpp':
        if args.encoder == 'constant':
            config["encoder_units_rnn"] = config["encoder_emb_dim"] = config ["encoder_layers_rnn"] = config["encoder_units_mlp"] = 0
        decoder_encoding = "times_only" 
        if args.encoding in ["log_times_only", "log_concatenate"]:
            decoder_encoding = "log_times_only"
        cmd = "python3 -u scripts/train.py --dataset={} --load-from-dir={} --save-results-dir={} --save-check-dir={} " \
                        "--eval-metrics={} --include-poisson={} --patience={} --batch-size={} --split={} --config={} --train-epochs={} " \
                        "--exp-name={} " \
                        "--use-coefficients={} --coefficients={} --coefficients={} --use-softmax={} " \
                        "--encoder={} --encoder-encoding={} --encoder-emb-dim={} " \
                        "--encoder-units-rnn={} --encoder-layers-rnn={} " \
                        "--encoder-units-mlp={} --encoder-activation-mlp 'relu' " \
                        "--decoder={} " \
                        "--decoder-units-mlp={} " \
                        "--decoder-encoding={}" \
                        .format(dataset, args.load_from_dir, save_dir, save_check_dir,
                                eval_metrics, args.include_poisson, 20, batch_size, split, config_num, args.train_epochs,
                                args.exp_name,
                                args.use_coefficients, args.coefficients[0], args.coefficients[1], args.use_softmax, 
                                args.encoder, args.encoding, config["encoder_emb_dim"],
                                config["encoder_units_rnn"], config["encoder_layers_rnn"],
                                config["encoder_units_mlp"],
                                args.decoder,
                                config['decoder_units_mlp'],
                                decoder_encoding)        
    elif args.encoder in ['gru', 'constant'] and args.decoder == 'selfattention-cm':
        assert(args.encoding  not in ['temporal', 'temporal_with_labels']), 'Wrong encoding for cumulative decoder'
        if args.encoder == 'constant':
            config["encoder_units_rnn"] = config ["encoder_layers_rnn"] = config["encoder_units_mlp"] = config["encoder_emb_dim"] = 0
        cmd = "python3 -u scripts/train.py --dataset={} --load-from-dir={} --save-results-dir={} --save-check-dir={} " \
                        "--eval-metrics={} --include-poisson={} --patience={} --batch-size={} --split={} --config={} --train-epochs={} " \
                        "--exp-name={} " \
                        "--use-coefficients={} --coefficients={} --coefficients={} --use-softmax={} " \
                        "--encoder={} --encoder-encoding={} --encoder-emb-dim={} --encoder-embedding-constraint 'nonneg' " \
                        "--encoder-units-rnn={} --encoder-layers-rnn={} --encoder-constraint-rnn 'nonneg' " \
                        "--encoder-units-mlp={} --encoder-activation-mlp 'relu' --encoder-constraint-mlp 'nonneg' " \
                        "--decoder={} --decoder-encoding={} --decoder-emb-dim={} --decoder-embedding-constraint 'nonneg' " \
                        "--decoder-units-mlp={} --decoder-activation-mlp 'gumbel_softplus' " \
                        "--decoder-activation-final-mlp 'parametric_softplus' --decoder-constraint-mlp 'nonneg' " \
                        "--decoder-attn-activation 'sigmoid' --decoder-activation-rnn 'gumbel_softplus' --decoder-layers-rnn={} " \
                        "--decoder-units-rnn={} --decoder-n-heads={} --decoder-constraint-rnn 'nonneg' " \
                        "--lr-rate-init={} " \
                        .format(dataset, args.load_from_dir, save_dir, save_check_dir,
                                eval_metrics, args.include_poisson, 20, batch_size, split, config_num, args.train_epochs,
                                args.exp_name,
                                args.use_coefficients, args.coefficients[0], args.coefficients[1], args.use_softmax, 
                                args.encoder, args.encoding, config["encoder_emb_dim"],
                                config["encoder_units_rnn"], config["encoder_layers_rnn"],
                                config["encoder_units_mlp"],
                                args.decoder, decoder_encoding, config["decoder_emb_dim"], 
                                config["decoder_units_mlp"],
                                config["decoder_layers_rnn"],
                                config["decoder_units_rnn"], config["decoder_n_heads"],
                                args.lr)
    elif args.encoder in ['gru', 'constant'] and args.decoder == 'selfattention-mc':
        if args.encoder == 'constant':
            config["encoder_units_rnn"] = config ["encoder_layers_rnn"] = config["encoder_emb_dim"] = config["encoder_units_mlp"] = 0
        cmd = "python3 -u scripts/train.py --dataset={} --load-from-dir={} --save-results-dir={} --save-check-dir={} " \
                        "--eval-metrics={} --include-poisson={} --patience={} --batch-size={} --split={} --config={} --train-epochs={} " \
                        "--exp-name={} " \
                        "--use-coefficients={} --coefficients={} --coefficients={} --use-softmax={} " \
                        "--encoder={} --encoder-encoding={} --encoder-emb-dim={} " \
                        "--encoder-units-rnn={} --encoder-layers-rnn={} " \
                        "--encoder-units-mlp={} --encoder-activation-mlp 'relu' " \
                        "--decoder={} --decoder-encoding={} --decoder-emb-dim={} " \
                        "--decoder-units-mlp={} --decoder-activation-mlp 'relu' " \
                        "--decoder-activation-final-mlp 'parametric_softplus' " \
                        "--decoder-attn-activation 'softmax' --decoder-activation-rnn 'relu' --decoder-layers-rnn={} " \
                        "--decoder-units-rnn={} --decoder-n-heads={} " \
                        "--decoder-mc-prop-est 10 " \
                        .format(dataset, args.load_from_dir, save_dir, save_check_dir,
                                eval_metrics, args.include_poisson, 20, batch_size, split, config_num, args.train_epochs,
                                args.exp_name,
                                args.use_coefficients, args.coefficients[0], args.coefficients[1], args.use_softmax, 
                                args.encoder, args.encoding, config["encoder_emb_dim"],
                                config["encoder_units_rnn"], config["encoder_layers_rnn"],
                                config["encoder_units_mlp"],
                                args.decoder, decoder_encoding, config["decoder_emb_dim"], 
                                config["decoder_units_mlp"],
                                config["decoder_layers_rnn"],
                                config["decoder_units_rnn"], config["decoder_n_heads"])
    elif args.encoder == 'selfattention' and args.decoder == 'conditional-poisson':
        cmd = "python3 -u scripts/train.py --dataset={} --load-from-dir={} --save-results-dir={} --save-check-dir={} " \
                        "--eval-metrics={} --include-poisson={} --patience={} --batch-size={} --split={} --config={} --train-epochs={} " \
                        "--exp-name={} " \
                        "--use-coefficients={} --coefficients={} --coefficients={} --use-softmax={} " \
                        "--encoder={} --encoder-encoding={} --encoder-emb-dim={} " \
                        "--encoder-units-rnn={} --encoder-layers-rnn={} --encoder-n-heads={} --encoder-attn-activation={} " \
                        "--encoder-units-mlp={} --encoder-activation-mlp={} " \
                        "--decoder={} --decoder-encoding={} --decoder-emb-dim={} " \
                        "--decoder-units-mlp={} --decoder-units-mlp={} --decoder-activation-mlp={} --decoder-activation-final-mlp={} " \
                        .format(dataset, args.load_from_dir, save_dir, save_check_dir,
                                eval_metrics, args.include_poisson, 20, batch_size, split, config_num, args.train_epochs,
                                args.exp_name,
                                args.use_coefficients, args.coefficients[0], args.coefficients[1], args.use_softmax, 
                                args.encoder, args.encoding, config["encoder_emb_dim"],
                                config["encoder_units_rnn"], config["encoder_layers_rnn"], config["encoder_n_heads"], 'softmax',
                                config["encoder_units_mlp"], "relu",
                                args.decoder, args.encoding, config["encoder_emb_dim"],
                                config["decoder_units_mlp"], config["decoder_units_mlp"], 'relu', 'parametric_softplus')
    elif args.encoder == 'selfattention' and args.decoder == 'log-normal-mixture':
        cmd = "python3 -u scripts/train.py --dataset={} --load-from-dir={} --save-results-dir={} --save-check-dir={} " \
                        "--eval-metrics={} --include-poisson={} --patience={} --batch-size={} --split={} --config={} --train-epochs={} " \
                        "--exp-name={} " \
                        "--use-coefficients={} --coefficients={} --coefficients={} --use-softmax={} " \
                        "--encoder={} --encoder-encoding={} --encoder-emb-dim={} " \
                        "--encoder-units-rnn={} --encoder-layers-rnn={} --encoder-n-heads={} --encoder-attn-activation={} " \
                        "--encoder-units-mlp={} --encoder-activation-mlp={} " \
                        "--decoder={} " \
                        "--decoder-units-mlp={} --decoder-units-mlp={} " \
                        "--decoder-n-mixture={} " \
                        .format(dataset, args.load_from_dir, save_dir, save_check_dir,
                                eval_metrics, args.include_poisson, 20, batch_size, split, config_num, args.train_epochs,
                                args.exp_name,
                                args.use_coefficients, args.coefficients[0], args.coefficients[1], args.use_softmax, 
                                args.encoder, args.encoding, config["encoder_emb_dim"],
                                config["encoder_units_rnn"], config["encoder_layers_rnn"], config["encoder_n_heads"], 'softmax',
                                config["encoder_units_mlp"], "relu",
                                args.decoder,
                                config["decoder_units_mlp"], config["decoder_units_mlp"],
                                config["decoder_n_mixture"])
    elif args.encoder == 'selfattention' and args.decoder == 'mlp-cm':
        assert(args.encoding  not in ['temporal', 'temporal_with_labels']), 'Wrong encoding for cumulative decoder'
        cmd = "python3 -u scripts/train.py --dataset={} --load-from-dir={} --save-results-dir={} --save-check-dir={} " \
                        "--eval-metrics={} --include-poisson={} --patience={} --batch-size={} --split={} --config={} --train-epochs={} " \
                        "--exp-name={} " \
                        "--use-coefficients={} --coefficients={} --coefficients={} --use-softmax={} " \
                        "--encoder={} --encoder-encoding={} --encoder-emb-dim={} --encoder-embedding-constraint 'nonneg' " \
                        "--encoder-units-rnn={} --encoder-layers-rnn={} --encoder-n-heads={} --encoder-attn-activation={} --encoder-constraint-rnn 'nonneg' " \
                        "--encoder-units-mlp={} --encoder-activation-mlp={} --encoder-constraint-mlp 'nonneg' " \
                        "--decoder={} --decoder-encoding={} --decoder-emb-dim={} --decoder-embedding-constraint 'nonneg' " \
                        "--decoder-units-mlp={} --decoder-units-mlp={} --decoder-activation-mlp={} --decoder-activation-final-mlp={} " \
                        "--decoder-constraint-mlp 'nonneg' " \
                        "--decoder-mc-prop-est 10" \
                        .format(dataset, args.load_from_dir, save_dir, save_check_dir,
                                eval_metrics, args.include_poisson, 20, batch_size, split, config_num, args.train_epochs,
                                args.exp_name,
                                args.use_coefficients, args.coefficients[0], args.coefficients[1], args.use_softmax, 
                                args.encoder, args.encoding, config["encoder_emb_dim"],
                                config["encoder_units_rnn"], config["encoder_layers_rnn"], config["encoder_n_heads"], 'sigmoid',
                                config["encoder_units_mlp"], "relu",
                                args.decoder, decoder_encoding, config["encoder_emb_dim"],
                                config["decoder_units_mlp"]+config["encoder_emb_dim"], config["decoder_units_mlp"], 'gumbel_softplus', 'gumbel_softplus')
    elif args.encoder == 'selfattention' and args.decoder == 'mlp-mc':
        cmd = "python3 -u scripts/train.py --dataset={} --load-from-dir={} --save-results-dir={} --save-check-dir={} " \
                        "--eval-metrics={} --include-poisson={} --patience={} --batch-size={} --split={} --config={} --train-epochs={} " \
                        "--exp-name={} " \
                        "--use-coefficients={} --coefficients={} --coefficients={} --use-softmax={} " \
                        "--encoder={} --encoder-encoding={} --encoder-emb-dim={} " \
                        "--encoder-units-rnn={} --encoder-layers-rnn={} --encoder-n-heads={} --encoder-attn-activation={} " \
                        "--encoder-units-mlp={} --encoder-activation-mlp={} " \
                        "--decoder={} --decoder-encoding={} --decoder-emb-dim={} " \
                        "--decoder-units-mlp={} --decoder-units-mlp={} --decoder-activation-mlp 'relu' --decoder-activation-final-mlp 'parametric_softplus' " \
                        "--decoder-mc-prop-est 10" \
                        .format(dataset, args.load_from_dir, save_dir, save_check_dir, 
                                eval_metrics, args.include_poisson, 20, batch_size, split, config_num, args.train_epochs,
                                args.exp_name,
                                args.use_coefficients, args.coefficients[0], args.coefficients[1], args.use_softmax, 
                                args.encoder, args.encoding, config["encoder_emb_dim"],
                                config["encoder_units_rnn"], config["encoder_layers_rnn"], config["encoder_n_heads"], 'softmax',
                                config["encoder_units_mlp"], "relu",
                                args.decoder, decoder_encoding, config["encoder_emb_dim"],
                                config["decoder_units_mlp"]+config["encoder_emb_dim"], config["decoder_units_mlp"])
    elif args.encoder == 'selfattention' and args.decoder == 'rmtpp':
        decoder_encoding = "times_only"
        if args.encoding in ["log_times_only", "log_concatenate"]:
            decoder_encoding = "log_times_only"
        cmd = "python3 -u scripts/train.py --dataset={} --load-from-dir={} --save-results-dir={} --save-check-dir={} " \
                        "--eval-metrics={} --include-poisson={} --patience={} --batch-size={} --split={} --config={} --train-epochs={} " \
                        "--exp-name={} " \
                        "--use-coefficients={} --coefficients={} --coefficients={} --use-softmax={} " \
                        "--encoder={} --encoder-encoding={} --encoder-emb-dim={} " \
                        "--encoder-units-rnn={} --encoder-layers-rnn={} --encoder-n-heads={} --encoder-attn-activation={} " \
                        "--encoder-units-mlp={} --encoder-activation-mlp={} " \
                        "--decoder={} " \
                        "--decoder-units-mlp={} " \
                        "--decoder-encoding={} " \
                        .format(dataset, args.load_from_dir, save_dir, save_check_dir, 
                                eval_metrics, args.include_poisson, 20, batch_size, split, config_num, args.train_epochs,
                                args.exp_name,
                                args.use_coefficients, args.coefficients[0], args.coefficients[1], args.use_softmax, 
                                args.encoder, args.encoding, config["encoder_emb_dim"],
                                config["encoder_units_rnn"], config["encoder_layers_rnn"], config["encoder_n_heads"], 'softmax',
                                config["encoder_units_mlp"], "relu",
                                args.decoder,
                                config["decoder_units_mlp"],
                                decoder_encoding)
    elif args.encoder == 'selfattention' and args.decoder == 'selfattention-cm':
        assert(args.encoding  not in ['temporal', 'temporal_with_labels']), 'Wrong encoding for cumulative decoder'
        cmd = "python3 -u scripts/train.py --dataset={} --load-from-dir={} --save-results-dir={} --save-check-dir={} " \
                        "--eval-metrics={} --include-poisson={} --patience={} --batch-size={} --split={} --config={} --train-epochs={} " \
                        "--exp-name={} " \
                        "--use-coefficients={} --coefficients={} --coefficients={} --use-softmax={} " \
                        "--encoder={} --encoder-encoding={} --encoder-emb-dim={} --encoder-embedding-constraint 'nonneg' " \
                        "--encoder-units-rnn={} --encoder-layers-rnn={} --encoder-n-heads={} --encoder-attn-activation={} --encoder-constraint-rnn 'nonneg' " \
                        "--encoder-units-mlp={} --encoder-activation-mlp={} --encoder-constraint-mlp 'nonneg' " \
                        "--decoder={} --decoder-encoding={} --decoder-emb-dim={} --decoder-embedding-constraint 'nonneg' " \
                        "--decoder-units-mlp={} --decoder-activation-mlp 'gumbel_softplus' " \
                        "--decoder-activation-final-mlp 'gumbel_softplus' --decoder-constraint-mlp 'nonneg' " \
                        "--decoder-attn-activation 'sigmoid' --decoder-activation-rnn 'gumbel_softplus' --decoder-layers-rnn={} " \
                        "--decoder-units-rnn={} --decoder-n-heads={} --decoder-constraint-rnn 'nonneg' " \
                        "--lr-rate-init={}" \
                        .format(dataset, args.load_from_dir, save_dir, save_check_dir,
                                eval_metrics, args.include_poisson, 20, batch_size, split, config_num, args.train_epochs,
                                args.exp_name,
                                args.use_coefficients, args.coefficients[0], args.coefficients[1], args.use_softmax, 
                                args.encoder, args.encoding, config["encoder_emb_dim"],
                                config["encoder_units_rnn"], config["encoder_layers_rnn"], config["encoder_n_heads"], 'sigmoid',
                                config["encoder_units_mlp"], "relu",
                                args.decoder, decoder_encoding, config["encoder_emb_dim"], 
                                config["decoder_units_mlp"],
                                config["decoder_layers_rnn"],
                                config["decoder_units_rnn"], config["decoder_n_heads"],
                                args.lr)
    elif args.encoder == 'selfattention' and args.decoder == 'selfattention-mc':
        cmd = "python3 -u scripts/train.py --dataset={} --load-from-dir={} --save-results-dir={} --save-check-dir={} " \
                        "--eval-metrics={} --include-poisson={} --patience={} --batch-size={} --split={} --config={} --train-epochs={} " \
                        "--exp-name={} " \
                        "--use-coefficients={} --coefficients={} --coefficients={} --use-softmax={} " \
                        "--encoder={} --encoder-encoding={} --encoder-emb-dim={} " \
                        "--encoder-units-rnn={} --encoder-layers-rnn={} --encoder-n-heads={} --encoder-attn-activation={} " \
                        "--encoder-units-mlp={} --encoder-activation-mlp={} " \
                        "--decoder={} --decoder-encoding={} --decoder-emb-dim={} " \
                        "--decoder-units-mlp={} --decoder-activation-mlp 'relu' " \
                        "--decoder-activation-final-mlp 'parametric_softplus' " \
                        "--decoder-attn-activation 'softmax' --decoder-activation-rnn 'relu' --decoder-layers-rnn={} " \
                        "--decoder-units-rnn={} --decoder-n-heads={} " \
                        "--decoder-mc-prop-est 10 " \
                        .format(dataset, args.load_from_dir, save_dir, save_check_dir, 
                                eval_metrics, args.include_poisson, 20, batch_size, split, config_num, args.train_epochs,
                                args.exp_name,
                                args.use_coefficients, args.coefficients[0], args.coefficients[1], args.use_softmax, 
                                args.encoder, args.encoding, config["encoder_emb_dim"],
                                config["encoder_units_rnn"], config["encoder_layers_rnn"], config["encoder_n_heads"], 'softmax',
                                config["encoder_units_mlp"], "relu",
                                args.decoder, decoder_encoding, config["encoder_emb_dim"], 
                                config["decoder_units_mlp"],
                                config["decoder_layers_rnn"],
                                config["decoder_units_rnn"], config["decoder_n_heads"])
    elif args.decoder == 'hawkes' or args.decoder == 'poisson':
        cmd = "python3 -u scripts/train.py --dataset={} --load-from-dir={} --save-results-dir={} --save-check-dir={} " \
                        "--eval-metrics={} --include-poisson={} --patience={} --batch-size={} --split={} --config={} --train-epochs={} " \
                        "--exp-name={} " \
                        "--encoder={} " \
                        "--decoder={} " \
                        .format(dataset, args.load_from_dir, save_dir, save_check_dir, 
                                eval_metrics, args.include_poisson, 20, batch_size, split, config_num, args.train_epochs,
                                args.exp_name,
                                args.encoder,
                                args.decoder, 
                                )
    elif args.decoder == 'neural-hawkes':
        cmd = "python3 -u scripts/train.py --dataset={} --load-from-dir={} --save-results-dir={} --save-check-dir={} " \
                        "--eval-metrics={} --include-poisson={} --patience={} --batch-size={} --split={} --config={} --train-epochs={} " \
                        "--exp-name={} " \
                        "--use-coefficients={} --coefficients={} --coefficients={} --use-softmax={} " \
                        "--encoder={} " \
                        "--decoder={} " \
                        "--decoder-units-mlp={} --decoder-activation-mlp 'relu' " \
                        "--decoder-activation-final-mlp 'parametric_softplus' " \
                        "--decoder-activation-rnn 'relu' --decoder-layers-rnn={} " \
                        "--decoder-units-rnn={} " \
                        "--decoder-mc-prop-est 10 " \
                        .format(dataset, args.load_from_dir, save_dir, save_check_dir, 
                                eval_metrics, args.include_poisson, 20, batch_size, split, config_num, args.train_epochs,
                                args.exp_name,
                                args.use_coefficients, args.coefficients[0], args.coefficients[1], args.use_softmax, 
                                args.encoder,
                                args.decoder, 
                                config["decoder_units_mlp"],
                                config["decoder_layers_rnn"],
                                config["decoder_units_rnn"])

    return cmd