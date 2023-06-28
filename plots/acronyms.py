def get_acronym(models):
    map_ec = {
        'gru_conditional-poisson_times_only': 'GRU-EC-TO',
        'gru_conditional-poisson_log_times_only' : 'GRU-EC-LTO',
        'gru_conditional-poisson_concatenate': 'GRU-EC-CONCAT',
        'gru_conditional-poisson_log_concatenate' : 'GRU-EC-LCONCAT',
        'gru_conditional-poisson_temporal' : 'GRU-EC-TEM',
        'gru_conditional-poisson_temporal_with_labels' : 'GRU-EC-TEMWL',
        'gru_conditional-poisson_learnable' : 'GRU-EC-LE',
        'gru_conditional-poisson_learnable_with_labels' : 'GRU-EC-LEWL',
        'selfattention_conditional-poisson_times_only' : 'SA-EC-TO',
        'selfattention_conditional-poisson_log_times_only' : 'SA-EC-LTO',
        'selfattention_conditional-poisson_concatenate' : 'SA-EC-CONCAT',
        'selfattention_conditional-poisson_log_concatenate': 'SA-EC-LCONCAT',
        'selfattention_conditional-poisson_temporal' : 'SA-EC-TEM',
        'selfattention_conditional-poisson_temporal_with_labels' : 'SA-EC-TEMWL',
        'selfattention_conditional-poisson_learnable' : 'SA-EC-LE',
        'selfattention_conditional-poisson_learnable_with_labels': 'SA-EC-LEWL',
        'constant_conditional-poisson_temporal': 'CONS-EC-TO',
        'gru_conditional-poisson_times_only_base': 'GRU-EC-TO + B',
        'gru_conditional-poisson_log_times_only_base': 'GRU-EC-LTO + B',
        'gru_conditional-poisson_concatenate_base' : 'GRU-EC-CONCAT + B',
        'gru_conditional-poisson_log_concatenate_base' : 'GRU-EC-LCONCAT + B',
        'gru_conditional-poisson_temporal_base' : 'GRU-EC-TEM + B',
        'gru_conditional-poisson_temporal_with_labels_base' : 'GRU-EC-TEMWL + B',
        'gru_conditional-poisson_learnable_base' : 'GRU-EC-LE + B',
        'gru_conditional-poisson_learnable_with_labels_base': 'GRU-EC-LEWL + B',
        'selfattention_conditional-poisson_times_only_base' : 'SA-EC-TO + B',
        'selfattention_conditional-poisson_log_times_only_base' : 'SA-EC-LTO + B',
        'selfattention_conditional-poisson_concatenate_base' : 'SA-EC-CONCAT + B',
        'selfattention_conditional-poisson_log_concatenate_base' : 'SA-EC-LCONCAT + B',
        'selfattention_conditional-poisson_temporal_base': 'SA-EC-TEM + B',
        'selfattention_conditional-poisson_temporal_with_labels_base': 'SA-EC-TEMWL + B',
        'selfattention_conditional-poisson_learnable_base': 'SA-EC-LE + B',
        'selfattention_conditional-poisson_learnable_with_labels_base' : 'SA-EC-LEWL + B',
        'constant_conditional-poisson_temporal_base' : 'CONS-EC-TO + B'
    }
    map_lnm = {
        'gru_log-normal-mixture_times_only': 'GRU-LNM-TO',
        'gru_log-normal-mixture_log_times_only':'GRU-LNM-LTO',
        'gru_log-normal-mixture_concatenate':'GRU-LNM-CONCAT',
        'gru_log-normal-mixture_log_concatenate':'GRU-LNM-LCONCAT',
        'gru_log-normal-mixture_temporal':'GRU-LNM-TEM',
        'gru_log-normal-mixture_temporal_with_labels':'GRU-LNM-TEMWL',
        'gru_log-normal-mixture_learnable':'GRU-LNM-LE',
        'gru_log-normal-mixture_learnable_with_labels':'GRU-LNM-LEWL',
        'selfattention_log-normal-mixture_times_only':'SA-LNM-TO',
        'selfattention_log-normal-mixture_log_times_only':'SA-LNM-LTO',
        'selfattention_log-normal-mixture_concatenate':'SA-LNM-CONCAT',
        'selfattention_log-normal-mixture_log_concatenate':'SA-LNM-LCONCAT',
        'selfattention_log-normal-mixture_temporal':'SA-LNM-TEM',
        'selfattention_log-normal-mixture_temporal_with_labels':'SA-LNM-TEMWL',
        'selfattention_log-normal-mixture_learnable':'SA-LNM-LE',
        'selfattention_log-normal-mixture_learnable_with_labels':'SA-LNM-LEWL',
        'constant_log-normal-mixture_temporal':'CONS-LNM-TO',
        'gru_log-normal-mixture_times_only_lnmk1':'GRU-LN-TO',
        'gru_log-normal-mixture_log_times_only_lnmk1':'GRU-LN-LTO',
        'gru_log-normal-mixture_concatenate_lnmk1':'GRU-LN-CONCAT',
        'gru_log-normal-mixture_log_concatenate_lnmk1':'GRU-LN-LCONCAT',
        'gru_log-normal-mixture_temporal_lnmk1':'GRU-LN-TEM',
        'gru_log-normal-mixture_temporal_with_labels_lnmk1':'GRU-LN-TEMWL',
        'gru_log-normal-mixture_learnable_lnmk1':'GRU-LN-LE',
        'gru_log-normal-mixture_learnable_with_labels_lnmk1':'GRU-LN-LEWL',
        'selfattention_log-normal-mixture_times_only_lnmk1':'SA-LN-TO',
        'selfattention_log-normal-mixture_log_times_only_lnmk1':'SA-LN-LTO',
        'selfattention_log-normal-mixture_concatenate_lnmk1':'SA-LN-CONCAT',
        'selfattention_log-normal-mixture_log_concatenate_lnmk1':'SA-LN-LCONCAT',
        'selfattention_log-normal-mixture_temporal_lnmk1':'SA-LN-TEM',
        'selfattention_log-normal-mixture_temporal_with_labels_lnmk1':'SA-LN-TEMWL',
        'selfattention_log-normal-mixture_learnable_lnmk1':'SA-LN-LE',
        'selfattention_log-normal-mixture_learnable_with_labels_lnmk1':'SA-LN-LEWL',
        'constant_log-normal-mixture_temporal_lnmk1':'CONS-LN-TO',
        'gru_log-normal-mixture_concatenate_THP_set1':'GRU-LNM-CONCAT'
    }
    
    map_mlp_cm = {
        'gru_mlp-cm_times_only_base':'GRU-FNN-TO',
        'gru_mlp-cm_log_times_only_base':'GRU-FNN-LTO',
        'gru_mlp-cm_concatenate_base':'GRU-FNN-CONCAT',
        'gru_mlp-cm_log_concatenate_base':'GRU-FNN-LCONCAT',
        'gru_mlp-cm_learnable_base':'GRU-FNN-LE',
        'gru_mlp-cm_learnable_with_labels_base':'GRU-FNN-LEWL',
        'selfattention_mlp-cm_times_only_base':'SA-FNN-TO',
        'selfattention_mlp-cm_log_times_only_base':'SA-FNN-LTO',
        'selfattention_mlp-cm_concatenate_base':'SA-FNN-CONCAT',
        'selfattention_mlp-cm_log_concatenate_base':'SA-FNN-LCONCAT',
        'selfattention_mlp-cm_learnable_base':'SA-FNN-LE',
        'selfattention_mlp-cm_learnable_with_labels_base':'SA-FNN-LEWL',
        'constant_mlp-cm_times_only_base':'CONS-FNN-TO',
        'constant_mlp-cm_log_times_only_base':'CONS-FNN-LTO',
        'constant_mlp-cm_concatenate_base':'CONS-FNN-CONCAT',
        'constant_mlp-cm_log_concatenate_base':'CONS-FNN-LCONCAT',
        'constant_mlp-cm_learnable_base':'CONS-FNN-LE',
        'constant_mlp-cm_learnable_with_labels_base':'CONS-FNN-LEWL'
    }
    
    map_mlp_mc = {
        'gru_mlp-mc_times_only_base':'GRU-MLP/MC-TO',
        'gru_mlp-mc_log_times_only_base':'GRU-MLP/MC-LTO',
        'gru_mlp-mc_concatenate_base':'GRU-MLP/MC-CONCAT',
        'gru_mlp-mc_log_concatenate_base':'GRU-MLP/MC-LCONCAT',
        'gru_mlp-mc_temporal_base':'GRU-MLP/MC-TEM',
        'gru_mlp-mc_temporal_with_labels_base':'GRU-MLP/MC-TEMWL',
        'gru_mlp-mc_learnable_base':'GRU-MLP/MC-LE',
        'gru_mlp-mc_learnable_with_labels_base':'GRU-MLP/MC-LEWL',
        'selfattention_mlp-mc_times_only_base':'SA-MLP/MC-TO',
        'selfattention_mlp-mc_log_times_only_base':'SA-MLP/MC-LTO',
        'selfattention_mlp-mc_concatenate_base':'SA-MLP/MC-CONCAT',
        'selfattention_mlp-mc_log_concatenate_base':'SA-MLP/MC-LCONCAT',
        'selfattention_mlp-mc_temporal_base':'SA-MLP/MC-TEM',
        'selfattention_mlp-mc_temporal_with_labels_base':'SA-MLP/MC-TEMWL',
        'selfattention_mlp-mc_learnable_base':'SA-MLP/MC-LE',
        'selfattention_mlp-mc_learnable_with_labels_base':'SA-MLP/MC-LEWL',
        'constant_mlp-mc_times_only_base':'CONS-MLP/MC-TO',
        'constant_mlp-mc_temporal_base':'CONS-MLP/MC-TEM',
        'constant_mlp-mc_temporal_with_labels_base':'CONS-MLP/MC-TEMWL',
        'constant_mlp-mc_log_times_only_base':'CONS-MLP/MC-LTO',
        'constant_mlp-mc_concatenate_base':'CONS-MLP/MC-CONCAT',
        'constant_mlp-mc_learnable_base':'CONS-MLP/MC-LE',
        'constant_mlp-mc_learnable_with_labels_base':'CONS-MLP/MC-LEWL',
        'constant_mlp-mc_log_concatenate_base':'CONS-MLP/MC-LCONCAT',
        'selfattention_mlp-mc_temporal_with_labels': 'SA-MLP/MC-TEMWL',
        'gru_mlp-mc_temporal_with_labels_THP_set1_base':'GRU-MLP/MC-TEMWL-THP'
    }

    map_rmtpp = {
        'gru_rmtpp_times_only':'GRU-RMTPP-TO',
        'gru_rmtpp_log_times_only':'GRU-RMTPP-LTO',
        'gru_rmtpp_concatenate':'GRU-RMTPP-CONCAT',
        'gru_rmtpp_log_concatenate':'GRU-RMTPP-LCONCAT',
        'gru_rmtpp_temporal':'GRU-RMTPP-TEM',
        'gru_rmtpp_temporal_with_labels':'GRU-RMTPP-TEMWL',
        'gru_rmtpp_learnable':'GRU-RMTPP-LE',
        'gru_rmtpp_learnable_with_labels':'GRU-RMTPP-LEWL',
        'selfattention_rmtpp_times_only':'SA-RMTPP-TO',
        'selfattention_rmtpp_log_times_only':'SA-RMTPP-LTO',
        'selfattention_rmtpp_concatenate':'SA-RMTPP-CONCAT',
        'selfattention_rmtpp_log_concatenate':'SA-RMTPP-LCONCAT',
        'selfattention_rmtpp_temporal':'SA-RMTPP-TEM',
        'selfattention_rmtpp_temporal_with_labels':'SA-RMTPP-TEMWL',
        'selfattention_rmtpp_learnable':'SA-RMTPP-LE',
        'selfattention_rmtpp_learnable_with_labels':'SA-RMTPP-LEWL',
        'constant_rmtpp_log_times_only':'CONS-RMTPP-LTO',
        'constant_rmtpp_temporal':'CONS-RMTPP-TO',
        'gru_rmtpp_times_only_base':'GRU-RMTPP-TO + B',
        'gru_rmtpp_log_times_only_base':'GRU-RMTPP-LTO + B',
        'gru_rmtpp_concatenate_base':'GRU-RMTPP-CONCAT + B',
        'gru_rmtpp_log_concatenate_base':'GRU-RMTPP-LCONCAT + B',
        'gru_rmtpp_temporal_base':'GRU-RMTPP-TEM + B',
        'gru_rmtpp_temporal_with_labels_base':'GRU-RMTPP-TEMWL + B',
        'gru_rmtpp_learnable_base':'GRU-RMTPP-LE + B',
        'gru_rmtpp_learnable_with_labels_base':'GRU-RMTPP-LEWL + B',
        'selfattention_rmtpp_times_only_base':'SA-RMTPP-TO + B',
        'selfattention_rmtpp_log_times_only_base':'SA-RMTPP-LTO + B',
        'selfattention_rmtpp_concatenate_base':'SA-RMTPP-CONCAT + B',
        'selfattention_rmtpp_log_concatenate_base':'SA-RMTPP-LCONCAT + B',
        'selfattention_rmtpp_temporal_base':'SA-RMTPP-TEM + B',
        'selfattention_rmtpp_temporal_with_labels_base':'SA-RMTPP-TEMWL + B',
        'selfattention_rmtpp_learnable_base':'SA-RMTPP-LE + B',
        'selfattention_rmtpp_learnable_with_labels_base':'SA-RMTPP-LEWL + B',
        'constant_rmtpp_log_times_only_base':'CONS-RMTPP-LTO + B',
        'constant_rmtpp_temporal_base':'CONS-RMTPP-TO + B',
        'gru_rmtpp_concatenate_THP_set1': 'GRU-RMTPP-CONCAT'
        }

    map_sa_cm = {
        'gru_selfattention-cm_times_only_base':'GRU-SA/CM-TO',
        'gru_selfattention-cm_log_times_only_base':'GRU-SA/CM-LTO',
        'gru_selfattention-cm_concatenate_base':'GRU-SA/CM-CONCAT',
        'gru_selfattention-cm_log_concatenate_base':'GRU-SA/CM-LCONCAT',
        'gru_selfattention-cm_learnable_base':'GRU-SA/CM-LE',
        'gru_selfattention-cm_learnable_with_labels_base':'GRU-SA/CM-LEWL',
        'selfattention_selfattention-cm_times_only_base':'SA-SA/CM-TO',
        'selfattention_selfattention-cm_log_times_only_base':'SA-SA/CM-LTO',
        'selfattention_selfattention-cm_concatenate_base':'SA-SA/CM-CONCAT',
        'selfattention_selfattention-cm_log_concatenate_base':'SA-SA/CM-LCONCAT',
        'selfattention_selfattention-cm_learnable_base':'SA-SA/CM-LE',
        'selfattention_selfattention-cm_learnable_with_labels_base':'SA-SA/CM-LEWL',
        'constant_selfattention-cm_times_only_base':'CONS-SA/CM-TO',
        'constant_selfattention-cm_log_times_only_base':'CONS-SA/CM-LTO',
        'constant_selfattention-cm_concatenate_base':'CONS-SA/CM-CONCAT',
        'constant_selfattention-cm_log_concatenate_base':'CONS-SA/CM-LCONCAT',
        'constant_selfattention-cm_learnable_base':'CONS-SA/CM-LE',
        'constant_selfattention-cm_learnable_with_labels_base':'CONS-SA/CM-LEWL' 
    }

    map_sa_mc = {
        'gru_selfattention-mc_times_only_base':'GRU-SA/MC-TO',
        'gru_selfattention-mc_log_times_only_base':'GRU-SA/MC-LTO',
        'gru_selfattention-mc_concatenate_base':'GRU-SA/MC-CONCAT',
        'gru_selfattention-mc_log_concatenate_base':'GRU-SA/MC-LCONCAT',
        'gru_selfattention-mc_temporal_base':'GRU-SA/MC-TEM',
        'gru_selfattention-mc_temporal_with_labels_base':'GRU-SA/MC-TEMWL',
        'gru_selfattention-mc_learnable_base':'GRU-SA/MC-LE',
        'gru_selfattention-mc_learnable_with_labels_base':'GRU-SA/MC-LEWL',
        'selfattention_selfattention-mc_times_only_base':'SA-SA/MC-TO',
        'selfattention_selfattention-mc_log_times_only_base':'SA-SA/MC-LTO',
        'selfattention_selfattention-mc_concatenate_base':'SA-SA/MC-CONCAT',
        'selfattention_selfattention-mc_log_concatenate_base': 'SA-SA/MC-LCONCAT',
        'selfattention_selfattention-mc_temporal_base':'SA-SA/MC-TEM',
        'selfattention_selfattention-mc_temporal_with_labels_base':'SA-SA/MC-TEMWL',
        'selfattention_selfattention-mc_learnable_base':'SA-SA/MC-LE',
        'selfattention_selfattention-mc_learnable_with_labels_base':'SA-SA/MC-LEWL',
        'constant_selfattention-mc_times_only_base':'CONS-SA/MC-TO',
        'constant_selfattention-mc_temporal_base':'CONS-SA/MC-TEM',
        'constant_selfattention-mc_temporal_with_labels_base':'CONS-SA/MC-TEMWL',
        'constant_selfattention-mc_log_times_only_base':'CONS-SA/MC-LTO',
        'constant_selfattention-mc_concatenate_base':'CONS-SA/MC-CONCAT',
        'constant_selfattention-mc_learnable_base':'CONS-SA/MC-LE',
        'constant_selfattention-mc_learnable_with_labels_base':'CONS-SA/MC-LEWL',
        'constant_selfattention-mc_log_concatenate_base':'CONS-SA/MC-LCONCAT',
    }

    map_nh = {
        'identity_neural-hawkes_times_only': 'NH',
        'identity_neural-hawkes_times_only_base': 'NH + B'
    }

    map_h_p_mrc = {
        'stub_hawkes_times_only':'Hawkes',
        'identity_poisson_times_only': 'Poisson',
        'MRC': 'MRC'
    }
    
    map_fixed = {
        'gru-fixed_conditional-poisson_times_only': 'GRU-Fixed + EC',
        'gru-fixed_log-normal-mixture_concatenate': 'GRU-Fixed + LNM',
        'gru-fixed_mlp-cm_log_concatenate_base': 'GRU-Fixed + FNN',
        'gru-fixed_mlp-mc_log_times_only_base': 'GRU-Fixed + MLP/MC',
        'gru-fixed_rmtpp_log_concatenate': 'GRU-Fixed + RMTPP',
        'gru-fixed_selfattention-cm_log_concatenate_base': 'GRU-Fixed + SA/CM',
        'gru-fixed_selfattention-mc_learnable_base': 'GRU-Fixed + SA/MC',
        'gru-fixed_conditional-poisson_temporal': 'GRU-Fixed + EC',
        'gru-fixed_log-normal-mixture_temporal': 'GRU-Fixed + LNM',
        'gru-fixed_mlp-cm_learnable': 'GRU-Fixed + FNN',
        'gru-fixed_mlp-mc_temporal': 'GRU-Fixed + MLP/MC',
        'gru-fixed_rmtpp_temporal': 'GRU-Fixed + RMTPP',
        'gru-fixed_selfattention-cm_learnable': 'GRU-Fixed + SA/CM',
        'gru-fixed_selfattention-mc_temporal': 'GRU-Fixed + SA/MC',
        'selfattention-fixed_selfattention-cm_log_concatenate_base': 'SA-Fixed + SA/CM'
    }
    
    
    map = {}
    maps = [map_ec, map_lnm, map_mlp_cm, map_mlp_mc, map_rmtpp, map_sa_cm, map_sa_mc, map_h_p_mrc, map_fixed, map_nh] 
    for mapping in maps:
        map.update(mapping)
    new_models = [map[model] for model in models]
    return new_models

def short_acronyms(model):
    if 'conditional-poisson' in model:
        short = 'EC'
    elif 'log-normal-mixture' in model:
        short = 'LNM'
    elif 'mlp-cm' in model:
        short = 'FNN'
    elif 'mlp-mc' in model:
        short = 'MLP/MC'
    elif 'rmtpp' in model:
        short = 'RMTPP'
    elif 'selfattention-cm' in model:
        short = 'SA/CM'
    elif 'selfattention-mc' in model:
        short = 'SA/MC'
    elif 'stub_hawkes' in model:
        short = 'Hawkes'
    elif 'neural-hawkes' in model:
        short = 'NH'
    elif 'identity_poisson' in model:
        short = 'Poisson'
    return short

def map_datasets_name(dataset):
    mapping = {'lastfm_filtered': 'LastFM',
               'mooc_filtered': 'MOOC',
               'wikipedia_filtered': 'Wikipedia',
               'github_filtered': 'Github',
               'mimic2_filtered': 'MIMIC2',
               'stack_overflow_filtered': 'Stack O.',
               'retweets_filtered':'Retweets',
               'taxi': 'Taxi',
               'twitter': 'Twitter',
               'reddit_askscience_comments': 'Reddit Comments',
               'reddit_politics_submissions': 'Reddit Subs.',
               'pubg': 'PUBG',
               'yelp_toronto': 'Yelp Toronto',
               'yelp_mississauga': 'Yelp Mississauga',
               'yelp_airport': 'Yelp Airport',
               'hawkes_exponential_mutual':'Hawkes'}
    return mapping[dataset]