
def config_elec(model_name, args):
    params = {}
    # some common params for all models
    params['input_dim'] = args.input_dim
    params['n_units'] = args.n_units
    params['bias'] = True
    params['time_depth']=args.depth
    params['output_dim']=args.output_dim

    if model_name in ['Delelstm']:
        params['N_units'] = args.N_units
    elif model_name in ['AttentionDeLELSTM', 'DeLELSTM_AttnNoDecomp']:
        params['N_units'] = args.N_units
        n_units = params['n_units']
        if hasattr(args, 'attention_heads') and args.attention_heads is not None:
            params['attention_heads'] = int(args.attention_heads)
        else:
            if n_units % 4 == 0:
                params['attention_heads'] = 4
            elif n_units % 2 == 0:
                params['attention_heads'] = 2
            else:
                params['attention_heads'] = 1
        if hasattr(args, 'attention_threshold') and args.attention_threshold is not None:
            params['attention_threshold'] = float(args.attention_threshold)
        else:
            params['attention_threshold'] = 0.05
        if hasattr(args, 'ridge_lambda') and args.ridge_lambda is not None:
            params['ridge_lambda'] = float(args.ridge_lambda)
        else:
            params['ridge_lambda'] = 1e-3

    elif model_name == 'IMV_full':
            params['n_units']=16

    elif model_name == 'IMV_tensor':
             pass

    elif model_name == 'Retain':
        params['intputDimSize'] = args.input_dim
        params['embDimSize'] = 256
        params['alphaHiddenDimSize']=128
        params['betaHiddenDimSize'] = 128
        params['outputDimSize']=args.output_dim
        params['keep_prob']=1.0

    elif model_name == 'LSTM':
        pass
    else:
        raise ModuleNotFoundError

    return params

def config_pm(model_name, args):
    params = {}
    # some common params for all models
    params['input_dim'] = args.input_dim
    params['n_units'] = args.n_units
    params['bias'] = True
    params['time_depth']=args.depth
    params['output_dim']=args.output_dim

    if model_name in ['Delelstm']:
        params['N_units'] = args.N_units
    elif model_name in ['AttentionDeLELSTM', 'DeLELSTM_AttnNoDecomp']:
        params['N_units'] = args.N_units
        n_units = params['n_units']
        if hasattr(args, 'attention_heads') and args.attention_heads is not None:
            params['attention_heads'] = int(args.attention_heads)
        else:
            if n_units % 4 == 0:
                params['attention_heads'] = 4
            elif n_units % 2 == 0:
                params['attention_heads'] = 2
            else:
                params['attention_heads'] = 1
        if hasattr(args, 'attention_threshold') and args.attention_threshold is not None:
            params['attention_threshold'] = float(args.attention_threshold)
        else:
            params['attention_threshold'] = 0.05
        if hasattr(args, 'ridge_lambda') and args.ridge_lambda is not None:
            params['ridge_lambda'] = float(args.ridge_lambda)
        else:
            params['ridge_lambda'] = 1e-3

    elif model_name == 'IMV_full':
            params['n_units']=16

    elif model_name == 'IMV_tensor':
             pass

    elif model_name == 'Retain':
        params['intputDimSize'] = args.input_dim
        params['embDimSize'] =64
        params['alphaHiddenDimSize']=64
        params['betaHiddenDimSize'] = 64
        params['outputDimSize']=args.output_dim
        params['keep_prob']=1.0
        params['batch_size']=args.batch_size

    elif model_name == 'LSTM':
        pass
    else:
        raise ModuleNotFoundError

    return params

def config_exchange(model_name, args):
    params = {}
    # some common params for all models
    params['input_dim'] = args.input_dim
    params['n_units'] = args.n_units
    params['bias'] = True
    params['time_depth']=args.depth
    params['output_dim']=args.output_dim

    if model_name in ['Delelstm']:
        params['N_units'] = args.N_units
    elif model_name in ['AttentionDeLELSTM', 'DeLELSTM_AttnNoDecomp']:
        params['N_units'] = args.N_units
        n_units = params['n_units']
        if hasattr(args, 'attention_heads') and args.attention_heads is not None:
            params['attention_heads'] = int(args.attention_heads)
        else:
            if n_units % 4 == 0:
                params['attention_heads'] = 4
            elif n_units % 2 == 0:
                params['attention_heads'] = 2
            else:
                params['attention_heads'] = 1
        if hasattr(args, 'attention_threshold') and args.attention_threshold is not None:
            params['attention_threshold'] = float(args.attention_threshold)
        else:
            params['attention_threshold'] = 0.05
        if hasattr(args, 'ridge_lambda') and args.ridge_lambda is not None:
            params['ridge_lambda'] = float(args.ridge_lambda)
        else:
            params['ridge_lambda'] = 1e-3

    elif model_name == 'IMV_full':
            params['n_units']=16

    elif model_name == 'IMV_tensor':
             pass

    elif model_name == 'Retain':
        params['intputDimSize'] = args.input_dim
        params['embDimSize'] = 64
        params['alphaHiddenDimSize']=64
        params['betaHiddenDimSize'] = 64
        params['outputDimSize']=args.output_dim
        params['keep_prob']=1.0
        params['batch_size'] = args.batch_size

    elif model_name == 'LSTM':
        pass
    else:
        raise ModuleNotFoundError

    return params


def config(model_name, args):
    if args.dataset == 'electricity':
        params = config_elec(model_name, args)

    elif args.dataset == 'PM':
        params = config_pm(model_name, args)

    elif args.dataset == 'exchange':
        params = config_exchange(model_name, args)

    return params

