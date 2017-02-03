import numpy as np
import pandas as pd
import time
import re
import xgboost as xgb
from xgboost.callback import _get_callback_context

def leaf_cnts(bst):
    dump = bst.get_dump()
    return([tree.count('leaf') for tree in dump])

def get_leaf_values(tree_str):
    # To find 'leaf=0.123\n'
    prog=re.compile(r"(?<=leaf\=)(.+)\n")
    result = [float(rval) for rval in prog.findall(tree_str)]
    return np.array(result)

def get_all_leaves(bst):
    dmp = bst.get_dump()
    return [get_leaf_values(tree) for tree in dmp]

def reset_parameters(param_name, param_values):
    """Reset paramter values after iteration 1
    Parameters
    ----------
    param_values: list or function
        List of parameter values for each boosting round
        or a customized function that calculates eta in terms of
        current number of round and the total number of boosting round (e.g. yields
        learning rate decay)
        - list l: eta = l[boosting round]
        - function f: eta = f(boosting round, num_boost_round)
    Returns
    -------
    callback : function
        The requested callback function.
    """
    def get_param_value(i, n, param_values):
        """helper providing the learning rate"""
        if isinstance(param_values, (list, np.ndarray)):
            if len(param_values) != n:
                raise ValueError("Length of list 'param_values' has to equal 'num_boost_round'.")
            new_param_value = param_values[i]
        else:
            new_param_value = param_values(i, n)
        return new_param_value

    def callback(env):
        """internal function"""
        context = _get_callback_context(env)

        if context == 'train':
            bst, i, n = env.model, env.iteration, env.end_iteration
            bst.set_param(param_name, get_param_value(i, n, param_values))
        elif context == 'cv':
            i, n = env.iteration, env.end_iteration
            for cvpack in env.cvfolds:
                bst = cvpack.bst
                bst.set_param(param_name, get_param_value(i, n, param_values))

    callback.before_iteration = True
    return callback


def experiment_xgb(X_train, y_train, X_valid, y_valid,
                   params_xgb_lst, model_str_lst,
                   n_rounds=10,
                   fname_header=None, fname_footer=None, n_skip=10):
    t000 = time.time()
    
    df_score_train = pd.DataFrame(index=range(n_rounds))
    df_score_valid = pd.DataFrame(index=range(n_rounds))
    feature_names = ['f%d' % i for i in range(X_train.shape[1])]
    feat_imps_dict = {}
    leaf_cnts_dict = {}
    w_L1_dict = {}
    w_L2_dict = {}
    time_sec_lst = []

    # XGBoost
    metric = params_xgb_lst[0]['eval_metric']
    xgmat_train = xgb.DMatrix(X_train, label=y_train)
    xgmat_valid = xgb.DMatrix(X_valid, label=y_valid)
    watchlist = [(xgmat_train,'train'), (xgmat_valid, 'valid')]
    print("training XGBoost")
    for params_xgb,  model_str in zip(params_xgb_lst, model_str_lst):
        evals_result = {}
        t0 = time.time()
        bst = xgb.train(params_xgb, xgmat_train, n_rounds, watchlist,
                        evals_result=evals_result, verbose_eval=False)
        time_sec_lst.append(time.time() - t0)
        print("%s: %s seconds" % (model_str, str(time_sec_lst[-1])))
        df_score_train[model_str] = evals_result['train'][metric]
        df_score_valid[model_str] = evals_result['valid'][metric]
        feat_imps_dict[model_str] = pd.Series(bst.get_score(importance_type='gain'), index=feature_names)
        leaves_lst = get_all_leaves(bst)
        leaf_cnts_dict[model_str] = [len(leaves) for leaves in leaves_lst]
        w_L1_dict[model_str] = [np.sum(np.abs(leaves)) for leaves in leaves_lst]
        w_L2_dict[model_str] = [np.sqrt(np.sum(leaves ** 2)) for leaves in leaves_lst]

    print('\n%s train' % metric)
    print(df_score_train.iloc[::n_skip,])
    print('\n%s valid' % metric)
    print(df_score_valid.iloc[::n_skip,])

    columns = model_str_lst
    
    print('\nLeaf counts')
    df_leaf_cnts = pd.DataFrame(leaf_cnts_dict, columns=columns)
    print(df_leaf_cnts.iloc[::n_skip,])
    
    print('\nw L1 sum')
    df_w_L1 = pd.DataFrame(w_L1_dict, columns=columns)
    print(df_w_L1.iloc[::n_skip,])

    print('\nw L2 sum')
    df_w_L2 = pd.DataFrame(w_L2_dict, columns=columns)
    print(df_w_L2.iloc[::n_skip,])
    
    df_feat_imps = pd.DataFrame(feat_imps_dict,
                                index=feature_names,
                                columns=columns).fillna(0)
    df_feat_imps /= df_feat_imps.sum(0)
    df_feat_imps = df_feat_imps.sort_values(model_str_lst[0], ascending=False)
    print('\nFeature importance(gain) sorted by ' + model_str_lst[0])
    print(df_feat_imps.head(5))
    if fname_header is not None:
        df_score_train.to_csv('log/' + fname_header + 'Score_Train_' + fname_footer)
        df_score_valid.to_csv('log/' + fname_header + 'Score_Valid_' + fname_footer)
        df_leaf_cnts.to_csv('log/' + fname_header + 'Leaf_cnts_' + fname_footer)
        df_w_L1.to_csv('log/' + fname_header + 'w_L1__' + fname_footer)
        df_w_L2.to_csv('log/' + fname_header + 'w_L2__' + fname_footer)
        df_feat_imps.to_csv('log/' + fname_header + 'Feat_imps_' + fname_footer)
    return{'time'     : time_sec_lst,
           'score'    : df_score_valid.tail(1).values[0].tolist(),
           'leaf_cnts': df_leaf_cnts.sum(0),
           'w_L1'     : df_w_L1.sum(0),
           'w_L2'     : df_w_L2.sum(0)}

def experiment(X_train, y_train, X_valid, y_valid,
               n_rounds, params_xgb,
               param_name=None, params_values=None):
    if param_name is None:
        callbacks = None
    else:
        callbacks = [reset_parameters(param_name, param_values)]
    xgmat_train = xgb.DMatrix(X_train, label=y_train)
    xgmat_valid = xgb.DMatrix(X_valid, label=y_valid)
    watchlist = [(xgmat_valid, 'valid')]
    evals_result = {}
    t0 = time.time()
    bst = xgb.train(params_xgb, xgmat_train, n_rounds, watchlist,
                    callbacks=callbacks,
                    early_stopping_rounds=30,
                    evals_result=evals_result, verbose_eval=False)

    ntree = len(evals_result['valid']['logloss'])
    df_scores = pd.DataFrame({'valid_loss':evals_result['valid']['logloss']},
                             index=pd.Index(range(1, ntree+1), name='Boosting iteration'))

    leaves_lst = get_all_leaves(bst)[:ntree]
    df_leaf_cnts = pd.DataFrame({'leaf_cnts':[len(leaves) for leaves in leaves_lst]},
                                index=pd.Index(range(1, ntree+1), name='Boosting iteration'))
    df_w_L2 = pd.DataFrame({'w_L2':[np.sqrt(np.sum(leaves**2)) for leaves in leaves_lst]},
                           index=pd.Index(range(1, ntree+1), name='Boosting iteration'))
    print("valid_loss:%.4f, ntree:%d, %.1fs" % \
          (evals_result['valid']['logloss'][bst.best_iteration],
           bst.best_ntree_limit,
           (time.time() - t0)))
    fig, ax = plt.subplots(3, sharex=True, figsize=(13,9))
    df_scores.plot(ax=ax[0])
    df_leaf_cnts.plot(ax=ax[1])
    df_w_L2.plot(ax=ax[2])
