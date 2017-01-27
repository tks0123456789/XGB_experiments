import numpy as np
import re
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
    NOTE: the initial learning rate will still take in-effect on first iteration.
    Parameters
    ----------
    param_values: list or function
        List of learning rate for each boosting round
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
        if isinstance(param_values, list):
            if len(param_values) != n:
                raise ValueError("Length of list 'param_values' has to equal 'num_boost_round'.")
            new_learning_rate = param_values[i]
        else:
            new_learning_rate = param_values(i, n)
        return new_learning_rate

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
