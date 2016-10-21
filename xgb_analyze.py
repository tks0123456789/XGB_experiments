import numpy as np
import pandas as pd
import xgboost as xgb
import re

from xgboost.sklearn import XGBClassifier
from datetime import datetime

dmp = model._Booster.get_dump()

dmp[0].count('f15')

cnt_leafs = []
for tree in dmp:
    cnt_leafs.append(tree.count('leaf'))

def get_leaf_values(tree_str):
    # To find 'leaf=0.123\n'
    prog=re.compile(r"(?<=leaf\=)(.+)\n")
    result = [float(rval) for rval in prog.findall(tree_str)]
    return np.array(result)

def get_all_leaves(bst):
    dmp = bst.get_dump()
    return [get_leaf_values(tree) for tree in dmp]
    

## BNP
sss = StratifiedShuffleSplit(y[tr_idx], n_iter=20, test_size=0.1, random_state=321)

param = {'max_depth':5, 'eta':.025, 'silent':1, 'objective':'binary:logistic',
         'eval_metric':'logloss', 'alpha':1, 'lambda':0, 'gamma':0,
         'min_child_weight':1, 'subsample':.8, 'colsample_bytree':1, 'nthread':8}

ntree_min = int(1.5 / param['eta'])
ntree_max = int(3. / param['eta'])

evals_result = {}
n_rounds=20000
feats = []
for i, idx in enumerate(sss):
    param['seed'] = 12324 + i
    dtrain = xgb.DMatrix(X[tr_idx[idx[1]]], label = y[tr_idx[idx[1]]])
    evallist = [(dtrain, 'train'), (dvalid, 'valid')]

    bst = xgb.train(param, dtrain, n_rounds, evallist,
                    early_stopping_rounds=50,
                    evals_result=evals_result, verbose_eval=False)
    obj = get_all_leaves(bst)
    ite = bst.best_iteration
    print(bst.best_score, ite, float(evals_result['train']['logloss'][ite]))
    pr_prev = np.zeros(X.shape[0])
    for ntree in range(ntree_min-1, ntree_max+1):
        pr = bst.predict(dall, ntree_limit=ntree)
        if ntree >= ntree_min:
            feats.append(pr - pr_prev)
        pr_prev = pr


X2 = sp.sparse.hstack((np.column_stack(feats), X_numeric, X_cat)).tocsr()
X2 = sp.sparse.hstack((X_numeric, X_cat)).tocsr()
X2 = np.column_stack(feats)

dtrain2 = xgb.DMatrix(X2[tr_idx], label = y[tr_idx])
dvalid2 = xgb.DMatrix(X2[va_idx], label = y[va_idx])
evallist = [(dtrain2, 'train'), (dvalid2, 'valid')]

evals_result = {}
n_rounds=20000
param2 = {'max_depth':10, 'eta':.1, 'silent':1, 'objective':'binary:logistic',
          'eval_metric':'logloss', 'alpha':0, 'lambda':1, 'gamma':0,
          'min_child_weight':100, 'subsample':.8, 'colsample_bytree':.2, 'nthread':8, 'seed':123}
bst = xgb.train(param2, dtrain2, n_rounds, evallist,
                early_stopping_rounds=20,
                evals_result=evals_result, verbose_eval=True)

imp = feature_importance(bst)
