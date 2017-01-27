"""

"""

import time
import numpy as np
import scipy as sp
import pandas as pd
import xgboost as xgb
import re
from sklearn.datasets import make_classification
from sklearn.cross_validation import StratifiedShuffleSplit

def get_leaf_values(tree_str):
    # To find 'leaf=0.123\n'
    prog=re.compile(r"(?<=leaf\=)(.+)\n")
    result = [float(rval) for rval in prog.findall(tree_str)]
    return np.array(result)

def get_all_leaves(bst):
    dmp = bst.get_dump()
    return [get_leaf_values(tree) for tree in dmp]

# init begin
n = 2 ** 15
n_classes = 3
X, y = make_classification(n_samples=n, n_classes=n_classes,
                           n_features=100, n_informative=75, n_redundant=20,
                           class_sep=0.5, shuffle=True, random_state=123)

sss = StratifiedShuffleSplit(y, n_iter=1, test_size=0.2, random_state=321)
train_idx, valid_idx = list(sss)[0]


df_data = pd.DataFrame(np.column_stack((y, X)))
df_data.iloc[train_idx].to_csv('cl3_train.csv', index=False, header=False)
df_data.iloc[valid_idx].to_csv('cl3_valid.csv', index=False, header=False)
# init end

# r013
# No preprocessing
# 2016/11/2 40m
dtrain = xgb.DMatrix(X[train_idx], label = y[train_idx])
dvalid = xgb.DMatrix(X[valid_idx], label = y[valid_idx])

n_rounds = 500
evals_result = {}
param = {'max_depth':100, 'eta':0.1, 'silent':1, 'objective':'multi:softmax',
         'num_class':n_classes, 'min_child_weight':100, 'lambda':0,
         'eval_metric':'mlogloss', 'nthread':-1, 'seed':123}

t0 = time.time()
bst = xgb.train(param, dtrain, n_rounds, [(dvalid, 'valid')],
                evals_result=evals_result, verbose_eval=True)
print(time.time() - t0)

tmp = get_all_leaves(bst)
n_nodes = np.array([len(s) for s in tmp]).reshape((n_rounds, n_classes))

df = pd.DataFrame({'valid_loss':evals_result['valid']['mlogloss'],
                   'leaf_cnt_0':n_nodes[:,0],
                   'leaf_cnt_1':n_nodes[:,1],
                   'leaf_cnt_2':n_nodes[:,2]})
df.to_csv('log/r013.csv')

print(df.iloc[99::100,:])
     leaf_cnt_0  leaf_cnt_1  leaf_cnt_2  valid_loss
99           48          48          49    0.447766
199          29          28          29    0.309277
299          22          19          21    0.250185
399          15          17          15    0.217990
499          14          13          14    0.197343

# r014
# equal_frequency_binning
# 2016/11/2 21.6m
def equal_frequency_binning(X, nbins=255):
    rval_X = []
    rval_bins = []
    for i in range(X.shape[1]):
        x = X[:, i]
        bins = np.percentile(x, np.linspace(0, 100, nbins))
        rval_bins.append(bins)
        x_cut = pd.cut(x, bins)
        rval_X.append(x_cut.codes)
    return np.column_stack(rval_X), rval_bins
        
X2, ignore = equal_frequency_binning(X, nbins=255)
    
dtrain = xgb.DMatrix(X2[train_idx], label = y[train_idx])
dvalid = xgb.DMatrix(X2[valid_idx], label = y[valid_idx])

n_rounds = 500
evals_result = {}
param = {'max_depth':100, 'eta':0.1, 'silent':1, 'objective':'multi:softmax',
         'num_class':n_classes, 'min_child_weight':100, 'lambda':0,
         'eval_metric':'mlogloss', 'nthread':-1, 'seed':123}

t0 = time.time()
bst = xgb.train(param, dtrain, n_rounds, [(dvalid, 'valid')],
                evals_result=evals_result, verbose_eval=True)
print(time.time() - t0)

tmp = get_all_leaves(bst)
n_nodes = np.array([len(s) for s in tmp]).reshape((n_rounds, n_classes))

df = pd.DataFrame({'valid_loss':evals_result['valid']['mlogloss'],
                   'leaf_cnt_0':n_nodes[:,0],
                   'leaf_cnt_1':n_nodes[:,1],
                   'leaf_cnt_2':n_nodes[:,2]})
df.to_csv('log/r014.csv')

print(df.iloc[99::100,:])
     leaf_cnt_0  leaf_cnt_1  leaf_cnt_2  valid_loss
99           50          50          46    0.447109
199          30          29          29    0.307661
299          22          21          21    0.247279
399          18          16          17    0.214577
499          13          13          14    0.193776

imp = pd.Series(bst.get_fscore())
imp.sort_values(ascending=False).head(10)
f35    682
f15    677
f39    676
f71    672
f90    657
f23    647
f94    641
f60    641
f10    632
f75    626

imp.sort_values(ascending=False).tail(10)
f73    307
f26    301
f72    296
f59    295
f13    278
f17     62
f47     39
f29     23
f34     10
f93      6

# r015
# equal_frequency_binning
# 2016/11/9 21.8m
def equal_frequency_binning(X, nbins=255):
    rval_X = []
    rval_bins = []
    for i in range(X.shape[1]):
        x = X[:, i]
        x_cut, bins = pd.qcut(x, nbins, retbins=True)
        rval_X.append(x_cut.codes)
        rval_bins.append(bins)
    return np.column_stack(rval_X), rval_bins
        
X2, ignore = equal_frequency_binning(X, nbins=255)

dtrain = xgb.DMatrix(X2[train_idx], label = y[train_idx])
dvalid = xgb.DMatrix(X2[valid_idx], label = y[valid_idx])

n_rounds = 500
evals_result = {}
param = {'max_depth':100, 'eta':0.1, 'silent':1, 'objective':'multi:softmax',
         'num_class':n_classes, 'min_child_weight':100, 'lambda':0,
         'eval_metric':'mlogloss', 'nthread':-1, 'seed':123}

t0 = time.time()
bst = xgb.train(param, dtrain, n_rounds, [(dvalid, 'valid')],
                evals_result=evals_result, verbose_eval=True)
print(time.time() - t0)

tmp = get_all_leaves(bst)
n_nodes = np.array([len(s) for s in tmp]).reshape((n_rounds, n_classes))

df = pd.DataFrame({'valid_loss':evals_result['valid']['mlogloss'],
                   'leaf_cnt_0':n_nodes[:,0],
                   'leaf_cnt_1':n_nodes[:,1],
                   'leaf_cnt_2':n_nodes[:,2]})
df.to_csv('log/r015.csv')

print(df.iloc[99::100,:])
     leaf_cnt_0  leaf_cnt_1  leaf_cnt_2  valid_loss
99           51          49          49    0.446756
199          29          30          30    0.307652
299          21          21          24    0.248894
399          16          16          16    0.216791
499          13          14          12    0.195948

imp = pd.Series(bst.get_fscore())
imp.sort_values(ascending=False).head(10)
f39    699
f35    686
f15    682
f60    680
f90    672
f23    672
f10    648
f71    639
f75    631
f62    630

imp.sort_values(ascending=False).tail(10)
f59    320
f56    302
f26    280
f13    272
f72    265
f17     62
f47     41
f29     26
f93     17
f34     10
