import xgboost as xgb
from sklearn.datasets import make_classification

n = 2 ** 15
X, y = make_classification(n_samples=n+1, n_features=10, n_informative=5, n_redundant=5,
                           shuffle=True, random_state=123)

param = {'objective':'binary:logistic','tree_method':'approx',
         'eval_metric':'logloss','seed':123}

print('num_row:%d tree_method:%s' % (n+1, 'approx'))
dtrain = xgb.DMatrix(X, y)
for cs in [1, 0.1, 0.01]:
    print("colsample_bylevel:%.2f" % cs)
    param['colsample_bylevel'] = cs
    bst = xgb.train(param, dtrain, 1, [(dtrain, 'train')])

print('num_row:%d tree_method:%s' % (n, 'approx'))
dtrain = xgb.DMatrix(X[:n], y[:n])
for cs in [1, 0.1, 0.01]:
    print("colsample_bylevel:%.2f" % cs)
    param['colsample_bylevel'] = cs
    bst = xgb.train(param, dtrain, 1, [(dtrain, 'train')])

print('num_row:%d tree_method:%s' % (n+1, 'exact'))
param['tree_method'] = 'exact'
dtrain = xgb.DMatrix(X, y)
for cs in [1, 0.1, 0.01]:
    print("colsample_bylevel:%.2f" % cs)
    param['colsample_bylevel'] = cs
    bst = xgb.train(param, dtrain, 1, [(dtrain, 'train')])

"""
num_row:32769 tree_method:approx
colsample_bylevel:1.00
[02:55:11] Tree method is selected to be 'approx'
[02:55:11] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 116 extra nodes, 0 pruned nodes, max_depth=6
[0]	train-logloss:0.505822
colsample_bylevel:0.10
[02:55:11] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 116 extra nodes, 0 pruned nodes, max_depth=6
[0]	train-logloss:0.505822
colsample_bylevel:0.01
[02:55:11] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 116 extra nodes, 0 pruned nodes, max_depth=6
[0]	train-logloss:0.505822

num_row:32768 tree_method:approx
colsample_bylevel:1.00
[02:55:44] Tree method is selected to be 'approx'
[02:55:44] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 118 extra nodes, 0 pruned nodes, max_depth=6
[0]	train-logloss:0.504609
colsample_bylevel:0.10
[02:55:44] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 114 extra nodes, 0 pruned nodes, max_depth=6
[0]	train-logloss:0.546038
colsample_bylevel:0.01
[02:55:44] dmlc-core/include/dmlc/logging.h:235: [02:55:44] src/tree/updater_colmaker.cc:637: Check failed: (n) > (0) colsample_bylevel is too small that no feature can be included

num_row:32769 tree_method:exact
colsample_bylevel:1.00
[03:04:47] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 118 extra nodes, 0 pruned nodes, max_depth=6
[0]	train-logloss:0.504607
colsample_bylevel:0.10
[03:04:47] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 114 extra nodes, 0 pruned nodes, max_depth=6
[0]	train-logloss:0.546035
colsample_bylevel:0.01
[02:56:02] dmlc-core/include/dmlc/logging.h:235: [02:56:02] src/tree/updater_colmaker.cc:637: Check failed: (n) > (0) colsample_bylevel is too small that no feature can be included
"""
