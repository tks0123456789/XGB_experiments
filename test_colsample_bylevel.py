import xgboost as xgb
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=2**15+1, n_features=10, n_informative=5, n_redundant=5,
                           shuffle=True, random_state=123)

param = {'objective':'binary:logistic','tree_method':'approx',
         'colsample_bylevel':0.01, 'eval_metric':'logloss',
         'seed':123, 'silent':1}

n_rounds = 1
for n in [2**15+1, 2**15]:
    print("\nn:%d" %n)
    dtrain = xgb.DMatrix(X[:n], y[:n])
    bst = xgb.train(param, dtrain, n_rounds, [(dtrain, 'train')])
