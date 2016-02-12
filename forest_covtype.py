import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.cross_validation import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import log_loss
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from datetime import datetime
import matplotlib.pyplot as plt

# 小数点以下第3位まで表示、指数表記しない
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# ローカルになければ自動的にダウンロードする
# デフォルトのセーブ先は ~/scikit_learn_data
covtype = fetch_covtype()

X = covtype.data
# 1..7 => 0..6
target = covtype.target-1

# 学習候補とテストに分割
# test_size=100000 だが Stratified で分割すると 100001 になる
kf = StratifiedShuffleSplit(target, n_iter=1, test_size=100000, random_state=132)

for train_idx, test_idx in kf:
    print train_idx.size, test_idx.size

X_test = X[test_idx]
y_test = target[test_idx]

# parameters
n_iter_n_train = 1
n_iter_cv = 3
n_iter_pred = 3

reg_lambda_s = np.logspace(2, 9, 8, base=2) * .1
xgb_params_lst = [{'reg_lambda':rl} for rl in reg_lambda_s]
                       
n_train_lst = [10000, 20000, 40000]

# 2016/2/12 1h46m
t0 = datetime.now()
scores = []
for n_train in n_train_lst:
    # 1.学習候補データから n_train 個の学習データを取る（n_iter_n_train回）
    # 2.  学習データ内で Cross-validation を行い n_estimators を決定
    # 3.  2で求めた n_estimators を使い学習、テストデータでの予測（n_iter_pred回）
    
    # 1.
    print 'Train size:%d' % n_train
    sss_train = StratifiedShuffleSplit(target[train_idx], n_iter=n_iter_n_train,
                                       train_size=n_train, random_state=123)
    for idx, ignore in sss_train:
        X_train = X[train_idx][idx]
        y_train = target[train_idx][idx]
        #
        # 2.
        sss_train_inner = StratifiedShuffleSplit(y_train, n_iter=n_iter_cv, test_size=.1,
                                                 random_state=456)
        model = XGBClassifier(n_estimators=1000, max_depth=10, subsample=.8, seed=987)
        params_lst_optimized = []
        for params in xgb_params_lst:
            n_estimators = 0
            for tr, va in sss_train_inner:
                X_tr, y_tr = X_train[tr], y_train[tr]
                X_va, y_va = X_train[va], y_train[va]
                model.set_params(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="mlogloss",
                          early_stopping_rounds=50, verbose=False)
                n_estimators += model.best_iteration
            sc = params.copy()
            sc.update({'n_estimators':n_estimators / n_iter_cv})
            params_lst_optimized.append(sc)
        print 'Step 2 Done.', datetime.now() - t0
        # 3.
        model = XGBClassifier(max_depth=10, subsample=.8)
        for params in params_lst_optimized:
            for seed_train in range(100, 100+n_iter_pred):
                params.update({'seed':seed_train})
                model.set_params(**params)
                model.fit(X_train, y_train)
                pr = model.predict_proba(X_test)
                sc = params.copy()
                sc.update({'n_train':n_train, 'mlogloss':log_loss(y_test, pr)})
                scores.append(sc)
            print scores[-1], datetime.now() - t0
        print 'Step 3 Done', datetime.now() - t0

r001 = pd.DataFrame(scores[:72])
r001.to_csv('log/r001.csv')

result = r001.groupby(['n_train', 'reg_lambda']).mlogloss.mean().unstack().T
ax=result.plot(xticks=np.logspace(2, 9, 8, base=2) * .1, xlim=[.3, 60], style='o--', logx=True,
               legend=True, subplots=True, figsize=(8,6), title='Logloss', rot=0)
g = ax[0].set_xticklabels(np.logspace(2, 9, 8, base=2) * .1)
plt.savefig('log/r001_logloss.jpg')

result = r001.groupby(['n_train', 'reg_lambda']).n_estimators.mean().unstack().T
ax = result.plot(xticks=np.logspace(2, 9, 8, base=2) * .1, xlim=[.3, 60], ylim=[0, 1000],
                 style='o--', logx=True, figsize=(8,4), title='n_estimators')
g = ax.set_xticklabels(np.logspace(2, 9, 8, base=2) * .1)
plt.savefig('log/r001_n_estimators.jpg')

r001.groupby(['n_train', 'reg_lambda']).mlogloss.mean().unstack()
# reg_lambda  0.400   0.800   1.600   3.200   6.400   12.800  25.600  51.200
# n_train                                                                   
# 10000        0.441   0.438   0.437   0.436   0.436   0.436   0.438   0.441
# 20000        0.353   0.352   0.348   0.348   0.347   0.347   0.348   0.351
# 40000        0.268   0.266   0.264   0.263   0.262   0.263   0.264   0.267

r001.groupby(['n_train', 'reg_lambda']).n_estimators.mean().unstack()
# reg_lambda  0.400   0.800   1.600   3.200   6.400   12.800  25.600  51.200
# n_train                                                                   
# 10000          114     129     136     160     179     240     313     471
# 20000          176     186     223     233     305     363     466     631
# 40000          325     351     396     437     500     592     773     940
