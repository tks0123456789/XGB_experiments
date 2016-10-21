import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.cross_validation import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import log_loss
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from datetime import datetime
import matplotlib.pyplot as plt

import pickle
import gzip

path = '/home/tks/github_tks/XGB_experiments/'
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
r001.to_csv(path + 'log/r001.csv')

result = r001.groupby(['n_train', 'reg_lambda']).mlogloss.mean().unstack().T
ax=result.plot(xticks=np.logspace(2, 9, 8, base=2) * .1, xlim=[.3, 60], style='o--', logx=True,
               legend=True, subplots=True, figsize=(8,6), title='Logloss', rot=0)
g = ax[0].set_xticklabels(np.logspace(2, 9, 8, base=2) * .1)
plt.savefig(path + 'log/r001_logloss.jpg')

result = r001.groupby(['n_train', 'reg_lambda']).n_estimators.mean().unstack().T
ax = result.plot(xticks=np.logspace(2, 9, 8, base=2) * .1, xlim=[.3, 60], ylim=[0, 1000],
                 style='o--', logx=True, figsize=(8,4), title='n_estimators')
g = ax.set_xticklabels(np.logspace(2, 9, 8, base=2) * .1)
plt.savefig(path + 'log/r001_n_estimators.jpg')

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

# r002
# 2016/2/15 1h25m
# parameters
n_iter_n_train = 1
n_iter_cv = 3
n_iter_pred = 3

reg_alpha_s = np.logspace(4, 10, 7, base=2) * .01
xgb_params_lst = [{'reg_alpha':rl, 'reg_lambda':0} for rl in reg_alpha_s]
                       
n_train_lst = [10000, 20000, 40000]


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

r002 = pd.DataFrame(scores)
r002.to_csv(path + 'log/r002.csv')

xticks = np.logspace(4, 10, 7, base=2) * .01
xlim = [.1, 11.]
result = r002.groupby(['n_train', 'reg_alpha']).mlogloss.mean().unstack().T
ax=result.plot(xticks=xticks, xlim=xlim, style='o--', logx=True,
               legend=True, subplots=True, figsize=(8,6), title='Logloss', rot=0)
g = ax[0].set_xticklabels(xticks)
plt.savefig(path + 'log/r002_logloss.jpg')

result = r002.groupby(['n_train', 'reg_alpha']).n_estimators.mean().unstack().T
ax = result.plot(xticks=xticks, xlim=xlim, ylim=[0, 1000],
                 style='o--', logx=True, figsize=(8,4), title='n_estimators')
g = ax.set_xticklabels(xticks)
plt.savefig(path + 'log/r002_n_estimators.jpg')

r002.groupby(['n_train', 'reg_alpha']).mlogloss.mean().unstack()
reg_alpha  0.160   0.320   0.640   1.280   2.560   5.120   10.240
n_train                                                          
10000       0.440   0.438   0.436   0.437   0.441   0.455   0.478
20000       0.350   0.347   0.345   0.347   0.351   0.362   0.390
40000       0.264   0.261   0.258   0.260   0.265   0.277   0.307

r002.groupby(['n_train', 'reg_alpha']).n_estimators.mean().unstack()
reg_alpha  0.160   0.320   0.640   1.280   2.560   5.120   10.240
n_train                                                          
10000         109     116     132     153     248     323     771
20000         185     192     217     231     370     710     989
40000         346     356     356     433     546     993     996

# r003
# 2016/2/15 1h22m
# Binary problem
# parameters

covtype = fetch_covtype()

X = covtype.data
# 1..7 => 0..6
target = covtype.target-1
# pd.Series(target).value_counts()
# 1    283301 *
# 0    211840 *
# 2     35754
# 6     20510
# 5     17367
# 4      9493
# 3      2747

# (495141, 54)
X = X[target < 2]
y = target[target < 2]

# 学習候補とテストに分割
# test_size=100000 だが Stratified で分割すると 100001 になる
kf = StratifiedShuffleSplit(y, n_iter=1, test_size=100000, random_state=132)

train_idx, test_idx = list(kf)[0]


X_test = X[test_idx]
y_test = y[test_idx]

n_iter_n_train = 5
n_iter_cv = 5
n_iter_pred = 4

reg_alpha_s = np.logspace(3, 9, 7, base=2) * .01
reg_lambda_s = np.logspace(2, 9, 8, base=2) * .1
xgb_params_lst = [{'reg_alpha':rl, 'reg_lambda':0} for rl in reg_alpha_s]
xgb_params_lst.extend([{'reg_alpha':0, 'reg_lambda':rl} for rl in reg_lambda_s])
                       
n_train_lst = [10000, 20000]

model_leaf_info = []
scores = []
t0 = datetime.now()
for n_train in n_train_lst:
    # 1.学習候補データから n_train 個の学習データを取る（n_iter_n_train回）
    # 2.  学習データ内で Cross-validation を行い n_estimators を決定
    # 3.  2で求めた n_estimators を使い学習、モデル複雑度を計算、テストデータでの予測（n_iter_pred回）
    
    # 1.
    print 'Train size:%d' % n_train
    sss_train = StratifiedShuffleSplit(y[train_idx], n_iter=n_iter_n_train,
                                       train_size=n_train, random_state=123)
    for idx, ignore in sss_train:
        X_train = X[train_idx][idx]
        y_train = y[train_idx][idx]
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
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="logloss",
                          early_stopping_rounds=50, verbose=False)
                n_estimators += model.best_iteration
            sc = params.copy()
            sc.update({'n_estimators':n_estimators / n_iter_cv})
            params_lst_optimized.append(sc)
        print 'Step 2 Done.', datetime.now() - t0
        # 3.
        model = XGBClassifier(max_depth=10, subsample=.8)
        for params in params_lst_optimized:
            for seed_train in range(234, 234+n_iter_pred):
                params.update({'seed':seed_train})
                model.set_params(**params)
                model.fit(X_train, y_train)
                # 4.モデル複雑度 = 葉の数、葉のweight
                dump = model._Booster.get_dump()
                leaf_values_lst = [get_leaf_values(tree) for tree in dump]
                mean_cnt_leafs = np.mean([len(v) for v in leaf_values_lst])
                mean_abs_leaf_values = np.mean([np.abs(v).mean() for v in leaf_values_lst])
                model_leaf_info.append(leaf_values_lst)
                pr = model.predict_proba(X_test)
                sc = params.copy()
                sc.update({'n_train':n_train, 'logloss':log_loss(y_test, pr),
                           'leaf_cnt':mean_cnt_leafs, 'leaf_val':mean_abs_leaf_values})
                scores.append(sc)
            print scores[-1], datetime.now() - t0
        print 'Step 3 Done', datetime.now() - t0

r003 = pd.DataFrame(scores)
r003.to_csv(path + 'log/r003.csv')

xticks = np.logspace(4, 10, 7, base=2) * .01
xlim = [.1, 11.]
result = r003.groupby(['n_train', 'reg_lambda', 'reg_lambda']).mlogloss.mean().unstack().T
ax=result.plot(xticks=xticks, xlim=xlim, style='o--', logx=True,
               legend=True, subplots=True, figsize=(8,6), title='Logloss', rot=0)
g = ax[0].set_xticklabels(xticks)
plt.savefig(path + 'log/r003_logloss.jpg')

result = r003.groupby(['n_train', 'reg_lambda']).n_estimators.mean().unstack().T
ax = result.plot(xticks=xticks, xlim=xlim, ylim=[0, 1000],
                 style='o--', logx=True, figsize=(8,4), title='n_estimators')
g = ax.set_xticklabels(xticks)
plt.savefig(path + 'log/r003_n_estimators.jpg')

names = ['logloss', 'n_estimators', 'leaf_cnt', 'leaf_val']
r003[r003.reg_lambda==0].groupby(['n_train', 'reg_alpha'])[names].mean()
                   logloss  n_estimators  leaf_cnt  leaf_val
n_train reg_alpha                                           
10000   0.080        0.324       124.200   133.560     0.070
        0.160        0.322       132.400   133.807     0.063
        0.320        0.322       119.600   141.787     0.056
        0.640        0.321       134.000   151.506     0.041
        1.280        0.320       146.200   160.991     0.027
        2.560        0.322       196.400   138.536     0.015
        5.120        0.329       322.200    87.770     0.007
20000   0.080        0.256       228.200   152.082     0.064
        0.160        0.255       228.000   155.376     0.058
        0.320        0.254       214.200   164.969     0.050
        0.640        0.252       235.200   178.985     0.037
        1.280        0.253       246.800   192.282     0.024
        2.560        0.256       324.000   167.639     0.013
        5.120        0.263       527.200   107.797     0.007

r003[r003.reg_alpha==0].groupby(['n_train', 'reg_lambda'])[names].mean()
                    logloss  n_estimators  leaf_cnt  leaf_val
n_train reg_lambda                                           
10000   0.400         0.322       139.800   129.065     0.058
        0.800         0.321       154.600   128.269     0.049
        1.600         0.320       171.200   131.145     0.038
        3.200         0.320       193.400   137.152     0.028
        6.400         0.321       230.400   145.378     0.019
        12.800        0.322       299.600   154.021     0.012
        25.600        0.323       379.400   160.509     0.008
        51.200        0.326       566.000   160.090     0.004
20000   0.400         0.257       243.400   148.635     0.054
        0.800         0.256       250.200   150.513     0.046
        1.600         0.254       296.000   152.205     0.036
        3.200         0.254       330.800   159.911     0.026
        6.400         0.254       390.800   171.094     0.018
        12.800        0.256       453.000   186.408     0.012
        25.600        0.258       578.800   200.489     0.008
        51.200        0.260       817.400   207.948     0.004
output = gzip.open(path + 'model003.pkl.gz', 'wb')
pickle.dump(model_leaf_info, output)
output.close()
