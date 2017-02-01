"""
2017/2/1 1.3h
exp name  : exp001
desciption: Complexity of XGB model
fname     : exp001.py
env       : i7 4790k, 32G, GTX1070, ubuntu 14.04.5LTS, Python 3.4.3
preprocess: None
result    : Logloss, Feature importance, w_L1, w_L2, Leaf counts, Time
params:
  model               : CPU, GPU, hist_256, hist_1024
  n_train             : 10K, 100K, 1M, 2M, 4M
  n_valid             : n_train/4
  n_features          : 32
  n_rounds            : 50
  n_clusters_per_class: 8
  max_depth           : 5, 10, 15

time
                     CPU    GPU  hist_256  hist_1024
n_train max_depth                                   
10000   5            0.3    0.3       0.6        2.5
        15           0.8    1.6       6.0       21.4
100000  5            3.3    1.0       1.3        3.5
        15          10.1    4.7      26.9      102.0
1000000 5           39.4   16.3       6.6        9.7
        15         134.8   44.7     114.4      417.9
2000000 5          107.2   36.0      12.4       16.1
        15         359.5  101.8     124.8      427.9
4000000 5          257.8   78.4      23.8       30.5
        15         950.3  227.6     218.6      684.6

leaf_cnts
                      CPU     GPU  hist_256  hist_1024
n_train max_depth                                     
10000   5            1350    1352      1381       1380
        15          20122   19418     20467      20357
100000  5            1569    1567      1560       1557
        15          91411   88088     92975      94300
1000000 5            1598    1599      1594       1595
        15         341319  338454    356076     351983
2000000 5            1600    1600      1600       1600
        15         332436  335480    335903     335697
4000000 5            1600    1600      1600       1600
        15         542859  528910    551295     544035

w_L1
                       CPU      GPU  hist_256  hist_1024
n_train max_depth                                       
10000   5             93.0     93.1      93.2       94.7
        15          1044.9   1045.8    1066.3     1056.1
100000  5            107.6    108.5     105.3      106.7
        15          5384.5   5369.3    5486.6     5521.3
1000000 5            108.4    109.9     109.0      107.0
        15         21782.9  22101.1   22675.5    22448.2
2000000 5            116.3    116.2     112.4      118.2
        15         22207.7  22916.0   22371.6    22370.2
4000000 5            101.6    108.2     101.6      100.5
        15         37074.2  36807.7   37554.5    37091.1

w_L2
                     CPU    GPU  hist_256  hist_1024
n_train max_depth                                   
10000   5           20.7   20.6      20.6       20.8
        15          60.1   60.0      60.5       60.1
100000  5           22.5   22.8      22.1       22.4
        15         140.3  139.7     141.9      142.3
1000000 5           22.5   22.7      22.7       22.3
        15         301.7  305.2     308.8      308.2
2000000 5           23.7   23.7      23.0       24.0
        15         317.5  325.5     317.8      319.1
4000000 5           21.1   27.7      21.2       20.8
        15         410.2  414.9     412.2      410.6

logloss
                      CPU     GPU  hist_256  hist_1024
n_train max_depth                                     
10000   5          0.4021  0.4075    0.4043     0.4007
        15         0.3074  0.3129    0.3173     0.3127
100000  5          0.4333  0.4365    0.4335     0.4361
        15         0.2672  0.2691    0.2689     0.2655
1000000 5          0.4373  0.4370    0.4383     0.4372
        15         0.2225  0.2217    0.2221     0.2210
2000000 5          0.4947  0.4928    0.4903     0.4949
        15         0.2284  0.2285    0.2307     0.2283
4000000 5          0.4582  0.4554    0.4572     0.4568
        15         0.2300  0.2305    0.2306     0.2311

Done: 4681.082534313202 seconds

"""
import pandas as pd
import numpy as np
import time
time_begin = time.time()

from sklearn.datasets import make_classification

from utility import experiment_xgb

params_xgb_cpu = {'objective'       : 'binary:logistic',
                  'eval_metric'     : 'logloss',
                  'tree_method'     : 'exact',
                  'updater'         : 'grow_colmaker',
                  'eta'             : 0.1, #default=0.3
                  'lambda'          : 1, #default
                  'min_child_weight': 1, #default
                  'silent'          : True,
                  'threads'         : 8}

params_xgb_hist_1 = params_xgb_cpu.copy()
params_xgb_hist_1.update({'tree_method': 'hist',
                          'updater'    : 'grow_fast_histmaker',
                          'grow_policy': 'depthwise',
                          'max_bin'    : 256,  #default
                         })
params_xgb_hist_2 = params_xgb_hist_1.copy()
params_xgb_hist_2.update({'max_bin'    : 1024})
                         

params_xgb_gpu = params_xgb_cpu.copy()
params_xgb_gpu.update({'updater':'grow_gpu'})

params_xgb_lst = [params_xgb_cpu, params_xgb_gpu, params_xgb_hist_1, params_xgb_hist_2]
model_str_lst = ['CPU', 'GPU', 'hist_256', 'hist_1024']

params = []
times = []
valid_scores = []
total_leaf_cnts = []
total_w_L1 = []
total_w_L2 = []
            
stat_name_lst = ['time', 'leaf_cnts', 'w_L1', 'w_L2', 'score']
stats_dict = {name:[] for name in stat_name_lst}

n_classes = 2
n_clusters_per_class = 8
n_features = 32
n_informative = n_redundant = n_features // 4
n_rounds = 50
fname_header = "exp001_"

N = 10**4
for n_train in [N, 10*N, 100*N, 200*N, 400*N]:
    n_valid = n_train // 4
    n_all = n_train + n_valid
    X, y = make_classification(n_samples=n_all, n_classes=n_classes,
                               n_features=n_features,
                               n_informative=n_informative,
                               n_redundant=n_redundant,
                               n_clusters_per_class=n_clusters_per_class,
                               shuffle=True, random_state=123+n_train)
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_valid = X[n_train:]
    y_valid = y[n_train:]
    for max_depth in [5, 15]:
        fname_footer = "n_%d_md_%d.csv" % (n_train, max_depth)
        for params_xgb in params_xgb_lst:
            params_xgb['max_depth'] = max_depth
        params.append({'n_train':n_train, 'max_depth':max_depth})
        print('\n')
        print(params[-1])
        stats = experiment_xgb(X_train, y_train, X_valid, y_valid,
                               params_xgb_lst, model_str_lst, n_rounds=n_rounds,
                               fname_header=fname_header,
                               fname_footer=fname_footer,
                               n_skip=4)
        for name in stat_name_lst:
            stats_dict[name].append(stats[name])

keys = ['n_train', 'max_depth']

df_stats_dict = {}
for name in stat_name_lst:
    df_stats_dict[name] = pd.DataFrame(stats_dict[name], columns=model_str_lst).join(pd.DataFrame(params)).set_index(keys)
    df_stats_dict[name].to_csv('log/' + fname_header + '%s.csv' % name)

#

pd.set_option('display.precision', 1)
pd.set_option('display.width', 100)
print('\n')
for name in stat_name_lst[:-1]:
    print('\n%s' % name)
    print(df_stats_dict[name])


pd.set_option('display.precision', 4)
metric = params_xgb['eval_metric']
print('\n%s' % metric)
print(df_stats_dict['score'])

print ("\nDone: %s seconds" % (str(time.time() - time_begin)))
