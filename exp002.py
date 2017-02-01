"""
2017/2/1-2 46m
exp name  : exp002
desciption: Complexity of XGB model, same dataset as exp001
fname     : exp002.py
env       : i7 4790k, 32G, GTX1070, ubuntu 14.04.5LTS, Python 3.4.3
preprocess: None
result    : Logloss, Feature importance, w_L1, w_L2, Leaf counts, Time
params:
  model               : hist_depthwise, hist_lossguide
  n_train             : 10K, 100K, 1M, 2M, 4M
  n_valid             : n_train/4
  n_features          : 32
  n_rounds            : 50
  n_clusters_per_class: 8
  max_depth           : 10, 15, 20
  num_leaves          : 256, 1024, 4096

time
                              hist_dw  hist_lg
n_train num_leaves max_depth                  
10000   256        10             3.2      3.1
                   15             3.8      3.8
                   20             3.8      3.9
        1024       10             3.5      3.4
                   15             5.6      5.5
                   20             6.0      5.9
        4096       15             5.6      5.5
                   20             6.0      6.0
100000  256        10             5.0      5.3
                   15             5.2      6.1
                   20             5.1      6.0
        1024       10             8.6      8.5
                   15            15.3     16.1
                   20            16.0     18.1
        4096       15            26.8     26.3
                   20            42.0     40.5
1000000 256        10            12.8     14.2
                   15            13.7     16.1
                   20            13.1     15.4
        1024       10            23.7     23.4
                   15            28.7     31.8
                   20            27.7     31.9
        4096       15            71.3     77.2
                   20            73.1     83.2
2000000 256        10            20.5     22.4
                   15            19.7     24.7
                   20            19.7     26.0
        1024       10            32.8     32.7
                   15            36.4     42.9
                   20            35.5     44.7
        4096       15            86.2     92.6
                   20            86.1    101.1
4000000 256        10            36.2     39.9
                   15            37.0     44.5
                   20            36.3     46.6
        1024       10            53.5     53.9
                   15            55.8     65.7
                   20            57.1     68.1
        4096       15           111.8    123.4
                   20           112.3    131.8

leaf_cnts
                              hist_dw  hist_lg
n_train num_leaves max_depth                  
10000   256        10            9973    10157
                   15           12624    12607
                   20           12800    12800
        1024       10           11701    11701
                   15           20467    20467
                   20           22594    22594
        4096       15           20467    20467
                   20           22594    22594
100000  256        10           12359    12098
                   15           12800    12800
                   20           12800    12800
        1024       10           23989    23989
                   15           48157    47753
                   20           51120    51200
        4096       15           92975    92975
                   20          155657   154534
1000000 256        10           12800    12800
                   15           12800    12800
                   20           12800    12800
        1024       10           39836    39836
                   15           51200    51200
                   20           51200    51200
        4096       15          196551   201731
                   20          204800   204800
2000000 256        10           12800    12800
                   15           12800    12800
                   20           12800    12800
        1024       10           41338    41338
                   15           51200    51200
                   20           51200    51200
        4096       15          204326   204792
                   20          204800   204800
4000000 256        10           12800    12800
                   15           12800    12800
                   20           12800    12800
        1024       10           44867    44867
                   15           51200    51200
                   20           51200    51200
        4096       15          202756   201534
                   20          204800   204800

w_L1
                              hist_dw  hist_lg
n_train num_leaves max_depth                  
10000   256        10           601.2    628.5
                   15           716.5    769.5
                   20           724.5    789.2
        1024       10           690.9    690.9
                   15          1066.3   1066.3
                   20          1143.4   1143.4
        4096       15          1066.3   1066.3
                   20          1143.4   1143.4
100000  256        10           865.5    886.4
                   15           889.1    920.3
                   20           889.1    907.9
        1024       10          1632.5   1632.5
                   15          3011.0   3312.4
                   20          3163.5   3588.2
        4096       15          5486.6   5486.6
                   20          8181.9   8174.2
1000000 256        10           949.0    909.2
                   15           949.0    843.5
                   20           949.0    811.5
        1024       10          2898.9   2898.9
                   15          3663.5   3675.9
                   20          3663.5   3533.9
        4096       15         13127.9  14658.4
                   20         13635.3  15084.5
2000000 256        10           998.5    934.8
                   15           998.5    833.1
                   20           998.5    814.0
        1024       10          3123.8   3123.8
                   15          3804.7   3664.1
                   20          3804.7   3507.8
        4096       15         14007.7  15575.8
                   20         14018.0  15329.4
4000000 256        10           944.4    881.6
                   15           944.4    805.2
                   20           944.4    773.6
        1024       10          3358.4   3358.4
                   15          3839.9   3467.4
                   20          3839.9   3337.3
        4096       15         14459.6  14933.9
                   20         14602.1  14303.1

w_L2
                              hist_dw  hist_lg
n_train num_leaves max_depth                  
10000   256        10            48.6     49.1
                   15            53.7     54.0
                   20            54.0     54.5
        1024       10            50.8     50.8
                   15            60.5     60.5
                   20            62.4     62.4
        4096       15            60.5     60.5
                   20            62.4     62.4
100000  256        10            64.5     64.1
                   15            65.5     64.7
                   20            65.5     63.7
        1024       10            84.2     84.2
                   15           114.6    118.3
                   20           118.2    123.5
        4096       15           141.9    141.9
                   20           170.5    170.2
1000000 256        10            68.6     64.2
                   15            68.6     59.3
                   20            68.6     57.2
        1024       10           117.3    117.3
                   15           133.3    129.0
                   20           133.3    124.0
        4096       15           246.6    259.5
                   20           252.5    262.1
2000000 256        10            71.9     65.6
                   15            71.9     58.3
                   20            71.9     57.0
        1024       10           125.5    125.5
                   15           138.5    127.5
                   20           138.5    122.2
        4096       15           258.5    271.7
                   20           258.5    266.0
4000000 256        10            68.4     63.0
                   15            68.4     57.2
                   20            68.4     55.1
        1024       10           129.4    129.4
                   15           139.7    123.1
                   20           139.7    118.5
        4096       15           267.1    267.0
                   20           268.7    254.0

logloss
                              hist_dw  hist_lg
n_train num_leaves max_depth                  
10000   256        10          0.3173   0.3212
                   15          0.3049   0.3099
                   20          0.3071   0.3065
        1024       10          0.3150   0.3150
                   15          0.3173   0.3173
                   20          0.3111   0.3111
        4096       15          0.3173   0.3173
                   20          0.3111   0.3111
100000  256        10          0.3351   0.3111
                   15          0.3320   0.2888
                   20          0.3320   0.2846
        1024       10          0.3049   0.3049
                   15          0.2795   0.2651
                   20          0.2769   0.2579
        4096       15          0.2689   0.2689
                   20          0.2585   0.2578
1000000 256        10          0.3344   0.3053
                   15          0.3344   0.2866
                   20          0.3344   0.2804
        1024       10          0.2866   0.2866
                   15          0.2762   0.2418
                   20          0.2762   0.2330
        4096       15          0.2335   0.2214
                   20          0.2310   0.2096
2000000 256        10          0.3834   0.3308
                   15          0.3834   0.2995
                   20          0.3834   0.2939
        1024       10          0.3122   0.3122
                   15          0.3164   0.2495
                   20          0.3164   0.2409
        4096       15          0.2442   0.2259
                   20          0.2453   0.2104
4000000 256        10          0.3589   0.3367
                   15          0.3589   0.3141
                   20          0.3589   0.3077
        1024       10          0.3112   0.3112
                   15          0.3101   0.2621
                   20          0.3101   0.2557
        4096       15          0.2582   0.2357
                   20          0.2576   0.2186

Done: 2750.117077589035 seconds

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

params_hist_dw = params_xgb_cpu.copy()
params_hist_dw.update({'tree_method': 'hist',
                       'updater'    : 'grow_fast_histmaker',
                       'grow_policy': 'depthwise',
                       'max_bin'    : 256,  #default
                   })
params_hist_lg = params_hist_dw.copy()
params_hist_lg.update({'grow_policy': 'lossguide'})

params_xgb_lst = [params_hist_dw, params_hist_lg]
model_str_lst = ['hist_dw', 'hist_lg']

params = []

stat_name_lst = ['time', 'leaf_cnts', 'w_L1', 'w_L2', 'score']
stats_dict = {name:[] for name in stat_name_lst}

n_classes = 2
n_clusters_per_class = 8
n_features = 32
n_informative = n_redundant = n_features // 4
n_rounds = 50
fname_header = "exp002_"

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
    for max_depth in [10, 15, 20]:
        for num_leaves in [256, 1024, 4096]:
            if num_leaves > 2 ** max_depth:
                continue
            fname_footer = "n_%d_md_%d_nl_%d.csv" % (n_train, max_depth, num_leaves)
            for params_xgb in params_xgb_lst:
                params_xgb.update({'max_depth':max_depth, 'max_leaves':num_leaves})
            params.append({'n_train':n_train, 'max_depth':max_depth, 'num_leaves':num_leaves})
            print('\n')
            print(params[-1])
            stats = experiment_xgb(X_train, y_train, X_valid, y_valid,
                                   params_xgb_lst, model_str_lst, n_rounds=n_rounds,
                                   fname_header=fname_header,
                                   fname_footer=fname_footer,
                                   n_skip=4)
            for name in stat_name_lst:
                stats_dict[name].append(stats[name])

keys = ['n_train', 'num_leaves', 'max_depth']

df_stats_dict = {}
for name in stat_name_lst:
    df_stats_dict[name] = pd.DataFrame(stats_dict[name], columns=model_str_lst).join(pd.DataFrame(params)).sort_values(keys).set_index(keys)
    df_stats_dict[name].to_csv('log/' + fname_header + '%s.csv' % name)

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
