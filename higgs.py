import time
import numpy as np
import scipy as sp
import pandas as pd
import xgboost as xgb
import re

def get_leaf_values(tree_str):
    # To find 'leaf=0.123\n'
    prog=re.compile(r"(?<=leaf\=)(.+)\n")
    result = [float(rval) for rval in prog.findall(tree_str)]
    return np.array(result)

def get_all_leaves(bst):
    dmp = bst.get_dump()
    return [get_leaf_values(tree) for tree in dmp]

# split -l 10000000 HIGGS.csv out
# mv outaa binary.train
# mv outab binary.test
# split -l 7500000 binary.train out
# mv outaa binary.train750
# mv outab binary.valid250

data = pd.read_csv('/home/tks/download/higgs/binary.train', header=None)
data_test = pd.read_csv('/home/tks/download/higgs/binary.test', header=None)

y = data[0].values
data.drop(0, axis=1, inplace=True)
y_test = data_test[0].values
data_test.drop(0, axis=1, inplace=True)

dtrain = xgb.DMatrix(data.values, label = y)
dtest = xgb.DMatrix(data_test.values, label = y_test)


param = {'objective':'binary:logistic', 'tree_method':'exact',
         'eta':.1, 'max_depth':8, 'min_child_weight':100,
         'nthread':8, 'seed':123, 'silent':1}

# 2016/10/17
# #train=10M
param = {'objective':'binary:logistic', 'tree_method':'approx', 'sketch_eps':0.004,
         'eta':.1, 'max_depth':8, 'eval_metric':'auc', 
         'nthread':8, 'seed':123, 'silent':1}

n_rounds=500
n_nodes = []
scores = []
for mc in [1, 100]:
    t0 = time.time()
    param['min_child_weight'] = mc
    bst = xgb.train(param, dtrain, n_rounds, [(dtrain, 'train'), (dtest, 'test')])
    tmp = get_all_leaves(bst)
    n_nodes.append([len(s) for s in tmp])
    scores.append({'min_c':mc, 'time':time.time() - t0,
                   'total_leaves':np.sum(n_nodes[-1])})
    print(scores[-1])


[499]	train-auc:0.846833	test-auc:0.840352    
[499]	train-auc:0.845507	test-auc:0.840589

pd.DataFrame(scores)
   min_c         time  total_leaves
0      1  3655.648553        120937
1    100  3628.614431         91478


# r004
# 2016/10/19 5.2h
data_train = pd.read_csv('/home/tks/download/higgs/binary.train750', header=None)
data_valid = pd.read_csv('/home/tks/download/higgs/binary.valid250', header=None)

y_train = data_train[0].values
data_train.drop(0, axis=1, inplace=True)
y_valid = data_valid[0].values
data_valid.drop(0, axis=1, inplace=True)

dtrain = xgb.DMatrix(data_train.values, label = y_train)
dvalid = xgb.DMatrix(data_valid.values, label = y_valid)

param = {'objective':'binary:logistic','tree_method':'approx', 'sketch_eps':0.004,
         'eta':.1, 'min_child_weight':100, 'lambda':0, 'eval_metric':'auc',
         'nthread':8, 'seed':123, 'silent':1}

n_rounds=500

n_nodes = []
scores = []
result = []
for max_depth in [8, 9, 10, 11, 12]:
    t0 = time.time()
    evals_result = {}
    param['max_depth'] = max_depth
    bst = xgb.train(param, dtrain, n_rounds, [(dtrain, 'train'), (dvalid, 'valid')],
                    evals_result=evals_result)
    tmp = get_all_leaves(bst)
    n_nodes.append([len(s) for s in tmp])
    scores.append({'max_depth':max_depth, 'total_leaves':np.sum(n_nodes[-1]),
                   'time':time.time() - t0})
    result.append(evals_result)
    print(scores[-1])

df_train = pd.DataFrame({i+8:result[i]['train']['auc'] for i in range(5)})
df_valid = pd.DataFrame({i+8:result[i]['valid']['auc'] for i in range(5)})
df_leaf_cnt = pd.DataFrame({i+8:n_nodes[i] for i in range(5)})

df_train.to_csv('log/r004_train.csv')
df_valid.to_csv('log/r004_valid.csv')
df_leaf_cnt.to_csv('log/r004_leaf_cnt.csv')

print(df_valid.tail(5))
           8         9         10        11        12
495  0.838657  0.842426  0.846096  0.848043  0.850070
496  0.838678  0.842448  0.846115  0.848057  0.850082
497  0.838705  0.842453  0.846135  0.848068  0.850089
498  0.838722  0.842472  0.846140  0.848096  0.850113
499  0.838746  0.842518  0.846142  0.848101  0.850116

print(pd.DataFrame(scores))
    max_depth         time  total_leaves
0          8  2724.755577         85618
1          9  3174.557975        139994
2         10  3698.960060        222218
3         11  4207.922560        319139
4         12  4858.091027        447266

# r005
# 2016/10/21 7.5m
data_train = pd.read_csv('/home/tks/download/higgs/binary.train750', header=None)
data_valid = pd.read_csv('/home/tks/download/higgs/binary.valid250', header=None)

y_train = data_train[0].values
data_train.drop(0, axis=1, inplace=True)
y_valid = data_valid[0].values
data_valid.drop(0, axis=1, inplace=True)

dtrain = xgb.DMatrix(data_train.values, label = y_train)
dvalid = xgb.DMatrix(data_valid.values, label = y_valid)

param = {'objective':'binary:logistic','tree_method':'approx', 'sketch_eps':0.00392,
         'eta':.1, 'max_depth':1000, 'lambda':0, 'eval_metric':'auc',
         'nthread':8, 'seed':123, 'silent':1}

n_rounds=10

n_nodes = []
scores = []
result = []
for mc in [1000, 2000, 4000]:
    t0 = time.time()
    evals_result = {}
    param['min_child_weight'] = mc
    bst = xgb.train(param, dtrain, n_rounds, [(dtrain, 'train'), (dvalid, 'valid')],
                    evals_result=evals_result)
    tmp = get_all_leaves(bst)
    n_nodes.append([len(s) for s in tmp])
    scores.append({'min_child_weight':mc, 'total_leaves':np.sum(n_nodes[-1]),
                   'time':time.time() - t0})
    result.append(evals_result)
    print(scores[-1])

df = pd.DataFrame(scores)
df_leaf_cnt = pd.DataFrame({2**i*1000:n_nodes[i] for i in range(3)})

df.to_csv('log/r005.csv')
df_leaf_cnt.to_csv('log/r005_leaf_cnt.csv')

print(df_leaf_cnt)
   1000  2000  4000
0  1464   733   368
1  1442   724   364
2  1444   721   363
3  1434   715   365
4  1410   714   359
5  1394   713   357
6  1374   702   345
7  1362   688   343
8  1348   671   342
9  1311   665   335

print(df)
   min_child_weight        time  total_leaves
0              1000  179.808144         13983
1              2000  153.772096          7046
2              4000  116.155873          3541

# r006
# 'min_child_weight':1000
# 2016/10/21 3.4h
data_train = pd.read_csv('/home/tks/download/higgs/binary.train750', header=None)
data_valid = pd.read_csv('/home/tks/download/higgs/binary.valid250', header=None)

y_train = data_train[0].values
data_train.drop(0, axis=1, inplace=True)
y_valid = data_valid[0].values
data_valid.drop(0, axis=1, inplace=True)

dtrain = xgb.DMatrix(data_train.values, label = y_train)
dvalid = xgb.DMatrix(data_valid.values, label = y_valid)

param = {'objective':'binary:logistic','tree_method':'approx', 'sketch_eps':0.00392,
         'eta':.1, 'min_child_weight':1000, 'max_depth':1000, 'lambda':0,
         'eval_metric':['logloss','auc'],
         'nthread':8, 'seed':123, 'silent':1}

n_rounds=500

n_nodes = []
scores = []
t0 = time.time()
evals_result = {}
bst = xgb.train(param, dtrain, n_rounds, [(dtrain, 'train'), (dvalid, 'valid')],
                evals_result=evals_result)
tmp = get_all_leaves(bst)
n_nodes.append([len(s) for s in tmp])
scores.append({'min_child_weight':mc, 'total_leaves':np.sum(n_nodes[-1]),
               'time':time.time() - t0})
print(scores[-1])

df = pd.DataFrame(scores)
df_auc_loss = pd.DataFrame({'auc_train':evals_result['train']['auc'],
                            'auc_valid':evals_result['valid']['auc'],
                            'loss_train':evals_result['train']['logloss'],
                            'loss_valid':evals_result['valid']['logloss'],
                            'leaf_cnt':n_nodes[0]})

   min_child_weight          time  total_leaves
0              4000  12198.695484        476780

df_auc_loss.tail(10)
     auc_train  auc_valid  leaf_cnt  loss_train  loss_valid
490   0.881540   0.852546       890    0.433844    0.472947
491   0.881603   0.852561       907    0.433757    0.472926
492   0.881663   0.852573       892    0.433671    0.472909
493   0.881726   0.852588       894    0.433581    0.472889
494   0.881788   0.852603       905    0.433493    0.472869
495   0.881846   0.852610       909    0.433415    0.472859
496   0.881901   0.852613       899    0.433338    0.472854
497   0.881962   0.852626       903    0.433247    0.472834
498   0.882019   0.852633       899    0.433165    0.472824
499   0.882082   0.852651       889    0.433071    0.472797

df.to_csv('log/r006.csv')
df_auc_loss.to_csv('log/r006_auc_loss.csv')


#

