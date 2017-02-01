### XGBoost experiments

name| #instances | #features | Task | #classes
--- | ---------- | --------- | ---- | --------
[Covertype Data Set](https://archive.ics.uci.edu/ml/datasets/Covertype) | 581012 | 54 | Classification | 7

Class|#Instances
---- | --------
    1| 283301
    2| 211840
    3|  35754
    4|   2747
    5|   9493
    6|  17367
    7|  20510


file name | description | data
--------- | ----------- | ----
XGBoost_refresh.ipynb | updater=refresh | Covtype

* exp001
  * model               : CPU, GPU, hist_256, hist_1024
  * objective           : Binary classification
  * metric              : Logloss
  * dataset             : make_classification
  * n_train             : 10K, 100K, 1M, 2M, 4M
  * n_valid             : n_train/4
  * n_features          : 32
  * n_rounds            : 50
  * n_clusters_per_class: 8
  * max_depth           : 5, 10, 15
* exp002
  * model               : hist_depthwise, hist_lossguide
  * objective           : Binary classification
  * metric              : Logloss
  * dataset             : make_classification
  * n_train             : 10K, 100K, 1M, 2M, 4M
  * n_valid             : n_train/4
  * n_features          : 32
  * n_rounds            : 50
  * n_clusters_per_class: 8
  * max_depth           : 10, 15, 20
  * num_leaves          : 256, 1024, 4096
