# Multi-Class Learning using Unlabeled Samples: Theory and Algorithm
Code for experiments in "Multi-Class Learning using Unlabeled Samples: Theory and Algorithm".
The paper has been accepted by IJCAI-19.

## Structure
- ./data/: Store parameter tuning results.
- ./datasets/: Store primal libsvm style datasets. All datasets are available in [LibSVM Data](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/).
- ./result/: Store final results for experiments.
- ./libsvm/: Libvm tools used in codes, including lisvmread, which is provided in [libsvm](https://github.com/cjlin1/libsvm).
- ./code/core_functions: Core functions used in expereiments, including the proposed algorithm, cross validition and so on.
- ./code/utils/: Common utils used in experiments.
- ./code/tune_parameters.m: Tune optimal parameters set and save them.
- ./code/parameter_observe.m: Load parameter tuning results and choose the best one.
- ./code/experiment_*.m: Scripts for experiments.

## Steps
Just run experiment_*.m individually.

## 