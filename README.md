# Learning Vector-valued Functions with Local Rademacher Complexity
## Intro
This repository provides the code used to run the experiments of the paper "Learning Vector-valued Functions with Local Rademacher Complexity" (https://arxiv.org/abs/1909.04883).
## Core functions
- lsvv_multi_train.m implements the algorithm and is used to train a model.
- record_batch.m is used to test on a batch examples.
- cross_validation.m is used to tune parameters to compare algorithms fairly.
- repeat_train.m is used to obtain significant difference between algorithms.
## Experiments
1. Download datasets for multi-class classification (https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/) and datasets for multi-label learning (http://mulan.sourceforge.net/datasets.html).
2. Run tune_parameter.m to obtain optimal parameters for tau_A, tau_I and tau_S.
3. Run tune_gaussian_kernel.m to obtain optimal Gaussian kernel for random features. (Manually and record the optimal kernel is recorded in select_gaussian_kernel).
4. Run experiment_1.m and results are stored in './result/exp1/result_table.txt'