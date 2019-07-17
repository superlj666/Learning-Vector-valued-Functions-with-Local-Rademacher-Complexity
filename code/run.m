% X : D*(n+u)
% y : K*(n+u)
addpath('./utils/');
addpath('./core_functions/');
clear;
rng('default');

dataset = 'aloi';
[X, y] = load_data(char(dataset));
L = construct_laplacian_graph(char(dataset), X, 10);

total_size = size(X, 2);
train_idx = randperm(total_size, ceil(total_size*0.7));
test_idx = setdiff(1:total_size, train_idx);
X_train = X(:, train_idx);
y_train = y(:, train_idx);
X_test = X(:, test_idx);
y_test = y(:, test_idx);

XLX = X(:, train_idx)*L(train_idx, train_idx)*X(:, train_idx)';

model.data_name = char(dataset);
model.tau_I = 0;1e-4;%1e-5;
model.tau_A = 0;%1e-5;
model.tau_S = 1e-5;
model.step = 1e-3;
model.xi = 0.5;
model.n_batch = 8;
model.T = 1000;
model.n_record_batch = 100;
model.test_batch = true;
model.X_test = X_test;
model.y_test = y_test;
model = ps3vt_multi_train(XLX, X(:,train_idx(1:100)), y(:, train_idx(1:100)), model);
model