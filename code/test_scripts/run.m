% X : D*(n+u)
% y : K*(n+u)
addpath('./utils/');
addpath('./core_functions/');

clear;
rng('default');

dataset = 'pendigits';
model.use_gpu = false;

[X, y] = load_data(char(dataset));
%X = random_fourier_features(X, 100, 1);
% X = gpuArray(X);
% tic;
% L = construct_laplacian_graph(char(dataset), X, 10);
% toc

total_size = size(X, 2);
train_idx = randperm(total_size, ceil(total_size*0.7));
test_idx = setdiff(1:total_size, train_idx);
%X = random_fourier_features(X, 500, 2);
X_train = X(:, train_idx);
y_train = y(:, train_idx);
X_test = X(:, test_idx);
y_test = y(:, test_idx);

%XLX = X(:, train_idx)*L(train_idx, train_idx)*X(:, train_idx)';
XLX = sparse(size(X,1), size(X, 1));

model.data_name = char(dataset);

model.step = 1;
model.tau_I = 0;
model.tau_A = 1e-5;
model.tau_S = 0;
model.varepsilon = 1e-2;
model.xi = 0.5;
model.n_batch = 32;
model.T = 30;

model.test_batch = true;
model.X_test = X_test;
model.y_test = y_test;
model = lsvv_multi_train(XLX, X_train, y_train, model);
mean(model.test_err(end-5:end))