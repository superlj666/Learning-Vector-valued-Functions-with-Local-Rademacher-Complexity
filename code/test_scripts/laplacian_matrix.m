addpath('./utils/');
addpath('./core_functions/');
clear;
rng('default');

dataset = 'letter';
[X, y] = load_data(char(dataset));

% CPU
% tic;
% L = construct_laplacian_graph(char(dataset), X, 10);
% toc

% GPU
tic();
X = gpuArray(full(X));
L = construct_laplacian_graph(char(dataset), X, 10);
L = gather(L);
toc;