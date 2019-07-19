% X : D*(n+u)
% y : K*(n+u)
addpath('./utils/');
addpath('./core_functions/');

clear;
rng('default');

dataset = 'aloi';
[X, y] = load_data(char(dataset));


ggg = gcp();
ggg.IdleTimeout = Inf;
clear ggg;
X = distributed(X);
spmd
    X = redistribute(X, codistributor1d(1));
end

tic;
L = construct_laplacian_graph(char(dataset), X, 10);
toc