addpath('./utils/');
addpath('./core_functions/');

% CPU
tic();
X = rand(3e4, 100);
f = gaussianKernel(2);
K = f(X,X);
toc;

% GPU
tic();
X = rand(3e4, 100);
X = gpuArray(single(X));
f = gaussianKernel(2);
K = f(X,X);
K = gather(K);
toc;