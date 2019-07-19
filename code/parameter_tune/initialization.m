%addpath('../libsvm/matlab/');
addpath('./utils/');
addpath('./core_functions/');
clear;
rng('default');

datasets = {
'iris', ...
% 'wine', ...
% 'glass', ...
% 'svmguide2', ...
% 'vowel', ...
% 'vehicle', ...
% 'dna', ...
% 'segment', ...
% 'satimage', ...
% 'usps', ...
%'pendigits', ...
% 'letter', ...
% 'protein', ...
% 'poker', ...
% 'shuttle', ...
% 'Sensorless', ...
% 'mnist', ...
% 'connect-4', ...
% 'acoustic',...
% 'covtype'
};
model.use_gpu = false;
model.n_folds = 5;
model.n_repeats = 3;
model.rate_test = 0.3;
model.rate_labeled = 0.1;
model.T = 10;
model.can_tau_I = [10.^-(3:2:7), 0];
model.can_tau_A = 10.^-(3:2:7);
model.can_tau_S = [10.^-(3:2:7), 0];

model.varepsilon = 1e-2;
model.xi = 0.5;
model.n_batch = 32;

% sigma for Gaussian kernel

