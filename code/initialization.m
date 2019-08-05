%addpath('../libsvm/matlab/');
addpath('./parameter_tune/');
addpath('./utils/*');
addpath('./core_functions/');
clear;
rng('default');

datasets = {
<<<<<<< HEAD
%'iris', ...
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
% 'covtype', ...
% 'scene', ...
% 'yeast', ...  
%'corel5k', ...
% 'bibtex', ...
%'rf2', ...
% 'scm1d', ...
};
model.use_gpu = false;
model.n_folds = 5;
model.n_repeats = 10;
=======
'iris', ...
'wine', ...
'glass', ...
'svmguide2', ...
'vowel', ...
'vehicle', ...
'dna', ...
'segment', ...
'satimage', ...
'usps', ...
'pendigits', ...
'letter', ...
'protein', ...
'poker', ...
'shuttle', ...
'Sensorless', ...
'mnist', ...
'connect-4', ...
'acoustic',...
'covtype', ...
'scene', ...
'yeast', ...  
'corel5k', ...
'bibtex', ...
'rf2', ...
'scm1d', ...
};
model.use_gpu = false;
model.n_folds = 5;
model.n_repeats = 30;
>>>>>>> edb2c6899f1c5a3c04669cc15f3538702ffb8018
model.rate_test = 0.3;
model.rate_labeled = 0.1;
model.T = 10;

model.can_tau_I = [10.^-(5:2:15), 0];
model.can_tau_A = 10.^-(3:2:9);
model.can_tau_S = [10.^-(5:2:15), 0];

model.varepsilon = 1;
model.xi = 0.5;
model.n_batch = 32;

% rand_paras = random_parameters(model, 4);
% 
% function rand_paras = random_parameters(model, times)
%     comb_matrix = allcomb(model.can_tau_I, model.can_tau_A, model.can_tau_S);
%     rand_paras = comb_matrix(randperm(length(comb_matrix), times),:,:);
% end
