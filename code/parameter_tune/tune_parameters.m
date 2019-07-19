datasets = {'iris'};

model.n_folds = 5;
model.n_repeats = 3;
model.rate_test = 0.3;
model.rate_labeled = 0.1;
model.n_batch = 100;
model.T = 10;
model.can_tau_I = [10.^-(3:2:7), 0];
model.can_tau_A = 0; %10.^-(7:2:11);
model.can_tau_S = [10.^-(3:2:7), 0];
model.can_step = 1e-3;%10.^(3:4);

for dataset = datasets
    rng('default');
    model.data_name = char(dataset);
    fprintf('cv for %s\n', char(dataset));

    % load datasets
    [X, y] = load_data(char(dataset));    
    L = construct_laplacian_graph(char(dataset), X, 10);

    % cross validation to choose parameters
    errors_validate = cross_validation(L, X, y, model);
end