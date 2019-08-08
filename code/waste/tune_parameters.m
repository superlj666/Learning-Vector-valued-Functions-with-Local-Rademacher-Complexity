initialization;
model.n_folds = 3;
model.T = 20;

for dataset = datasets
    rng('default');
    model.data_name = char(dataset);
    model.task_type = check_task_type(model.data_name);
    fprintf('cv for %s\n', char(dataset));

    % load datasets
    [X, y] = load_data(char(dataset));    
    L = construct_laplacian_graph(char(dataset), X, 10);
    n_sample = size(y, 2);

    % cross validation to choose parameters
    if(model.use_gpu)%%
        X = full(gpuArray(X));
        y = gpuArray(y);
    end
    errors_validate_linear = cross_validation(L, X, y, model);
    
    X = random_fourier_features(X, 100, select_gaussian_kernel(model.data_name));
    errors_validate_rf = cross_validation(L, X, y, model);

    can_tau_I = model.can_tau_I;
    can_tau_A = model.can_tau_A;
    can_tau_S = model.can_tau_S;
    save(['../data/', model.data_name, '/', 'cross_validation.mat'], ...
        'errors_validate_linear', 'errors_validate_rf', ...
        'can_tau_S', 'can_tau_A', 'can_tau_I');
end