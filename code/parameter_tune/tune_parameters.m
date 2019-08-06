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
%     if (size(X, 1) > 200)
%         X = random_fourier_features(X, 100, 16);
%     end
    n_sample = size(y, 2);

    % cross validation to choose parameters
    if(model.use_gpu)%%
        X = full(gpuArray(X));
        y = gpuArray(y);
    end
    errors_validate = cross_validation(L, X, y, model);
end