initialization;

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