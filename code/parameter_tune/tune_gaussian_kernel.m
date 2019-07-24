initialization;

for dataset = datasets
    model.data_name = char(dataset);
    parameter_observe(char(dataset));
    choose_gaussian_kernel(char(dataset), model);
end

%% Choose gaussian kernel parameters for our method
function choose_gaussian_kernel(data_name, model)
    load(['../result/', data_name, '_models.mat'], 'model_lrc_ssl');
    if exist('../result/sigma_gaussian_kernel.mat', 'file') ~= 2
        M = containers.Map();
        save('../result/sigma_gaussian_kernel','M');
    end
    load('../result/sigma_gaussian_kernel');
    
    % load datasets
    [X, y] = load_data(data_name);
    L = construct_laplacian_graph(data_name, X, 10);
    n_samples = size(X, 2);
    min_sigma = 1;
    min_error = 1;
    for sigma = linspace(10,11,10)
        %sigma = select_gaussian_kernel(data_name, model_lrc_ssl)
        X_rf_100 = random_fourier_features(X, 100, sigma);

        test_errs = zeros(1, model.n_repeats);
        for i_repeat = 1 : model.n_repeats
            idx_rand = randperm(n_samples);
            % makse of Laplacian matrix
            idx_test = idx_rand(1:ceil(model.rate_test * n_samples));
            idx_train = setdiff(idx_rand, idx_test);
            idx_train = idx_train(randperm(numel(idx_train)));
            idx_labeled = idx_train(sampling_with_labels(y(:, idx_train), model.rate_labeled));  

            y_train = y(:, idx_labeled);
            X_test = X(:, idx_test);
            y_test = y(:, idx_test);

            i_model = model;
            %i_model.n_record_batch = 1 : floor(numel(idx_labeled) / i_model.n_batch * model.T /30) : ceil(numel(idx_labeled) / i_model.n_batch) * model.T;
            i_model.n_record_batch = 1 : floor(numel(idx_labeled) / i_model.n_batch * model.T /30) : ceil(numel(idx_labeled) / i_model.n_batch) * model.T;
            i_model.test_batch = true;
            i_model.X_test = X_test;
            i_model.y_test = y_test;

            model_lrc_ssl_rf_100 = model_combination(i_model, model_lrc_ssl);

            model_lrc_ssl_rf_100.X_test = X_rf_100(:, idx_test);
            XLX = X_rf_100(:, idx_train) * L(idx_train, idx_train) * X_rf_100(:, idx_train)';
            model_lrc_ssl_rf_100 = lsvv_multi_train(XLX, X_rf_100(:, idx_labeled), y_train, model_lrc_ssl_rf_100);

            test_errs(1, i_repeat) = mean(model_lrc_ssl_rf_100.test_err(max(end-5,end) : end));
        end
        if mean(test_errs(1, :)) < min_error
            min_error = mean(test_errs(1, :));
            min_sigma = sigma;
        end
    end
    fprintf('Dateset: %s\t Method: model_lrc_ssl_rf_100\t Error: %.4f\t Sigma: %.4f\t tau_I: %s\t tau_A: %s\t tau_S: %s\n', ...
            model.data_name, min_error, min_sigma, num2str(model_lrc_ssl_rf_100.tau_I), num2str(model_lrc_ssl_rf_100.tau_A), num2str(model_lrc_ssl_rf_100.tau_S));

    M(data_name) = min_sigma;
    save('../result/sigma_gaussian_kernel','M');
end