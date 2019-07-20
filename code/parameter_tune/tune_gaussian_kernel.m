initialization;
model.T = 30;

for dataset = datasets
    model.data_name = char(dataset);
    parameter_observe(char(dataset));
    choose_gaussian_kernel(char(dataset), model);
end

%% Choose gaussian kernel parameters for our method
function choose_gaussian_kernel(data_name, model)
    load(['../result/', data_name, '_models.mat'], 'model_lrc_ssl');

    % load datasets
    [X, y] = load_data(data_name);
    L = construct_laplacian_graph(data_name, X, 10);
    for sigma = 16:2:32
        %sigma = select_gaussian_kernel(data_name, model_lrc_ssl)
        X_rf_100 = random_fourier_features(X, 100, sigma);

        test_errs = zeros(1, model.n_repeats);
        for i_repeat = 1 : model.n_repeats
            idx_rand = randperm(numel(y));
            % makse of Laplacian matrix
            idx_test = idx_rand(1:ceil(model.rate_test * numel(y)));
            idx_train = setdiff(idx_rand, idx_test);
            idx_train = idx_train(randperm(numel(idx_train)));
            idx_labeled = idx_train(sampling_with_labels(y(idx_train), model.rate_labeled));  

            y_train = y(idx_labeled);
            X_test = X(:, idx_test);
            y_test = y(idx_test);

            i_model = model;
            %i_model.n_record_batch = 1 : floor(numel(idx_labeled) / i_model.n_batch * model.T /30) : ceil(numel(idx_labeled) / i_model.n_batch) * model.T;
            i_model.n_record_batch = 1 : floor(numel(idx_labeled) / i_model.n_batch * model.T /30) : ceil(numel(idx_labeled) / i_model.n_batch) * model.T;
            i_model.test_batch = true;
            i_model.X_test = X_test;
            i_model.y_test = y_test;

            model_lrc_ssl_rf_100 = model_combination(i_model, model_lrc_ssl);

            model_lrc_ssl_rf_100.X_test = X_rf_100(:, idx_test);
            XLX = X_rf_100(:, idx_train) * L(idx_train, idx_train) * X_rf_100(:, idx_train)';
            model_lrc_ssl_rf_100 = ps3vt_multi_train(XLX, X_rf_100(:, idx_labeled), y_train, model_lrc_ssl_rf_100);

            test_errs(1, i_repeat) = min(model_lrc_ssl_rf_100.test_err);
        end

        fprintf('Dateset: %s\t Method: model_lrc_ssl_rf_100\t Mean: %.4f\t STD: %.4f\t tau_I: %s\t tau_A: %s\t tau_S: %s\n', ...
            model.data_name, mean(test_errs(1, :)), std(test_errs(1, :)), num2str(model_lrc_ssl.tau_I), num2str(model_lrc_ssl.tau_A), num2str(model_lrc_ssl.tau_S));
    end
end