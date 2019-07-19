function sigma = select_gaussian_kernel(data_name, model_lrc_ssl_rf_100)
if exist('../result/sigma_gaussian_kernel.mat', 'file') ~= 2
    M = containers.Map();
    save('../result/sigma_gaussian_kernel','M');
end
load('../result/sigma_gaussian_kernel');
if(M.isKey(data_name) == false)
    [X, y] = load_data(data_name);        
    L = construct_laplacian_graph(data_name, X, 10);
    % load datasets
    
    idx_rand = randperm(numel(y));
    idx_train = idx_rand(min(ceil(numel(y)*0.7), 1000) + 1 : min(numel(y), 3500));

    model_lrc_ssl_rf_100.n_record_batch = 1:30*32;
    model_lrc_ssl_rf_100.test_batch = true;
    min_err = 1;
    for i_sigma = 2.^(-10:5)
        X = random_fourier_features(X, 100, i_sigma);
        X_train = X(:, idx_train);
        y_train = y(idx_train);
        X_test = X(:, idx_train);
        y_test = y(idx_train);
        model_lrc_ssl_rf_100.X_test = X_test;
        model_lrc_ssl_rf_100.y_test = y_test;    
        XLX = X_train * L(idx_train, idx_train) * X_train';
        
        model_lrc_ssl_rf_100 = ps3vt_multi_train(XLX, X_train, y_train, model_lrc_ssl_rf_100);
        
        test_err = mean(model_lrc_ssl_rf_100.test_err(1, end - min(5, length(model_lrc_ssl_rf_100.test_err) - 1): end));
        if test_err < min_err
            min_err = test_err;
            min_sigma = i_sigma;
        end
    end
    M(data_name) = min_sigma;
    save('../result/sigma_gaussian_kernel','M');
    sigma = min_sigma;
else
    sigma = M(data_name);
end
end