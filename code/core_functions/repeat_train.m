function [test_all_errs, run_times] = repeat_train(model, X, X_rf_100, y, L)
    %% Choose parameters for our method
    load(['../result/', model.data_name, '_models.mat'], 'model_linear', 'model_lrc', 'model_ssl', 'model_lrc_ssl');
    n_sample = size(y, 2);
    test_all_errs = cell(8, model.n_repeats);
    run_times = zeros(8, model.n_repeats);
    for i_repeat = 1 : model.n_repeats
        idx_rand = randperm(n_sample);
        % makse of Laplacian matrix
        idx_test = idx_rand(1:ceil(model.rate_test * n_sample));
        idx_train = setdiff(idx_rand, idx_test);
        idx_train = idx_train(randperm(numel(idx_train)));

        XLX = X(:, idx_train) * L(idx_train, idx_train) * X(:, idx_train)';
        idx_labeled = idx_train(sampling_with_labels(y(:, idx_train), model.rate_labeled));

        X_train = X(:, idx_labeled);
        X_test = X(:, idx_test);
        y_train = y(:, idx_labeled);
        y_test = y(:, idx_test);

        i_model = model;
        i_model.test_batch = true;
        i_model.X_test = X_test;
        i_model.y_test = y_test;

        t = tic();
        model_lrc_ssl = model_combination(i_model, model_lrc_ssl);
        model_lrc_ssl.T = 30;
        model_lrc_ssl = lsvv_multi_train(XLX, X_train, y_train, model_lrc_ssl);
        run_time = toc(t);
        run_times(1, i_repeat) = run_time;

        t = tic();
        model_ssl = model_combination(i_model, model_ssl);
        model_ssl = lsvv_multi_train(XLX, X_train, y_train, model_ssl);
        run_time = toc(t);
        run_times(2, i_repeat) = run_time;

        t = tic();
        model_lrc = model_combination(i_model, model_lrc);
        model_lrc = lsvv_multi_train(XLX, X_train, y_train, model_lrc);
        run_time = toc(t);
        run_times(3, i_repeat) = run_time;

        t = tic();
        model_linear = model_combination(i_model, model_linear);
        model_linear = lsvv_multi_train(XLX, X_train, y_train, model_linear);
        run_time = toc(t);
        run_times(4, i_repeat) = run_time;

        test_all_errs{1, i_repeat} = model_lrc_ssl.test_err;
        test_all_errs{2, i_repeat} = model_ssl.test_err;
        test_all_errs{3, i_repeat} = model_lrc.test_err;
        test_all_errs{4, i_repeat} = model_linear.test_err;

        clear X_train
        clear X_test
        clear XLX

        X_train_100 = X_rf_100(:, idx_labeled);
        X_test_100 = X_rf_100(:, idx_test);
        XLX_100 = X_rf_100(:, idx_train) * L(idx_train, idx_train) * X_rf_100(:, idx_train)';
        i_model.X_test = X_test_100;

        t = tic;
        model_lrc_ssl_rf_100 = model_combination(i_model, model_lrc_ssl);
        model_lrc_ssl_rf_100.T = 30;
        model_lrc_ssl_rf_100 = lsvv_multi_train(XLX_100, X_train_100, y_train, model_lrc_ssl_rf_100);
        run_time = toc(t);
        run_times(5, i_repeat) = run_time;

        t = tic();
        model_ssl_100 = model_combination(i_model, model_ssl);
        model_ssl_100 = lsvv_multi_train(XLX_100, X_train_100, y_train, model_ssl_100);
        run_time = toc(t);
        run_times(6, i_repeat) = run_time;

        t = tic();
        model_lrc_100 = model_combination(i_model, model_lrc);
        model_lrc_100 = lsvv_multi_train(XLX_100, X_train_100, y_train, model_lrc_100);
        run_time = toc(t);
        run_times(7, i_repeat) = run_time;

        t = tic();
        model_linear_100 = model_combination(i_model, model_linear);
        model_linear_100 = lsvv_multi_train(XLX_100, X_train_100, y_train, model_linear_100);
        run_time = toc(t);
        run_times(8, i_repeat) = run_time;

        test_all_errs{5, i_repeat} = model_lrc_ssl_rf_100.test_err;
        test_all_errs{6, i_repeat} = model_ssl_100.test_err;
        test_all_errs{7, i_repeat} = model_lrc_100.test_err;
        test_all_errs{8, i_repeat} = model_linear_100.test_err;
    end
end