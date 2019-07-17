function repeat_errors = repeat_test(model, model_name, X, y, L)    
    % rng('default');

    test_models = cell(model.n_repeats, 1);
    for i_repeat = 1 : model.n_repeats
        idx_rand = randperm(numel(y));
        % take use of Laplacian matrix
        idx_test = idx_rand(1:ceil(model.rate_test * numel(y)));
        idx_train = setdiff(idx_rand, idx_test);
        idx_train = idx_train(randperm(numel(idx_train)));

        XLX = X(idx_train, :)' * L(idx_train, idx_train) * X(idx_train, :);

        %idx_labeled = idx_train(1 : ceil(numel(idx_train) * model.rate_labeled));
        idx_labeled = idx_train(sampling_with_labels(y(idx_train), model.rate_labeled));

        % record training and testing
        i_model = model;
        i_model.n_record_batch = (ceil(numel(idx_labeled) / i_model.n_batch) * model.T - 99) : ceil(numel(idx_labeled) / i_model.n_batch) * model.T;
        i_model.test_batch = true;
        i_model.X_test = X(idx_test, :);
        i_model.y_test = y(idx_test);
        i_model = ps3vt_multi_train(XLX, X(idx_labeled, :), y(idx_labeled), i_model);

        test_models{i_repeat, 1} = i_model.test_err;
    end
    
    repeat_errors = cell2mat(test_models);
    test_errs = mean(repeat_errors(:,end-4:end), 2);
    fprintf('Dateset: %s\t Method: %s\t Mean: %.4f\t STD: %.4f\t tau_I: %s\t tau_A: %s\t tau_S: %s\t\n', ... 
        model.data_name, model_name, mean(test_errs), std(test_errs), num2str(model.tau_I), num2str(model.tau_A), num2str(model.tau_S));
end