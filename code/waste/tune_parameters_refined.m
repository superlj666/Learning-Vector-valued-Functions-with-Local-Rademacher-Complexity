initialization;
model.n_folds = 3;
model.T = 30;

for dataset = datasets
    rng('default');
    model.data_name = char(dataset);
    model.task_type = check_task_type(model.data_name);
    %fprintf('cv for %s\n', char(dataset));

    % load datasets
    [X, y] = load_data(char(dataset));    
    L = construct_laplacian_graph(char(dataset), X, 10);
    n_sample = size(y, 2);

    % cross validation to choose parameters
    if(model.use_gpu)%%
        X = full(gpuArray(X));
        y = gpuArray(y);
    end
    errors_validate_linear = cross_validation_refined(L, X, y, model);
    fprintf('Dataset: %s, tau_A: %.0e, tau_I: %.0e, tau_S: %.0e, Error: %.2f\n', model.data_name, ...
        errors_validate_linear(2), errors_validate_linear(3), errors_validate_linear(4), errors_validate_linear(1)*100);
    
    X = random_fourier_features(X, 100, select_gaussian_kernel(model.data_name));
    errors_validate_rf = cross_validation_refined(L, X, y, model);    
    fprintf('Dataset: %s, tau_A: %.0e, tau_I: %.0e, tau_S: %.0e, Error: %.2f for RF\n', model.data_name, ...
        errors_validate_rf(2), errors_validate_rf(3), errors_validate_rf(4), errors_validate_rf(1)*100);

    save(['../data/', model.data_name, '/', 'cross_validation_refined.mat'], ...
        'errors_validate_linear', 'errors_validate_rf');
end

function result_matrix = cross_validation_refined(L, X_train, y_train, model)
    % rng('default');
    rate_labeled = model.rate_labeled;
    
    % data split
    n_samples = size(X_train, 2);
    idx_rand = randperm(n_samples);
    step_fold = ceil(n_samples / model.n_folds);
    folds_XLX = cell(model.n_folds, 1);
    folds_train_labeled = cell(model.n_folds, 1);
    folds_validate = cell(model.n_folds, 1);
    for i_fold = 1 : model.n_folds
        % i-th fold samples as validation data and the others as trainning data
        folds_validate{i_fold, 1} = idx_rand((i_fold - 1) * step_fold + 1:min(i_fold * step_fold, n_samples));
        i_fold_train = setdiff(idx_rand, folds_validate{i_fold, 1});
        i_fold_train = i_fold_train(randperm(numel(i_fold_train)));

        folds_XLX{i_fold, 1} = X_train(:, i_fold_train) * L(i_fold_train, i_fold_train) * X_train(:, i_fold_train)';

        % a part of i-th fold data as labeled data
        idx_labeled = sampling_with_labels(y_train(:, i_fold_train), rate_labeled);
        folds_train_labeled{i_fold, 1} = i_fold_train(idx_labeled);
        % folds_train_labeled{i_fold, 1} = i_fold_train(1 : ceil(numel(i_fold_train) * rate_labeled));
    end
    
    % error/tau_A/tau_I/tau_S
    result_matrix = ones(1,4);
    
    % choose tau_A
    for i_tau_A = model.can_tau_A
        current_error = 0;
        for i_fold = 1 : model.n_folds
            XLX = folds_XLX{i_fold, 1};

            % training
            i_model = model;
            i_model.tau_A = i_tau_A;
            i_model.tau_I = 0;
            i_model.tau_S = 0;
            i_model.test_batch = true;
            i_model.X_test = X_train(:, folds_validate{i_fold, 1});
            i_model.y_test = y_train(:, folds_validate{i_fold, 1});
            i_model = lsvv_multi_train(XLX, X_train(:, folds_train_labeled{i_fold, 1}), y_train(:, folds_train_labeled{i_fold, 1}), i_model);
            current_error = current_error + mean(i_model.test_err(min(end-5,end) : end));            
        end
        current_error = current_error / model.n_folds;
        if current_error < result_matrix(1)
            result_matrix = [current_error, i_tau_A, 0, 0];
        end
    end    
    
    % choose tau_I
    result_matrix(1) = 1;
    for i_tau_I = model.can_tau_I
        current_error = 0;
        for i_fold = 1 : model.n_folds
            XLX = folds_XLX{i_fold, 1};
            
            % training
            i_model = model;
            i_model.tau_A = result_matrix(2);
            i_model.tau_I = i_tau_I;
            i_model.tau_S = 0;
            i_model.test_batch = true;
            i_model.X_test = X_train(:, folds_validate{i_fold, 1});
            i_model.y_test = y_train(:, folds_validate{i_fold, 1});
            i_model = lsvv_multi_train(XLX, X_train(:, folds_train_labeled{i_fold, 1}), y_train(:, folds_train_labeled{i_fold, 1}), i_model);
            current_error = current_error + mean(i_model.test_err(max(end-5,end) : end));
        end
        current_error = current_error / model.n_folds;
        if current_error < result_matrix(1)
            result_matrix(1) = current_error;
            result_matrix(3) = i_tau_I;
        end
    end
        
    % choose tau_S
    result_matrix(1) = 1;
    for i_tau_S = model.can_tau_S
        current_error = 0;
        for i_fold = 1 : model.n_folds
            XLX = folds_XLX{i_fold, 1};
            
            % training
            i_model = model;
            i_model.tau_A = result_matrix(2);
            i_model.tau_I = result_matrix(3);
            i_model.tau_S = i_tau_S;
            i_model.test_batch = true;
            i_model.X_test = X_train(:, folds_validate{i_fold, 1});
            i_model.y_test = y_train(:, folds_validate{i_fold, 1});
            i_model = lsvv_multi_train(XLX, X_train(:, folds_train_labeled{i_fold, 1}), y_train(:, folds_train_labeled{i_fold, 1}), i_model);
            current_error = current_error + i_model.test_err(end);%mean(i_model.test_err(max(end-5,end) : end));
        end
        current_error = current_error / model.n_folds;
        if current_error < result_matrix(1)
            result_matrix(1) = current_error;
            result_matrix(4) = i_tau_S;
        end
    end
end