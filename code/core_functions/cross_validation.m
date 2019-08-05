function errors_validate = cross_validation(L, X_train, y_train, model)
% rng('default');
n_folds = model.n_folds;
rate_labeled = model.rate_labeled;
data_name = model.data_name;
can_tau_I = model.can_tau_I;
can_tau_A = model.can_tau_A;
can_tau_S = model.can_tau_S;

% data split
n_samples = size(X_train, 2);
idx_rand = randperm(n_samples);
step_fold = ceil(n_samples / n_folds);
folds_XLX = cell(n_folds, 1);
folds_train_labeled = cell(n_folds, 1);
folds_validate = cell(n_folds, 1);
for i_fold = 1 : n_folds
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

% choose the best parameters
counter = 1;
errors_validate = cell(numel(can_tau_I) * numel(can_tau_A) * numel(can_tau_S), 2);
for para_I = can_tau_I
    for para_A = can_tau_A
        for para_S = can_tau_S
            model.tau_I = para_I;
            model.tau_A = para_A;
            model.tau_S = para_S;
            
            test_errs = zeros(n_folds, 1);
            for i_fold = 1 : n_folds
                XLX = folds_XLX{i_fold, 1};
                
                % training
                i_model = model;
                n_sample = length(folds_train_labeled{i_fold, 1});
                
                i_model.test_batch = true;
                i_model.X_test = X_train(:, folds_validate{i_fold, 1});
                i_model.y_test = y_train(:, folds_validate{i_fold, 1});
                i_model = lsvv_multi_train(XLX, X_train(:, folds_train_labeled{i_fold, 1}), y_train(:, folds_train_labeled{i_fold, 1}), i_model);
                test_errs(i_fold) = mean(i_model.test_err(max(end-5,end) : end));
            end
            
            fprintf('Grid: %.0f/%.0f\t ERR: %.4f\t tau_I: %s\t tau_A: %s\t tau_S: %s\n', ...
                counter, numel(can_tau_I) * numel(can_tau_A) * numel(can_tau_S), ...
                mean(test_errs), num2str(para_I), num2str(para_A), num2str(para_S))
            errors_validate{counter, 1} = mean(test_errs);
            errors_validate{counter, 2} = [para_I, para_A, para_S];
            counter = counter + 1;
        end
    end
end

[~, loc_best] = min([errors_validate{:, 1}]);
[d2, d3, d4] = ind2sub([numel(can_tau_S), numel(can_tau_A), numel(can_tau_I)], loc_best);
fprintf('-----Best ERR: %.4f\t tau_I: %s\t tau_A: %s\t tau_S: %s\n-----\n', ...
    errors_validate{loc_best, 1}, num2str(can_tau_I(d4)), num2str(can_tau_A(d3)), num2str(can_tau_S(d2)));
save(['../data/', data_name, '/', 'cross_validation.mat']);

% cv_results = reshape(errors_validate, [numel(can_tau_S), numel(can_tau_A), numel(can_tau_I)]);
end
