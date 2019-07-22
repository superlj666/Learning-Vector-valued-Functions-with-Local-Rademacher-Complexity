initialization;
model.T = 30;
model.n_repeats = 10;

for dataset = datasets
    model.data_name = char(dataset);
    parameter_observe(char(dataset));
    exp1_dataset(char(dataset), model);
end

function exp1_dataset(data_name, model)
%% Choose parameters for our method
load(['../result/', data_name, '_models.mat'], 'model_linear', 'model_lrc', 'model_ssl', 'model_lrc_ssl');

% load datasets
[X, y] = load_data(data_name);
sigma = select_gaussian_kernel(data_name);
X_rf_100 = random_fourier_features(X, 100, sigma);
X_rf_500 = random_fourier_features(X, 500, sigma);
L = construct_laplacian_graph(data_name, X, 10);

test_all_errs = cell(6, model.n_repeats);
test_errs = zeros(6, model.n_repeats);
for i_repeat = 1 : model.n_repeats
    idx_rand = randperm(numel(y));
    % makse of Laplacian matrix
    idx_test = idx_rand(1:ceil(model.rate_test * numel(y)));
    idx_train = setdiff(idx_rand, idx_test);
    idx_train = idx_train(randperm(numel(idx_train)));

    XLX = X(:, idx_train) * L(idx_train, idx_train) * X(:, idx_train)';
    idx_labeled = idx_train(sampling_with_labels(y(idx_train), model.rate_labeled));  

    X_train = X(:, idx_labeled); 
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
    model_lrc_ssl_rf_500 = model_combination(i_model, model_lrc_ssl);
    model_lrc_ssl = model_combination(i_model, model_lrc_ssl);
    model_ssl = model_combination(i_model, model_ssl);
    model_lrc = model_combination(i_model, model_lrc);
    model_linear = model_combination(i_model, model_linear);
    
    model_lrc_ssl = lsvv_multi_train(XLX, X_train, y_train, model_lrc_ssl);
    model_ssl = lsvv_multi_train(XLX, X_train, y_train, model_ssl);
    model_lrc = lsvv_multi_train(XLX, X_train, y_train, model_lrc);
    model_linear = lsvv_multi_train(XLX, X_train, y_train, model_linear);
    
    model_lrc_ssl_rf_100.X_test = X_rf_100(:, idx_test);
    XLX = X_rf_100(:, idx_train) * L(idx_train, idx_train) * X_rf_100(:, idx_train)';
    model_lrc_ssl_rf_100 = lsvv_multi_train(XLX, X_rf_100(:, idx_labeled), y_train, model_lrc_ssl_rf_100);
    
    model_lrc_ssl_rf_500.X_test = X_rf_500(:, idx_test);
    XLX = X_rf_500(:, idx_train) * L(idx_train, idx_train) * X_rf_500(:, idx_train)';
    model_lrc_ssl_rf_500 = lsvv_multi_train(XLX, X_rf_500(:, idx_labeled), y_train, model_lrc_ssl_rf_500);
    
    test_all_errs{1, i_repeat} = model_lrc_ssl.test_err;
    test_all_errs{2, i_repeat} = model_ssl.test_err;
    test_all_errs{3, i_repeat} = model_lrc.test_err;
    test_all_errs{4, i_repeat} = model_linear.test_err;
    test_all_errs{5, i_repeat} = model_lrc_ssl_rf_100.test_err;
    test_all_errs{6, i_repeat} = model_lrc_ssl_rf_500.test_err;
    
    test_errs(1, i_repeat) = mean(model_lrc_ssl.test_err(1, end - min(5, length(model_lrc_ssl.test_err) - 1): end));
    test_errs(2, i_repeat) = mean(model_ssl.test_err(1, end - min(5, length(model_ssl.test_err) - 1): end));
    test_errs(3, i_repeat) = mean(model_lrc.test_err(1, end - min(5, length(model_lrc.test_err) - 1): end));
    test_errs(4, i_repeat) = mean(model_linear.test_err(1, end - min(5, length(model_linear.test_err) - 1): end));
    test_errs(5, i_repeat) = mean(model_lrc_ssl_rf_100.test_err(1, end - min(5, length(model_linear.test_err) - 1): end));
    test_errs(6, i_repeat) = mean(model_lrc_ssl_rf_500.test_err(1, end - min(5, length(model_linear.test_err) - 1): end));
end

output(test_errs, data_name);

fprintf('Dateset: %s\t Method: model_lrc_ssl\t Mean: %.4f\t STD: %.4f\t tau_I: %s\t tau_A: %s\t tau_S: %s\n', ...
    model.data_name, mean(test_errs(1, :)), std(test_errs(1, :)), num2str(model_lrc_ssl.tau_I), num2str(model_lrc_ssl.tau_A), num2str(model_lrc_ssl.tau_S));
fprintf('Dateset: %s\t Method: model_ssl\t Mean: %.4f\t STD: %.4f\t tau_I: %s\t tau_A: %s\t tau_S:  %s\n', ...
    model.data_name, mean(test_errs(2, :)), std(test_errs(2, :)), num2str(model_ssl.tau_I), num2str(model_ssl.tau_A), num2str(model_ssl.tau_S));
fprintf('Dateset: %s\t Method: model_lrc\t Mean: %.4f\t STD: %.4f\t tau_I: %s\t tau_A: %s\t tau_S:  %s\n', ...
    model.data_name, mean(test_errs(3, :)), std(test_errs(3, :)), num2str(model_lrc.tau_I), num2str(model_lrc.tau_A), num2str(model_lrc.tau_S));
fprintf('Dateset: %s\t Method: model_linear\t Mean: %.4f\t STD: %.4f\t tau_I: %s\t tau_A: %s\t tau_S:  %s\n', ...
    model.data_name, mean(test_errs(4, :)), std(test_errs(4, :)), num2str(model_linear.tau_I), num2str(model_linear.tau_A), num2str(model_linear.tau_S));
fprintf('Dateset: %s\t Method: random features 100\t Mean: %.4f\t STD: %.4f\t tau_I: %s\t tau_A: %s\t tau_S:  %s\n', ...
    model.data_name, mean(test_errs(5, :)), std(test_errs(5, :)), num2str(model_lrc_ssl_rf_100.tau_I), num2str(model_lrc_ssl_rf_100.tau_A), num2str(model_lrc_ssl_rf_100.tau_S));
fprintf('Dateset: %s\t Method: random features 500\t Mean: %.4f\t STD: %.4f\t tau_I: %s\t tau_A: %s\t tau_S:  %s\n', ...
    model.data_name, mean(test_errs(6, :)), std(test_errs(6, :)), num2str(model_lrc_ssl_rf_500.tau_I), num2str(model_lrc_ssl_rf_500.tau_A), num2str(model_lrc_ssl_rf_500.tau_S));


errors_matrix = cell_matrix(test_all_errs);
save(['../result/', data_name, '_results.mat'], ...
    'errors_matrix');
end

function output(errs, data_name)
    errs = errs' .* 100;

    [~, loc_min] = min(mean(errs));
    d = errs - errs(:, loc_min);
    t = mean(d)./(std(d)/size(d,1));
    t(isnan(t)) = Inf;
    % if bigger than 1.676, it is significantly better one.

    fid = fopen('../result/exp1/table_result.txt', 'a');
    fprintf(fid, '%s\t', data_name);
    [X, y] = load_data(data_name);
    if strcmp(check_task_type(data_name), 'mc')
        fprintf(fid, '&%d\t', ceil(numel(y)*0.1*0.7));
        fprintf(fid, '&%d\t', ceil(numel(y)*0.9*0.7));
        fprintf(fid, '&%d\t', size(X, 1));
        fprintf(fid, '&%d\t', max(y));
    end
    clear X;
    clear y;
    
    for i = 1 : size(errs, 2)
        if i == loc_min
            fprintf(fid, '&\\textbf{%2.2f$\\pm$%.2f}\t', mean(errs(:, i)), std(errs(:, i)));
        elseif t(i) < 1.676
            fprintf(fid, '&\\underline{%2.2f$\\pm$%.2f}\t', mean(errs(:, i)), std(errs(:, i)));
        else
            fprintf(fid,'&%2.2f$\\pm$%.2f\t', mean(errs(:, i)), std(errs(:, i)));
        end
    end
    fprintf(fid, '\\\\\n');
    fclose(fid);
end

function errors_matrix = cell_matrix(errors_cell)
    length_max = -Inf;
    for i_row = 1 : size(errors_cell, 1)
        for i_column = 1 : size(errors_cell, 2)
            length_max = max(length_max, numel(errors_cell{i_row, i_column}));
        end
    end
    errors_matrix = zeros(size(errors_cell, 1), size(errors_cell, 2), length_max);
    for i_row = 1 : size(errors_cell, 1)
        for i_column = 1 : size(errors_cell, 2)
            length_cur = length(errors_cell{i_row, i_column});
            errors_matrix(i_row, i_column, 1 : length_cur) = errors_cell{i_row, i_column};
            if length_cur < length_max
                errors_matrix(i_row, i_column, length_cur + 1 : length_max) = errors_matrix(i_row, i_column, length_cur);
            end
        end
    end
end