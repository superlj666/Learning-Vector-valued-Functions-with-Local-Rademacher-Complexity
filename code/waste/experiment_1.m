initialization;
%model.n_repeats = 30;

for dataset = datasets
    model.data_name = char(dataset);
    parameter_observe(char(dataset));
    exp1_dataset(model);
end

function exp1_dataset(model)
    % load datasets
    [X, y] = load_data(model.data_name);
    sigma = select_gaussian_kernel(model.data_name);
    X_rf_100 = random_fourier_features(X, 100, sigma);
    L = construct_laplacian_graph(model.data_name, X, 10);

    [test_all_errs, run_times] = repeat_train(model, X, X_rf_100, y, L);

    test_errs = zeros(8, model.n_repeats);
    for i_repeat = 1:model.n_repeats
        for i_method = 1:8
            i_repeat_errors = test_all_errs{i_method, i_repeat};
            test_errs(i_method, i_repeat) = min(i_repeat_errors(1, end - 5: end));
        end
    end
    output(test_errs, model.data_name);

    fprintf('Dateset: %s\t Method: model_lrc_ssl\t Mean: %.4f\t STD: %.4f\n', ...
        model.data_name, mean(test_errs(1, :)), std(test_errs(1, :)));
    fprintf('Dateset: %s\t Method: model_ssl\t Mean: %.4f\t STD: %.4f\n', ...
        model.data_name, mean(test_errs(2, :)), std(test_errs(2, :)));
    fprintf('Dateset: %s\t Method: model_lrc\t Mean: %.4f\t STD: %.4f\n', ...
        model.data_name, mean(test_errs(3, :)), std(test_errs(3, :)));
    fprintf('Dateset: %s\t Method: model_linear\t Mean: %.4f\t STD: %.4f\n', ...
        model.data_name, mean(test_errs(4, :)), std(test_errs(4, :)));

    fprintf('Dateset: %s\t Method: model_lrc_ssl_100\t Mean: %.4f\t STD: %.4f\n', ...
        model.data_name, mean(test_errs(5, :)), std(test_errs(5, :)));
    fprintf('Dateset: %s\t Method: model_ssl_100\t Mean: %.4f\t STD: %.4f\n', ...
        model.data_name, mean(test_errs(6, :)), std(test_errs(6, :)));
    fprintf('Dateset: %s\t Method: model_lrc_100\t Mean: %.4f\t STD: %.4f\n', ...
        model.data_name, mean(test_errs(7, :)), std(test_errs(7, :)));
    fprintf('Dateset: %s\t Method: model_linear_100\t Mean: %.4f\t STD: %.4f\n', ...
        model.data_name, mean(test_errs(8, :)), std(test_errs(8, :)));

    errors_matrix = cell_matrix(test_all_errs);
    save(['../result/', model.data_name, '_results.mat'], ...
        'errors_matrix', 'run_times');
end

function output(errs, data_name)
    errs = errs' .* 100;

    errs_linear = errs(:, 1:4);
    [~, loc_min] = min(mean(errs_linear));
    d = errs_linear - errs_linear(:, loc_min);
    t = mean(d)./(std(d)/size(d,1));
    t(isnan(t)) = Inf;

    fid = fopen('../result/exp1/table_result_linear.txt', 'a');
    fprintf(fid, '%s\t', data_name);
    for i = 1 : size(errs_linear, 2)
        if i == loc_min
            fprintf(fid, '&\\textbf{%2.2f$\\pm$%.2f}\t', mean(errs_linear(:, i)), std(errs_linear(:, i)));
        elseif t(i) < refer_t_table(length(errs_linear(:, i))) % t statics for significantly difference
            fprintf(fid, '&\\underline{%2.2f$\\pm$%.2f}\t', mean(errs_linear(:, i)), std(errs_linear(:, i)));
        else
            fprintf(fid,'&%2.2f$\\pm$%.2f\t', mean(errs_linear(:, i)), std(errs_linear(:, i)));
        end
    end
    fprintf(fid, '\\\\\n');
    fclose(fid);


    errs_kernel = errs(:, 5:8);
    [~, loc_min] = min(mean(errs_kernel));
    d = errs_kernel - errs_kernel(:, loc_min);
    t = mean(d)./(std(d)/size(d,1));
    t(isnan(t)) = Inf;
    fid = fopen('../result/exp1/table_result_kernel.txt', 'a');
    fprintf(fid, '%s\t', data_name);
    for i = 1 : size(errs_kernel, 2)
        if i == loc_min
            fprintf(fid, '&\\textbf{%2.2f$\\pm$%.2f}\t', mean(errs_kernel(:, i)), std(errs_kernel(:, i)));
        elseif t(i) < refer_t_table(length(errs_kernel(:, i))) % t statics for significantly difference
            fprintf(fid, '&\\underline{%2.2f$\\pm$%.2f}\t', mean(errs_kernel(:, i)), std(errs_kernel(:, i)));
        else
            fprintf(fid,'&%2.2f$\\pm$%.2f\t', mean(errs_kernel(:, i)), std(errs_kernel(:, i)));
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