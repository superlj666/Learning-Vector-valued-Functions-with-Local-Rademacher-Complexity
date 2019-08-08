initialization;
model.n_repeats = 10;
model.T = 30;
model.can_tau_A = 10.^-(3:9);
model.can_tau_I = 10.^-(1:10);%[10.^-(1:10), 0];
model.can_tau_S = 10.^-(1:10);%[10.^-(1:10), 0];
model.tau_A = 0;
model.tau_I = 0;
model.tau_S = 0;

model.varepsilon = 1; % 1 for linear, 10 for RF
model.xi = 0.5;

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
    [optimal_paras_linear, test_all_errs_linear, run_times_linear] = tune_parameters_refined(L, X, y, model);
    
    X_rf_100 = random_fourier_features(X, 100, select_gaussian_kernel(model.data_name));
    [optimal_paras_kernel, test_all_errs_kernel, run_times_kernel] = tune_parameters_refined(L, X_rf_100, y, model);

    save(['../result/exp1_new/', model.data_name, '_model.mat']);

    test_errs = [test_all_errs_linear; test_all_errs_kernel];
    output(test_errs, model.data_name);
end

function output(errs, data_name)
    errs = errs'.* 100;

    errs_linear = errs(:, 1:4);
    [~, loc_min] = min(mean(errs_linear));
    d = errs_linear - errs_linear(:, loc_min);
    t = mean(d)./(std(d)/size(d,1));
    t(isnan(t)) = Inf;

    fid = fopen('../result/exp1_new/table_result_linear.txt', 'a');
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
    fid = fopen('../result/exp1_new/table_result_kernel.txt', 'a');
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