initialization;
model.n_repeats = 3;
model.T = 10;

for dataset = datasets    
    rng('default');
    model.data_name = char(dataset);
    parameter_observe(model.data_name)
    exp3_run(model);
end

function exp3_run(model)
    % load datasets
    [X, y] = load_data(model.data_name);
    sigma = select_gaussian_kernel(model.data_name);
    X_rf_100 = random_fourier_features(X, 100, sigma);
    L = construct_laplacian_graph(model.data_name, X, 10);
    
    can_labeled = 0.1 : 0.1 : 1;
    errors_labeled = zeros(2, numel(can_labeled));
    test_all_errs = cell(2, model.n_repeats, numel(can_labeled));
    run_times = zeros(2, model.n_repeats);
    
    load(['../result/', model.data_name, '_models.mat'], 'model_linear', 'model_lrc', 'model_ssl', 'model_lrc_ssl');
    n_sample = size(y, 2);
    idx_rand = randperm(n_sample);
    for i_labeled = 1 : numel(can_labeled)
        model.rate_labeled = can_labeled(i_labeled);
        for i_repeat = 1 : model.n_repeats
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

            X_train_100 = X_rf_100(:, idx_labeled);
            X_test_100 = X_rf_100(:, idx_test);
            XLX_100 = X_rf_100(:, idx_train) * L(idx_train, idx_train) * X_rf_100(:, idx_train)';

            i_model.X_test = X_test;

            t = tic();
            model_lrc_ssl = model_combination(i_model, model_lrc_ssl);
            model_lrc_ssl = lsvv_multi_train(XLX, X_train, y_train, model_lrc_ssl);
            run_time = toc(t);
            run_times(1, i_repeat) = run_time;
            test_all_errs{1, i_repeat, i_labeled} = model_lrc_ssl.test_err;

            t = tic();
            model_lrc = model_combination(i_model, model_lrc);
            model_lrc = lsvv_multi_train(XLX, X_train, y_train, model_lrc);
            run_time = toc(t);
            run_times(2, i_repeat) = run_time;
            test_all_errs{2, i_repeat, i_labeled} = model_lrc.test_err;
            
            i_model.X_test = X_test_100;
            t = tic;
            model_lrc_ssl_rf_100 = model_combination(i_model, model_lrc_ssl);
            model_lrc_ssl_rf_100 = lsvv_multi_train(XLX_100, X_train_100, y_train, model_lrc_ssl_rf_100);
            run_time = toc(t);
            run_times(3, i_repeat) = run_time;            
            test_all_errs{3, i_repeat, i_labeled} = model_lrc_ssl_rf_100.test_err;

            t = tic;
            model_lrc_100 = model_combination(i_model, model_lrc);
            model_lrc_100 = lsvv_multi_train(XLX_100, X_train_100, y_train, model_lrc_100);
            run_time = toc(t);
            run_times(4, i_repeat) = run_time;            
            test_all_errs{4, i_repeat, i_labeled} = model_lrc_100.test_err;
       end
    end
     
    for i_method = 1:2
        for i_labeled = 1 : numel(can_labeled)   
            errors_repeat = zeros(1, model.n_repeats);
            for i_repeat = 1:model.n_repeats
                i_repeat_errors = test_all_errs{i_method, i_repeat, i_labeled};
                errors_repeat(i_repeat) = min(i_repeat_errors(1, end - 5: end));
            end
            errors_labeled(i_method, i_labeled) = mean(errors_repeat);
        end
    end
    save(['../result/exp3/', model.data_name, '_results.mat'], ...
        'errors_labeled');
end

function error_curve_save(file_path, linear_errs, lrc_errs, ssl_errs, lrc_ssl_errs)
    linear_errs = linear_errs(:)*100;
    lrc_errs = lrc_errs(:)*100;
    ssl_errs = ssl_errs(:)*100;
    lrc_ssl_errs = lrc_ssl_errs(:)*100;
    fig=figure('Position', [100, 100, 850, 600]);
    x_length = min([size(linear_errs, 1), size(ssl_errs, 1), size(lrc_errs, 1), size(lrc_ssl_errs, 1)]);
    plot(1:x_length, linear_errs, '-','LineWidth',3.5);
    hold on;
    plot(1:x_length, lrc_errs, '-','LineWidth',3.5);
    plot(1:x_length, ssl_errs, '-','LineWidth',3.5, 'Color', 'black');
    plot(1:x_length, lrc_ssl_errs, '-','LineWidth',3.5);

    grid on
    legend({ 'Linear-MC','LRC-MC', 'SS-MC', 'LSVV'}, 'FontSize',40);
    ylabel('Error Rate(%)');
    xlabel('The number of iterations');
    set(gca,'FontSize',45,'Fontname', 'Times New Roman');
    hold off;
    
    print(fig,file_path,'-depsc')
end

%experiment_2
%error_curve_save(file_path,errors_matrix(4,:,:),errors_matrix(3,:,:), errors_matrix(2,:,:), errors_matrix(1,:,:));