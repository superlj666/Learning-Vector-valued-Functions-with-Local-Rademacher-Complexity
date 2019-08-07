initialization;
model.n_repeats = 1;
model.T = 20;

for dataset = datasets    
    rng('default');
    model.data_name = char(dataset);
    parameter_observe(model.data_name)
    exp3_run(model);
    draw_error_curve(char(dataset));
end

function exp3_run(model)
    % load datasets
    [X, y] = load_data(model.data_name);
    sigma = select_gaussian_kernel(model.data_name);
    X_rf_100 = random_fourier_features(X, 100, sigma);
    L = construct_laplacian_graph(model.data_name, X, 10);
    
    can_labeled = 0.1 : 0.1 : 1;
    errors_labeled = zeros(4, numel(can_labeled));
    test_all_errs = cell(4, model.n_repeats, numel(can_labeled));
    run_times = zeros(4, model.n_repeats);
    
    load(['../result/', model.data_name, '_models.mat'], 'model_lrc', 'model_lrc_ssl', 'model_lrc_100', 'model_lrc_ssl_100');
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
            model_lrc_ssl_rf_100 = model_combination(i_model, model_lrc_ssl_100);
            model_lrc_ssl_rf_100 = lsvv_multi_train(XLX_100, X_train_100, y_train, model_lrc_ssl_rf_100);
            run_time = toc(t);
            run_times(3, i_repeat) = run_time;            
            test_all_errs{3, i_repeat, i_labeled} = model_lrc_ssl_rf_100.test_err;

            t = tic;
            model_lrc_100 = model_combination(i_model, model_lrc_100);
            model_lrc_100 = lsvv_multi_train(XLX_100, X_train_100, y_train, model_lrc_100);
            run_time = toc(t);
            run_times(4, i_repeat) = run_time;            
            test_all_errs{4, i_repeat, i_labeled} = model_lrc_100.test_err;
       end
    end
     
    for i_method = 1:4
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

function draw_error_curve(data_name)
    load(['../result/exp3/', data_name, '_results.mat']);
    errors_labeled = errors_labeled.*100;
    figure(1);
    set(gcf,'Position',[100 100 1000 360]);
    
    subplot(121);
    ax = gca;
    plot(0:9, errors_labeled(1,:), '-', 'linewidth', 2, 'color', 'r');   
    hold on;  
    plot(0:9, errors_labeled(2,:), '--', 'linewidth', 2, 'color', 'b');   
    grid on;
    max_level = max(max(errors_labeled(1:2,:)));
    min_level = min(min(errors_labeled(1:2,:)));
    step = max_level - min_level;
    ylim([min_level - 0.5 * step, max_level + 0.5 * step]);
    xlim([0;9]);
    ax.XTick = 0:9;
    ax.XTickLabel = 0.1:0.1:1;
    ytickformat('%.2f')
    legend({ 'LSVV','SS-VV', 'GRC-SS-VV'});
    ylabel('Error Rate(%)');
    xlabel('Label Rate');
    title([data_name, ' in input space']);
    hold off;
    
    subplot(122);
    ax = gca;
    plot(0:9, errors_labeled(3,:), '-', 'linewidth', 2, 'color', 'r');   
    hold on;  
    plot(0:9, errors_labeled(4,:), '--', 'linewidth', 2, 'color', 'b');   
    grid on;
    max_level = max(max(errors_labeled(3:4,:)));
    min_level = min(min(errors_labeled(3:4,:)));
    step = max_level - min_level;
    ylim([min_level - 0.5 * step, max_level + 0.5 * step]);
    xlim([0;9]);
    ax.XTick = 0:9;
    ax.XTickLabel = 0.1:0.1:1; 
    ytickformat('%.2f')
    legend({ 'LSVV','SS-VV', 'GRC-SS-VV'});
    ylabel('Error Rate(%)');
    xlabel('Label Rate');
    title([data_name, ' with 100 random features']);
    hold off;
    
    saveas(gcf, ['../result/exp3/exp3_', data_name], 'epsc');
end