initialization;
model.n_repeats = 1;
%model.T = 10;

datasets = {
    'iris'
% 'scene', ...
% 'yeast', ...  
%'corel5k', ...
% 'bibtex', ...
% 'rf2', ...
% 'scm1d', ...
};

for dataset = datasets    
    rng('default');
    model.data_name = char(dataset);
    parameter_observe(model.data_name)
    exp2_run(model);
end

function exp2_run(model)
    % load datasets
    [X, y] = load_data(model.data_name);
    sigma = select_gaussian_kernel(model.data_name);
    X_rf_100 = random_fourier_features(X, 100, sigma);
    L = construct_laplacian_graph(model.data_name, X, 10);
    
    if strcmp(check_task_type(model.data_name), 'mc')
        K = min(size(X, 1), max(y));
    else
        K = min(size(X, 1), size(y, 1));
    end
    
    can_theta = 0 : 0.1 : 1;
    errors_theta = zeros(2, numel(can_theta));
    for i = 1 : numel(can_theta)
        i_theta_partition = can_theta(i);
        model.tail_start = ceil(K * i_theta_partition);
        [test_all_errs, run_times] = repeat_train(model, X, X_rf_100, y, L);        
        test_errs = zeros(2, model.n_repeats);
        for i_repeat = 1:model.n_repeats
            for i_method = 1:2
                i_repeat_errors = test_all_errs{i_method, i_repeat};
                test_errs(i_method, i_repeat) = min(i_repeat_errors(1, end - 5: end));
            end
        end
        errors_theta(:, i) = test_errs;
    end
    save(['../result/exp2/', model.data_name, '_results.mat'], ...
        'errors_theta');
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