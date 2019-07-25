initialization;

datasets = {
'iris', ...
% 'wine', ...
% 'glass', ...
% 'svmguide2', ...
% 'vowel', ...
% 'vehicle', ...
% 'dna', ...
% 'segment', ...
% 'satimage', ...
% 'usps', ...
% 'pendigits', ...
% 'letter', ...
% 'protein', ...
% 'poker', ...
% 'shuttle', ...
% 'Sensorless', ...
% 'mnist', ...
% 'connect-4', ...
% 'acoustic',...
% 'covtype', ...
% 'corel5k', ...
% 'scene', ...
% 'yeast', ...  
% 'bibtex', ...
% 'rf2', ...
% 'scm1d', ...
};

for dataset = datasets    
    rng('default');
    model.data_name = char(dataset);
    exp2_run(char(dataset), model);
end

function exp2_run(data_name, model)
    % load datasets
    idx_rand = randperm(n_sample);
    idx_test = idx_rand(1:ceil(model.rate_test * n_sample));
    idx_train = setdiff(idx_rand, idx_test);
    idx_train = idx_train(randperm(numel(idx_train)));

    XLX = X(:, idx_train) * L(idx_train, idx_train) * X(:, idx_train)';
    idx_labeled = idx_train(sampling_with_labels(y(:, idx_train), model.rate_labeled));  

    X_train = X(:, idx_labeled); 
    X_test = X(:, idx_test);
    y_train = y(:, idx_labeled);
    y_test = y(:, idx_test);
    
    model.n_record_batch = ceil(linspace(1, ceil(numel(idx_labeled) / model.n_batch) * model.T, 30));    
    
    %% Choose parameters for our method
    load(['../result/', data_name, '_models.mat'], 'model_linear', 'model_lrc', 'model_ssl', 'model_lrc_ssl');
    
    model_linear = model_combination(model, model_linear);
    model_lrc = model_combination(model, model_lrc);
    model_ssl = model_combination(model, model_ssl);
    model_lrc_ssl = model_combination(model, model_lrc_ssl);
    model_linear_100 = model_linear;
    model_lrc_100 = model_lrc;
    model_ssl_100 = model_ssl;
    model_lrc_ssl_100 = model_lrc_ssl;

    errs_theta = zeros(8, 11, model_lrc_ssl.n_repeats);
    min_dimension = min(size(X, 2), size(y, 2));
    
    for theta_partition = 0 :0.1:1 
        theta = ceil(min_dimension * theta_partition);
        model_linear.tail_start = theta;
        model_lrc.tail_start = theta;
        model_ssl.tail_start = theta;
        model_lrc_ssl.tail_start = theta;
        model_linear_100.tail_start = theta;
        model_lrc_100.tail_start = theta;
        model_ssl_100.tail_start = theta;
        model_lrc_ssl_100.tail_start = theta;
    
        linear_errs = repeat_test(model_linear, 'linear', X, y, L);
        lrc_errs = repeat_test(model_lrc, 'lrc', X, y, L);
        ssl_errs = repeat_test(model_ssl, 'ssl', X, y, L);
        lrc_ssl_errs = repeat_test(model_lrc_ssl, 'lrc_ssl', X, y, L);
        linear_100_errs = repeat_test(model_linear, 'linear_100', X, y, L);
        lrc_100_errs = repeat_test(model_lrc, 'lrc_100', X, y, L);
        ssl_100_errs = repeat_test(model_ssl, 'ssl_100', X, y, L);
        lrc_ssl_100_errs = repeat_test(model_lrc_ssl, 'lrc_ssl_100', X, y, L);
        
        errs_theta(1, i_partition, :) = mean(linear_errs, 2);
        errs_theta(2, i_partition, :) = mean(lrc_errs, 2);
        errs_theta(3, i_partition, :) = mean(ssl_errs, 2);
        errs_theta(4, i_partition, :) = mean(lrc_ssl_errs, 2);
        errs_theta(5, i_partition, :) = mean(linear_100_errs, 2);
        errs_theta(6, i_partition, :) = mean(lrc_100_errs, 2);
        errs_theta(7, i_partition, :) = mean(ssl_100_errs, 2);
        errs_theta(8, i_partition, :) = mean(lrc_ssl_100_errs, 2);
    end
    
    save(['../result/exp2/', data_name, '_errs_partition.mat'], ...
    'errs_theta');
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