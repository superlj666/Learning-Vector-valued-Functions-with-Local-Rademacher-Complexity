initialization;
for dataset = datasets
    rng('default');
    model.data_name = char(dataset);
    %exp3_dataset(char(dataset), model);
    draw_sample_curve1(char(dataset));
end

function exp3_dataset(data_name, model)
    %% Choose parameters for our method
    load(['../result/', data_name, '_models.mat'], 'model_linear', 'model_lrc', 'model_ssl', 'model_lrc_ssl');
    model_linear = model_combination(model, model_linear);
    model_lrc = model_combination(model, model_lrc);
    model_ssl = model_combination(model, model_ssl);
    model_lrc_ssl = model_combination(model, model_lrc_ssl);

    % load datasets
    [X, y] = load_data(data_name);    
    L = construct_laplacian_graph(data_name, X, 10);

    errs_partition = zeros(4, 9, model_lrc_ssl.n_repeats);
    for i_partition = 1 : 9        
        model_linear.rate_labeled = i_partition * 0.1 + 0.1;
        model_lrc.rate_labeled = i_partition * 0.1 + 0.1;
        model_ssl.rate_labeled = i_partition * 0.1 + 0.1;
        model_lrc_ssl.rate_labeled = i_partition * 0.1 + 0.1;

        linear_errs = repeat_test(model_linear, 'linear', X, y, L);
        lrc_errs = repeat_test(model_lrc, 'lrc', X, y, L);
        ssl_errs = repeat_test(model_ssl, 'ssl', X, y, L);
        lrc_ssl_errs = repeat_test(model_lrc_ssl, 'lrc_ssl', X, y, L);
        
        errs_partition(1, i_partition, :) = mean(linear_errs, 2);
        errs_partition(2, i_partition, :) = mean(lrc_errs, 2);
        errs_partition(3, i_partition, :) = mean(ssl_errs, 2);
        errs_partition(4, i_partition, :) = mean(lrc_ssl_errs, 2);
    end
    
    save(['../result/exp3/', data_name, '_errs_partition.mat'], ...
    'errs_partition');
end

function draw_sample_curve1(data_name)
    load(['../result/exp3/', data_name, '_errs_partition.mat'], 'errs_partition');
    errs_partition = errs_partition*100;
    lrc_ssl_errs_mean = mean(errs_partition(1, :, :), 3);
    ssl_errs_mean = mean(errs_partition(2, :, :), 3);
    lrc_errs_mean = mean(errs_partition(3, :, :), 3);
    linear_errs_mean = mean(errs_partition(4, :, :), 3);
    

    fig=figure('Position', [100, 100, 850, 600]);
    x_list=1:9;
    plot(x_list, linear_errs_mean, '-o','MarkerSize', 10,'LineWidth',3.5);
    hold on;
    plot(x_list, lrc_errs_mean,'-.^','MarkerSize', 10,'LineWidth',3.5);
    plot(x_list, ssl_errs_mean,'--x','MarkerSize', 10,'LineWidth',3.5, 'Color', 'black');
    plot(x_list, lrc_ssl_errs_mean,'-*','MarkerSize', 10,'LineWidth',3.5);
%     max_level=max(lrc_ssl_errs_mean);
%     min_level=min(linear_errs_mean);
%     step=max_level-min_level;
    
    xticklabels({'20', '30', '40', '50' ,'60', '70', '80', '90', '100'});
    grid on
%     set(gca,'XLim',[0.5 7.5])
%     set(gca,'YLim',[min_level-0.5*step max_level+1.2*step])
    set(gca,'YLim',[5, 13]);
    set(gca,'yTick',[5:1:13]);
    set(gca,'XLim',[1 9]);
    set(gca,'xTick',[1:1:9])
    legend({ 'Linear-MC','LRC-MC', 'SS-MC', 'PS3VT'}, 'FontSize',40);
    ylabel('Error Rate(%)');
    xlabel('Labeled Samples Rate (%)');
    set(gca,'FontSize',40,'Fontname', 'Times New Roman');
    hold off;

    print(fig,['../result/exp3/', data_name],'-depsc')
end

function draw_sample_curve(data_name)
    load(['../result/exp3/', data_name, '_errs_partition.mat'], 'errs_partition');
    linear_errs_mean = mean(errs_partition(1, :, :), 3);
    lrc_errs_mean = mean(errs_partition(2, :, :), 3);
    ssl_errs_mean = mean(errs_partition(3, :, :), 3);
    lrc_ssl_errs_mean = mean(errs_partition(4, :, :), 3);
    
    linear_errs_std = zeros(1, 9);
    lrc_errs_std = zeros(1, 9);
    ssl_errs_std = zeros(1, 9);
    lrc_ssl_errs_std = zeros(1, 9);
    for i = 1:9
        linear_errs_std(1, i) = std(errs_partition(1, i, :));
        lrc_errs_std(1, i) = std(errs_partition(2, i, :));
        ssl_errs_std(1, i) = std(errs_partition(3, i, :));
        lrc_ssl_errs_std(1, i) = std(errs_partition(4, i, :));
    end

    fig=figure;
    x_list=1:9;
    errorbar(x_list, lrc_ssl_errs_mean, lrc_ssl_errs_std,'-*','LineWidth',1);
    hold on;
    errorbar(x_list, lrc_errs_mean, lrc_errs_std,'-.^','LineWidth',1);
    errorbar(x_list, ssl_errs_mean, ssl_errs_std,'-x','LineWidth',1);
    errorbar(x_list, linear_errs_mean, linear_errs_std, '--o','LineWidth',1);

%     max_level=max(lrc_ssl_errs_mean);
%     min_level=min(linear_errs_mean);
%     step=max_level-min_level;
    
    xticklabels({'20%', '30%', '40%', '50%' ,'60%', '70%', '80%', '90%', '100%'});
    grid on
%     set(gca,'XLim',[0.5 7.5])
%     set(gca,'YLim',[min_level-0.5*step max_level+1.2*step])
    legend({'PS3VT', 'SS-MC','LRC-MC', 'Linear-MC'}, 'FontSize',12);
    ylabel('Error Rate(%)');
    xlabel('%# Labeled Samples');
    set(gca,'FontSize',20,'Fontname', 'Times New Roman');
    hold off;

    print(fig,['../result/exp3/', data_name],'-depsc')
end
% 
% experiment_3
% linear_errs_mean = errs_partition(4, :);
% lrc_errs_mean = errs_partition(3, :);
% ssl_errs_mean = errs_partition(2, :);
% lrc_ssl_errs_mean = errs_partition(1, :);