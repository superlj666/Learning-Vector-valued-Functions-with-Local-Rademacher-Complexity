initialization;
for dataset = datasets
    exp2_dataset(char(dataset));
end

function exp2_dataset(data_name)    
    %% Choose parameters for our method
    load(['../result/', data_name, '_results.mat'], 'errors_matrix');
    file_path = ['../result/exp2/', data_name];
    error_curve_save(file_path, mean(errors_matrix(4,:,:), 1), mean(errors_matrix(3,:,:), 1), mean(errors_matrix(2,:,:), 1), mean(errors_matrix(1,:,:), 1));
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
    legend({ 'Linear-MC','LRC-MC', 'SS-MC', 'PS3VT'}, 'FontSize',40);
    ylabel('Error Rate(%)');
    xlabel('The number of iterations');
    set(gca,'FontSize',45,'Fontname', 'Times New Roman');
    hold off;
    
    print(fig,file_path,'-depsc')
end

%experiment_2
%error_curve_save(file_path,errors_matrix(4,:,:),errors_matrix(3,:,:), errors_matrix(2,:,:), errors_matrix(1,:,:));