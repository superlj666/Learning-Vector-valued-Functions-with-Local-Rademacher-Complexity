initialization;
for dataset = datasets
    data_name = char(dataset);
    [X, y] = load_data(data_name);
    
    N = size(y, 2);
    if strcmp(check_task_type(data_name), 'mc')
        K = max(y);
    else
        K = size(y, 1);
    end
    load_data(data_name);
    fprintf('%s\t', data_name);
    fprintf('&%d\t', ceil(N*0.1*0.7));
    fprintf('&%d\t', ceil(N*0.9*0.7));
    fprintf('&%d\t', N - ceil(N*0.9*0.7) - ceil(N*0.1*0.7));
    fprintf('&%d\t', size(X, 1));
    fprintf('&%d \\\\ \n', K);
end