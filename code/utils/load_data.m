function [X, y] = load_data(data_name)
    dataset_path = '../../../../../datasets/';
    if strcmp(check_task_type(data_name), 'ml')
        load([dataset_path, data_name]);
    elseif strcmp(check_task_type(data_name), 'mc')
        [y, X] = libsvmread([dataset_path, data_name]);
    end
    %[y, X] = libsvmread(['/home/lijian/datasets/', data_name]);

    max_columns = max(X);
    min_columns = min(X);
    step = max_columns - min_columns;
    if find(step == 0)
        keep_dimensions = setdiff(1:size(X, 2), find(step == 0));
        X = X(:, keep_dimensions);
        min_columns = min_columns(keep_dimensions);
        step = step(keep_dimensions);
    end
    X = (X - min_columns)./step;

    % regularize labels to 1..C 
    if(size(y, 2)==1) 
        y_labels = unique(y);
        y_tmp = y;
        for i_label = 1 : numel(y_labels)
            y(y_tmp == y_labels(i_label)) = i_label;
        end

%         y_one_hot = zeros(numel(y_labels), length(y));
%         for i = 1 : length(y)
%             y_one_hot(y(i), i) = 1;
%         end
%         y = y_one_hot;
    end
    y = y';
    X = X';
end