function [X, y] = load_data(data_name)
    if strcmp(computer('arch'), 'win64')
        dataset_path = '../../../../../Datasets/';
    else
        dataset_path = '/home/lijian/datasets/';
    end
    if strcmp(check_task_type(data_name), 'mlc') || strcmp(check_task_type(data_name), 'mlr')
        load([dataset_path, data_name]);
    elseif strcmp(check_task_type(data_name), 'mc')
        [y, X] = libsvmread([dataset_path, data_name]);
    end
    
    % regularize features to [0,1]
    max_columns = max(X);
    min_columns = min(X);
    step = max_columns - min_columns;
    % remove features which always keeps constant
    if find(step == 0)
        keep_dimensions = setdiff(1:size(X, 2), find(step == 0));
        X = X(:, keep_dimensions);
        min_columns = min_columns(keep_dimensions);
        step = step(keep_dimensions);
    end
    X = (X - min_columns)./step;
    
    % take into consider the bias : wx+b
    X = [X, ones(size(X, 1), 1)];

    % regularize multi-class labels to 1..K
    if(strcmp(check_task_type(data_name), 'mc')) 
        y_labels = unique(y);
        y_tmp = y;
        for i_label = 1 : numel(y_labels)
            y(y_tmp == y_labels(i_label)) = i_label;
        end
    end
    % regularize multi-label regression labels to [0,1]
    if(strcmp(check_task_type(data_name), 'mlr')) 
        y = (y - min(y))./(max(y) - min(y));
    end
    y = y';
    X = X';
end