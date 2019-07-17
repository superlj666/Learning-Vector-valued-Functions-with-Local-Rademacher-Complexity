function [X, y] = load_data(data_name)
    dataset_path = ['../datasets/', data_name];
    [y, X] = libsvmread(dataset_path);

    % regularize labels to 1..C    
    y_labels = unique(y);
    y_tmp = y;
    for i_label = 1 : numel(y_labels)
        y(y_tmp == y_labels(i_label)) = i_label;
    end
end