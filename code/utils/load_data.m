function [X, y] = load_data(data_name)
    [y, X] = libsvmread(['/home/lijian/datasets/', data_name]);

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
        y = y';
    end
    X = X';
end