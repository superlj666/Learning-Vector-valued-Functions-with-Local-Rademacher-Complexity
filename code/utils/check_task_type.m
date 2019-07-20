function task_type = check_task_type(data_name)
    task_type = 'mc';
    if strcmp(data_name, 'bibtex') || strcmp(data_name, 'corel5k') || strcmp(data_name, 'yeast') ...
            || strcmp(data_name, 'scm1d') || strcmp(data_name, 'rf2') || strcmp(data_name, 'scene')
        task_type = 'ml';
    end
end