initialization;
for dataset = datasets
    [X, y] = load_data(char(dataset));
    if sum(sum(ismissing(X)))>0
        fprintf('%s exists missing data\n', char(dataset));
    end
end