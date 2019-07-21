initialization;
for dataset = datasets
    [y, X] = load_data(char(dataset));
    if ismissing(X)
        fprintf('%s exists missing data', char(dataset));
    end
end