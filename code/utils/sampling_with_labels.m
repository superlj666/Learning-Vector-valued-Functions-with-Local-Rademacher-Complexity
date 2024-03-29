function idx_labeled = sampling_with_labels(y, rate_labeled)
    idx_labeled = [];
    if size(y, 1) == 1
        for i_label = 1 : max(y)
            idx_label = find(y==i_label);
            idx_label = idx_label(randperm(numel(idx_label)));
            idx_labeled = [idx_labeled, idx_label(1 : ceil(numel(idx_label) * rate_labeled)),];
        end
    else
        idx_label = randperm(size(y, 2));
        idx_labeled = idx_label(1:ceil(numel(idx_label) * rate_labeled));
    end
end