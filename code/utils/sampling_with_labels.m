function idx_labeled = sampling_with_labels(y, rate_labeled)
    idx_labeled = [];
    for i_label = 1 : max(y)
        idx_label = find(y==i_label);
        idx_label = idx_label(randperm(numel(idx_label)));
        idx_labeled = [idx_labeled; idx_label(1 : ceil(numel(idx_label) * rate_labeled), 1);];
    end
end