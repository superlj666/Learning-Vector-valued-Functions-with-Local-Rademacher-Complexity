function sigma = select_gaussian_kernel(data_name)
    sigma = 1;
    if strcmp(data_name, 'dna')
        sigma = 6;
    elseif strcmp(data_name, 'wine')
        sigma = 1.5;
    elseif strcmp(data_name, 'usps')
        sigma = 7;
    elseif strcmp(data_name, 'protein')
        sigma = 3.5;
    elseif strcmp(data_name, 'poker')
        sigma = 8;
    elseif strcmp(data_name, 'connect-4')
        sigma = 8;
    elseif strcmp(data_name, 'covtype')
        sigma = 2;
    end
end