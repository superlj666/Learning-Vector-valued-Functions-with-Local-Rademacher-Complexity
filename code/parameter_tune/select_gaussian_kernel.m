function sigma = select_gaussian_kernel(data_name)
    load('../result/sigma_gaussian_kernel');
    sigma = 0.5;
    if(M.isKey(data_name) == false)
        fprintf(['%s has not been tuned for optimal kernel parameter and use the default 0.5.\n'...
            'Please run tune_gaussian_kerne.m firstly.'], data_name);
    else
        sigma = M(data_name);
    end
end