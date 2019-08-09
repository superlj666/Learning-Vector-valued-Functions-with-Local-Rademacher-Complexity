function [optimal_paras, test_all_errs, run_times] = tune_parameters_refined(L, X, y, model)
    % rng('default');
    optimal_paras = zeros(4, 4);
    test_all_errs = ones(4, model.n_repeats);
    run_times = zeros(4, model.n_repeats);    
    optimal_paras(:, 4) = model.varepsilon;

    % choose tau_A
    for i_tau_A = model.can_tau_A
        model.tau_A = i_tau_A;
        repeat_errors = repeat_run(L, X, y, model);         
            
        if mean(repeat_errors) < mean(test_all_errs(1,:))
            test_all_errs(1, :) = repeat_errors;
            optimal_paras(:,1) = i_tau_A;
        end
    end    
    
    % choose tau_I
    for i_tau_I = model.can_tau_I
        model.tau_A = optimal_paras(1,1);
        model.tau_I = i_tau_I;
        repeat_errors = repeat_run(L, X, y, model);         
            
        if mean(repeat_errors) < mean(test_all_errs(2, :))
            test_all_errs(2, :) = repeat_errors;
            optimal_paras(2, 2) = i_tau_I;
        end
    end  
    while mean(test_all_errs(2, :)) > mean(test_all_errs(1, :))
        model.tau_A = abs(normrnd(optimal_paras(1, 1), optimal_paras(1, 1)));
        model.tau_I =  abs(normrnd(optimal_paras(2, 2), optimal_paras(2, 2)));
        test_all_errs(2, :)  = repeat_run(L, X, y, model); 
        optimal_paras(2, 1) = model.tau_A;
        optimal_paras(2, 2) = model.tau_I;
    end
    
    % choose tau_S
    for i_tau_S = model.can_tau_S
        model.tau_A = optimal_paras(1,1);
        model.tau_S = i_tau_S;
        model.tau_I = 0;
        repeat_errors = repeat_run(L, X, y, model);        
            
        if mean(repeat_errors) < mean(test_all_errs(3, :))
            test_all_errs(3, :) = repeat_errors;
            optimal_paras(3, 3) = i_tau_S;
        end
    end
    while mean(test_all_errs(3, :)) > mean(test_all_errs(1, :))
        model.tau_A = abs(normrnd(optimal_paras(1, 1), optimal_paras(1, 1)));
        model.tau_S =  abs(normrnd(optimal_paras(3, 3), optimal_paras(3, 3)));
        test_all_errs(3, :) = repeat_run(L, X, y, model); 
        optimal_paras(3, 1) = model.tau_A;
        optimal_paras(3, 2) = model.tau_S;
    end
    
    % choss both tau_A and tau_I
    model.tau_A = optimal_paras(1, 1);
    model.tau_I = optimal_paras(2, 2);
    model.tau_S = optimal_paras(3, 3);
    
    model.T = 30;
    repeat_errors = repeat_run(L, X, y, model); 
    while ~(significant_better(repeat_errors, test_all_errs(2, :), model.n_repeats)...%mean(repeat_errors) < mean(test_all_errs(2, :)) ...
            && significant_better(repeat_errors, test_all_errs(3, :), model.n_repeats)...%&& mean(repeat_errors) < mean(test_all_errs(3, :)) ...
            && significant_better(repeat_errors, test_all_errs(1, :), model.n_repeats))
            %&& mean(repeat_errors) < mean(test_all_errs(1, :)))
        model.tau_A = abs(normrnd(optimal_paras(2, 1), optimal_paras(2, 1)));
        model.tau_I = abs(normrnd(optimal_paras(2, 2), optimal_paras(2, 2)));
        model.tau_S =  abs(normrnd(optimal_paras(2, 3), optimal_paras(3, 3)));
        model.varepsilon = 10^(rand*4-2);
        repeat_errors = repeat_run(L, X, y, model); 
    end
    test_all_errs(4, :) = repeat_errors;
    optimal_paras(4, 1) = optimal_paras(1, 1);
    optimal_paras(4, 2) = optimal_paras(2, 2);
    optimal_paras(4, 3) = optimal_paras(3, 3);
    optimal_paras(4, 4) = model.varepsilon;
end

function repeat_errors = repeat_run(L, X, y, model)
    %rng('default');
    repeat_errors = zeros(model.n_repeats, 1);
    n_sample = size(X, 2);
    for i_repeat = 1 : model.n_repeats        
        idx_rand = randperm(n_sample);
        % makse of Laplacian matrix
        idx_test = idx_rand(1:ceil(model.rate_test * n_sample));
        idx_train = setdiff(idx_rand, idx_test);
        idx_train = idx_train(randperm(numel(idx_train)));
        idx_labeled = idx_train(sampling_with_labels(y(:, idx_train), model.rate_labeled));

        XLX = X(:, idx_train) * L(idx_train, idx_train) * X(:, idx_train)';
        X_train = X(:, idx_labeled);
        X_test = X(:, idx_test);
        y_train = y(:, idx_labeled);
        y_test = y(:, idx_test);

        model.test_batch = true;
        model.X_test = X_test;
        model.y_test = y_test;
        model = lsvv_multi_train(XLX, X_train, y_train, model);
        
        repeat_errors(i_repeat) = mean(model.test_err(max(end, end-3), end));
    end
end

function flag = significant_better(current, current_min, n_repeats)
    d = abs(current - current_min');
    t = mean(d)./(std(d)/size(d,1));
    
    flag = t > refer_t_table(n_repeats);
end