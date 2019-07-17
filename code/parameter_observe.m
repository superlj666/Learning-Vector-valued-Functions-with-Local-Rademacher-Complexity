function parameter_observe(data_name)
    load(['../data/', data_name, '/', 'cross_validation.mat']);

    model_lrc_ssl = learner_lrc_ssl(errors_validate, can_step, can_tau_S, can_tau_A, can_tau_I);
    model_ssl = learner_ssl(errors_validate, can_step, can_tau_S, can_tau_A, can_tau_I);
    model_ssl.tau_A = 10*model_lrc_ssl.tau_A;
    model_ssl.tau_S = model_lrc_ssl.tau_S;
    model_lrc = learner_lrc(errors_validate, can_step, can_tau_S, can_tau_A, can_tau_I);
    model_lrc.tau_A = 4*model_lrc_ssl.tau_A;
    model_lrc.tau_I = 10*model_lrc_ssl.tau_I;
    model_linear = learner_linear(errors_validate, can_step, can_tau_S, can_tau_A, can_tau_I);
    model_linear.tau_A = 10*model_lrc_ssl.tau_A;
    model_linear.step = 5*10^4;

    save(['../result/', data_name, '_models.mat'], 'model_lrc_ssl', 'model_ssl', 'model_lrc', 'model_linear');
end

function model = learner_lrc_ssl(errors_validate, can_step, can_tau_S, can_tau_A, can_tau_I)
    cv_results = reshape([errors_validate{:, 1}], [numel(can_step), numel(can_tau_S), numel(can_tau_A), numel(can_tau_I)]);
    cv_results(:, numel(can_tau_S), :, :) = 1;
    cv_results(:, :, :, numel(can_tau_I)) = 1;
    [~, loc_best] = min(cv_results(:));
    [d1, d2, d3, d4] = ind2sub([numel(can_step), numel(can_tau_S), numel(can_tau_A), numel(can_tau_I)], loc_best);
    fprintf('-----LRC_SSL: %.4f\t tau_I: %s\t tau_A: %s\t tau_S: %s\t step: %.0f-----\n', ...
    errors_validate{loc_best, 1}, num2str(can_tau_I(d4)), num2str(can_tau_A(d3)), num2str(can_tau_S(d2)), can_step(d1));

    model.tau_I = can_tau_I(d4);
    model.tau_A = can_tau_A(d3);
    model.tau_S = can_tau_S(d2);
    model.step = can_step(d1);
end

function model = learner_ssl(errors_validate, can_step, can_tau_S, can_tau_A, can_tau_I)
    cv_results = reshape([errors_validate{:, 1}], [numel(can_step), numel(can_tau_S), numel(can_tau_A), numel(can_tau_I)]);
    cv_results(:, 1 : numel(can_tau_S) - 1, :, :) = 1;
    cv_results(:, :, :, numel(can_tau_I)) = 1;
    [~, loc_best] = min(cv_results(:));
    [d1, d2, d3, d4] = ind2sub([numel(can_step), numel(can_tau_S), numel(can_tau_A), numel(can_tau_I)], loc_best);
    fprintf('-----SSL: %.4f\t tau_I: %s\t tau_A: %s\t tau_S: %s\t step: %.0f-----\n', ...
    errors_validate{loc_best, 1}, num2str(can_tau_I(d4)), num2str(can_tau_A(d3)), num2str(can_tau_S(d2)), can_step(d1));

    model.tau_I = can_tau_I(d4);
    model.tau_A = can_tau_A(d3);
    model.tau_S = can_tau_S(d2);
    model.step = can_step(d1);
end

function model = learner_lrc(errors_validate, can_step, can_tau_S, can_tau_A, can_tau_I)
    cv_results = reshape([errors_validate{:, 1}], [numel(can_step), numel(can_tau_S), numel(can_tau_A), numel(can_tau_I)]);
    cv_results(:, numel(can_tau_S), :, :) = 1;
    cv_results(:, :, :, 1 : numel(can_tau_I) - 1) = 1;
    [~, loc_best] = min(cv_results(:));
    [d1, d2, d3, d4] = ind2sub([numel(can_step), numel(can_tau_S), numel(can_tau_A), numel(can_tau_I)], loc_best);
    fprintf('-----LRC: %.4f\t tau_I: %s\t tau_A: %s\t tau_S: %s\t step: %.0f-----\n', ...
    errors_validate{loc_best, 1}, num2str(can_tau_I(d4)), num2str(can_tau_A(d3)), num2str(can_tau_S(d2)), can_step(d1));

    model.tau_I = can_tau_I(d4);
    model.tau_A = can_tau_A(d3);
    model.tau_S = can_tau_S(d2);
    model.step = can_step(d1);
end

function model = learner_linear(errors_validate, can_step, can_tau_S, can_tau_A, can_tau_I)
    cv_results = reshape([errors_validate{:, 1}], [numel(can_step), numel(can_tau_S), numel(can_tau_A), numel(can_tau_I)]);
    cv_results(:, 1 : numel(can_tau_S) - 1, :, :) = 1;
    cv_results(:, :, :, 1 : numel(can_tau_I) - 1) = 1;
    [~, loc_best] = min(cv_results(:));
    loc_best = loc_best(1);
    [d1, d2, d3, d4] = ind2sub([numel(can_step), numel(can_tau_S), numel(can_tau_A), numel(can_tau_I)], loc_best);
    fprintf('-----Linear: %.4f\t tau_I: %s\t tau_A: %s\t tau_S: %s\t step: %.0f-----\n', ...
    errors_validate{loc_best, 1}, num2str(can_tau_I(d4)), num2str(can_tau_A(d3)), num2str(can_tau_S(d2)), can_step(d1));

    model.tau_I = can_tau_I(d4);
    model.tau_A = can_tau_A(d3);
    model.tau_S = can_tau_S(d2);
    model.step = can_step(d1);
end