function model = record_batch(XLX, X, y, model, type)
    if strcmpi(type, 'test')
        if ~isfield(model, 'test_err'), model.test_err = [];  end
        if ~isfield(model, 'test_loss'), model.test_loss = [];  end
        if ~isfield(model, 'test_complexity'), model.test_complexity = [];  end
        if ~isfield(model, 'test_unlabeled'), model.test_unlabeled = [];  end
        if ~isfield(model, 'test_trace'), model.test_trace = [];  end
        if ~isfield(model, 'test_objective'), model.test_objective = [];  end
    elseif strcmpi(type, 'train')
        if ~isfield(model, 'train_err'), model.train_err = [];  end
        if ~isfield(model, 'train_loss'), model.train_loss = [];  end
        if ~isfield(model, 'train_complexity'), model.train_complexity = [];  end
        if ~isfield(model, 'train_unlabeled'), model.train_unlabeled = [];  end
        if ~isfield(model, 'train_trace'), model.train_trace = [];  end
        if ~isfield(model, 'train_objective'), model.train_objective = [];  end
    end

    % predict and calculate all terms
    out = model.weights' * X';
    loss = 0;
    err = 0;
    for i_sample = 1 : numel(y)
        h_x = out( : ,i_sample);
        margin_true = h_x(y(i_sample));
        h_x(y(i_sample)) = - Inf;
        margin_pre = max(h_x);
        loss = loss + max(1 - margin_true + margin_pre, 0);
        err = err + (margin_true <=  margin_pre);
    end

    if strcmpi(type, 'test')
        model.test_err(end + 1) = err/numel(y);
        model.test_loss(end + 1) = loss/numel(y);
        model.test_complexity(end + 1) = model.tau_A * norm(model.weights, 'fro')^2;
        model.test_unlabeled(end + 1) = model.tau_I * trace(model.weights' * XLX * model.weights);
        model.test_trace(end + 1) =  model.tau_S * sum(sqrt(eig(model.S' * model.S)));
        model.test_objective(end + 1) = model.test_loss(end) + model.test_complexity(end) + model.test_unlabeled(end) + model.test_trace(end);
    elseif strcmpi(type, 'train')
        model.train_err(end + 1) = err/numel(y);
        model.train_loss(end + 1) = loss/numel(y);
        model.train_complexity(end + 1) = model.tau_A * norm(model.weights, 'fro')^2;
        model.train_unlabeled(end + 1) = model.tau_I * trace(model.weights' * XLX * model.weights);
        model.train_trace(end + 1) =  model.tau_S * sum(sqrt(eig(model.S' * model.S)));
        model.train_objective(end + 1) = model.train_loss(end) + model.train_complexity(end) + model.train_unlabeled(end) + model.train_trace(end);
    end
end