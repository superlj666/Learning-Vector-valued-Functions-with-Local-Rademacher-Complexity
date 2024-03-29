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
out = model.weights' * X;
loss = 0;
err = 0;
task_type = check_task_type(model.data_name);
n_test = size(y, 2);

if strcmp(task_type, 'mc')
    for i_sample = 1 : n_test
        h_x = out( : ,i_sample);
        label_true = y(i_sample);
        margin_true = h_x(label_true);
        h_x(label_true) = -Inf;
        [margin_subopt, label_subopt] = max(h_x);
        loss = loss + max(1 - margin_true + margin_subopt, 0);
        err = err + (margin_true <=  margin_subopt);
    end
    loss = loss/n_test;
    err = err/n_test;
elseif strcmp(task_type, 'mlc')
    loss = norm(out - y, 'fro')^2/n_test;
    err = sum(sum(xor(out > 0.5, y)))/(size(y, 1)*n_test);
elseif strcmp(task_type, 'mlr')
    loss = norm(out - y, 'fro')^2/n_test;
    err = norm(out - y, 'fro')^2/(size(y, 1)*n_test);
end

if strcmpi(type, 'test')
    model.test_err(end + 1) = gather(err);
    model.test_loss(end + 1) = gather(loss);
    model.test_complexity(end + 1) = gather(model.tau_A * norm(model.weights, 'fro')^2);
    model.test_unlabeled(end + 1) = gather(model.tau_I * trace(model.weights' * XLX * model.weights));
    model.test_trace(end + 1) =  gather(model.tau_S * sum(sqrt(eig(model.S' * model.S))));
    model.test_objective(end + 1) = gather(model.test_loss(end) + model.test_complexity(end) + model.test_unlabeled(end) + model.test_trace(end));
elseif strcmpi(type, 'train')
    model.train_err(end + 1) = gather(err);
    model.train_loss(end + 1) = gather(loss);
    model.train_complexity(end + 1) = gather(model.tau_A * norm(model.weights, 'fro')^2);
    model.train_unlabeled(end + 1) = gather(model.tau_I * trace(model.weights' * XLX * model.weights));
    model.train_trace(end + 1) =  gather(model.tau_S * sum(sqrt(eig(model.S' * model.S))));
    model.train_objective(end + 1) = gather(model.train_loss(end) + model.train_complexity(end) + model.train_unlabeled(end) + model.train_trace(end));
end
end