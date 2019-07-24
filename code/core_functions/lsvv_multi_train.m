function model = lsvv_multi_train(XLX, X_train, y_train, model)
%   Gradient Descent Train for Multi Class Learning
%
%   Inputs :
%   XLX -   2 - D d * d Precomputed Matrices, which is to take use of unlabeled data.
%   X_train   -   2 - D d * n Matrices, which is training features.
%   y_train   -   1 - D n * 1 Vector, which is trainning labels.
%
%   Additional parameters :
%   - model.model.tau_A  is the weight of the frobenius norm term of weight matrix W. It regulates
%     the complexity of the model.   
%     Default value is 0.01.
%   - model.model.tau_I is the weight of the trace norm term of Laplacain regularization. It regulates
%     the influnce of unlabeled data.
%     Default value is 0.01.
%   - model.model.tau_S is the weight of the tail sum of singular values. It regulates
%     the influnce of local Rademacher complexity.
%     Default value is 0.01.
%   - model.T is numer of training epochs for the batch stage.
%     Default value is 100.
%   - model.Test_batch is numer of batchations to test.
%     Default value is 0, meaning no testing.
%
%
%   Author : Jian Li
%   Date : 2019 / 02 / 05
%
tic();
%rng('default');

n_dimension = size(X_train, 1);
n_sample = size(X_train, 2);
if size(y_train, 1) == 1
    n_class = max(y_train);
else
    n_class = size(y_train, 1);
end

if ~isfield(model, 'tau_A'), model.tau_A = 0; end
if ~isfield(model, 'tau_I'), model.tau_I = 1e-8; end
if ~isfield(model, 'tau_S'), model.tau_S = 1e-6; end
if ~isfield(model, 'tail_start'), model.tail_start = ceil(min(n_class, n_dimension) * 0.5); end
if ~isfield(model, 'xi'), model.xi = 0.5; end
if ~isfield(model, 'varepsilon'), model.varepsilon = 1e-2; end
if ~isfield(model, 'n_batch'), model.n_batch = 32; end
if ~isfield(model, 'T'), model.T = 30; end
if ~isfield(model, 'iter_batch'), model.iter_batch = 0; end
if ~isfield(model, 'epoch'), model.epoch = 0; end
if ~isfield(model, 'time_train'), model.time_train = 0; end

if isa(X_train, 'gpuArray')
    W = gpuArray.rand(n_dimension, n_class);
else
    W = rand(n_dimension, n_class);
end

converge = false;

G = 0;
W_x = 0;
for epoch = 1 : model.T
    model.epoch = model.epoch + 1;
    idx_rand = randperm(n_sample);
    
    errTot = 0;
    lossTot = 0;
    n_update = 0;
    
    for i_batch = 1 : ceil(n_sample / model.n_batch)
        if isa(X_train, 'gpuArray')
            grad_g = zeros(n_dimension, n_class, 'gpuArray');
        else
            grad_g = zeros(n_dimension, n_class);
        end
        model.iter_batch = model.iter_batch + 1;
        
        for i_sample = (i_batch - 1) * model.n_batch + 1 : min(i_batch * model.n_batch, n_sample)
            i_idx = idx_rand(i_sample);
            
            % find true margin and predict margin
            h_x = W' * X_train(:, i_idx);
            if size(y_train, 1) == 1
                % multi-class or binary class
                margin_true = h_x(y_train(i_idx));
                h_x(y_train(i_idx)) = -Inf;
                [margin_pre, loc_pre] = max(h_x);

                % rough estimation of loss and error for every epoch
                errTot = errTot + (margin_true <= margin_pre);
                lossTot = lossTot + max(1-margin_true + margin_pre, 0);

                % calculate gradient for every instance
                if margin_true-margin_pre < 1
                    grad_g( : , y_train(i_idx)) = grad_g( : , y_train(i_idx)) -X_train(:, i_idx);
                    grad_g( : , loc_pre) = grad_g( : , loc_pre) + X_train(:, i_idx);
                    n_update = n_update + 1;
                end
            else
                % multi-label learning
                grad_g = grad_g + 2*X_train(:, i_idx)*(h_x - y_train(:, i_idx))';
            end
        end
        % update gradient for every batch
        grad_g = grad_g ./ model.n_batch + 2 * model.tau_A  * W + 2 * model.tau_I * XLX * W;
        
        % Adadelta
        G = model.xi*G + (1-model.xi)*norm(grad_g, 2)^2;
        i_step = sqrt(W_x + model.varepsilon)/sqrt(G + model.varepsilon);
        W = W - i_step* grad_g;
        W_x = model.xi*W_x + (1-model.xi)*norm(i_step* grad_g, 2)^2;
        
        % SGD
        %W = W - model.step*grad_g;

        % SVT with proximal gradient
        S = zeros(n_dimension, n_class);
        if model.tau_S ~= 0
            [U, S, V] = svd(W);
            model.tail_start = min(model.tail_start, min(n_dimension, n_class));
            for i_diag = model.tail_start : min(n_dimension, n_class)
                S(i_diag, i_diag) = max(0, S(i_diag, i_diag)-i_step * model.tau_S);
            end
            W = (U * S) * V';
        end
        model.S = S;
        
        if isfield(model, 'n_record_batch') && (ismember(model.iter_batch, model.n_record_batch) ...
                || (epoch == model.T && i_batch == ceil(n_sample / model.n_batch)))
            model.time_train = model.time_train + toc();
            model.weights = W;
            model = record_batch(XLX, X_train, y_train, model, 'train');
            if isfield(model, 'test_batch') && isfield(model, 'X_test') && isfield(model, 'y_test')
                model = record_batch(XLX, model.X_test, model.y_test, model, 'test');
            end
            tic();
        end
        % early stop only controled by test error
        if isfield(model, 'test_batch') && model.iter_batch > 0.5 * ceil(n_sample / model.n_batch) ...
                && isfield(model, 'test_err') && numel(model.test_err) > 10 ...
                && numel(unique(model.test_err(end - 5 : end))) == 1 ...
                && numel(unique(model.train_err(end - 5 : end))) == 1
            if isfield(model, 'n_record_batch')
                model.time_train = model.time_train + toc();
                model.weights = W;
                model = record_batch(XLX, X_train, y_train, model, 'train');
                if isfield(model, 'test_batch') && isfield(model, 'X_test') && isfield(model, 'y_test')
                    model = record_batch(XLX, model.X_test, model.y_test, model, 'test');
                end
                tic();
            end
            converge = true;
            fprintf("early stop at %d epoch.\n", model.epoch);
            break;
        end
    end
    
    %fprintf('#Batch : %5.0f(epoch %3.0f)\tAER : %2.2f\tAEL : %2.2f\tUpdates : %5.0f\n', ...
    %model.iter_batch, epoch, errTot / n_sample * 100, lossTot / n_sample, n_update);
    if converge == true
        break;
    end
    
end

model.weights = W;
model.time_train = model.time_train + toc();
end