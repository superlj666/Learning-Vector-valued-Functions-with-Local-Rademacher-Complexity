% Reference : Robust multi-class transductive learning with graphs
function L = construct_laplacian_graph(data_name, X_train, K)
    tic();
    str=['../data/',data_name,'/laplacian_',num2str(K),'.mat'];
    if exist(str,'file')
        load(str);
    else
        if ~exist(['../data/',data_name], 'dir')
            mkdir(['../data/',data_name]);
        end
        n_sample=size(X_train, 2);
        % knn
        start=tic();
        knn=knnsearch(X_train', X_train','K',K+1);
        time=toc(start);
        disp([num2str(K), '-NN using: ', num2str(time)]);

       %% directed graph -------
        W=sparse(n_sample, n_sample);

        for col=1:n_sample
            W(col, knn(col, :))= true;
        end
        W = W&W';
        
        D = spdiags(sum(W)', 0, n_sample, n_sample);
        L = D - W;
        save(str, 'L');
    end

    disp(['generate L using: ',num2str(toc)]);
end