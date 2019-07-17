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
        n_sample=size(X_train,1);
        % knn
        start=tic();
        knn=knnsearch(X_train,X_train,'K',K+1);
        time=toc(start);
        disp([num2str(K), '-NN using: ', num2str(time)]);

        p_sigma = 0;
        for i_sample = 1 : n_sample
            p_sigma = p_sigma + norm(X_train(i_sample,:)-X_train(knn(i_sample, end),:), 2) / n_sample;
        end

        %% directed graph -------
        W=sparse(n_sample, n_sample);
        for i=1:n_sample
            for j=2:K+1
                col=knn(i,j);
                if ~W(i,col)
                    if ismember(i, knn(col, :))
                        W(i,col) = 2*exp(-norm(X_train(i,:)-X_train(col,:), 2)^2/p_sigma^2);
                    else
                        W(i,col) = exp(-norm(X_train(i,:)-X_train(col,:), 2)^2/p_sigma^2);
                    end
                    W(col,i)=W(i,col);
                end
            end
        end
        %D=sum(W)';
        % D=spdiags(sqrt(1./D), 0, n_sample, n_sample);
        % L=speye(n_sample)-D*W*D;
        D = spdiags(sum(W)', 0, n_sample, n_sample);
        L = D - W;
        save(str, 'L');
    end

    disp(['generate L using: ',num2str(toc)]);
end