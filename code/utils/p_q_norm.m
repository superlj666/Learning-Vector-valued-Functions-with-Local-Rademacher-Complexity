function norm_result = p_q_norm(M, p, q)
    row_size = size(M, 1);
    column_size = size(M, 2);
    
    norm_result = 0;
    for j = 1:column_size
        p_norm = 0;
        for i = 1:row_size
            p_norm = p_norm + abs(M(i,j))^p;
        end
        norm_result = norm_result + p_norm^(q/p);
    end
    norm_result = norm_result^(1/q);
end