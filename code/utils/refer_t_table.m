function t_value = refer_t_table(repeats)
    % 90 level
    t_value = 1.476;
    if repeats == 5
        t_value = 1.476;
    elseif repeats == 10
        t_value = 1.372;
    elseif repeats == 20
        t_value = 1.325;
    elseif repeats == 30
        t_value = 1.310;
    end
end