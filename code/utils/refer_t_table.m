function t_value = refer_t_table(repeats)
    t_value = 2.015;
    if repeats == 5
        t_value = 2.015;
    elseif repeats == 10
        t_value = 1.812;
    elseif repeats == 20
        t_value = 1.725;
    elseif repeats == 30
        t_value = 1.697;
    end
end