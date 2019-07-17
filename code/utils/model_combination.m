function model_out = model_combination(model_a, model_b)
    model_out = model_a;
    model_out.tau_A = model_b.tau_A;
    model_out.tau_I = model_b.tau_I;
    model_out.tau_S = model_b.tau_S;
    model_out.step = model_b.step;
end