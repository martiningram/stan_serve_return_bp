data {
    int n_matches; // Number of (serve) matches
    int n_players;
    int n_surfaces;
    
    int n[n_matches];
    int p[n_matches];
    
    int server_id[n_matches];
    int returner_id[n_matches];
    int surface_id[n_matches];
}
parameters {
    vector<lower=0>[n_surfaces] tau_serve;
    vector<lower=0>[n_surfaces] tau_return;

    matrix[n_surfaces, n_players] z_serve_surf;
    matrix[n_surfaces, n_players] z_return_surf;
    
    cholesky_factor_corr[n_surfaces] L_Omega_serve;
    cholesky_factor_corr[n_surfaces] L_Omega_return;
    
    vector[n_surfaces] intercepts;
    
    vector[n_players] serve_skills_raw;
    vector[n_players] return_skills_raw;
    
    real<lower=0> serve_skills_sd;
    real<lower=0> return_skills_sd;
}
transformed parameters {
    matrix[n_players, n_surfaces] beta_serve;
    matrix[n_players, n_surfaces] beta_return;
    vector[n_players] serve_skills;
    vector[n_players] return_skills;
    
    beta_serve = (diag_pre_multiply(tau_serve, L_Omega_serve) * z_serve_surf)';
    beta_return = (diag_pre_multiply(tau_return, L_Omega_return) * z_return_surf)';
    
    serve_skills = serve_skills_raw * serve_skills_sd;
    return_skills = return_skills_raw * return_skills_sd;
}
model {
    // Priors
    to_vector(z_serve_surf) ~ std_normal();
    to_vector(z_return_surf) ~ std_normal();
    
    L_Omega_serve ~ lkj_corr_cholesky(1);
    L_Omega_return ~ lkj_corr_cholesky(1);
    
    serve_skills_raw ~ std_normal();
    return_skills_raw ~ std_normal();
    
    intercepts ~ normal(0, 1);
    
    tau_serve ~ normal(0, 0.5);
    tau_return ~ normal(0, 0.5);
    
    serve_skills_sd ~ normal(0, 1);
    return_skills_sd ~ normal(0, 1);
    
    // Likelihood
    {
    vector[n_matches] match_skills;
    for (i in 1:n_matches) {
        match_skills[i] = serve_skills[server_id[i]] - return_skills[returner_id[i]] 
                          + beta_serve[server_id[i], surface_id[i]] 
                          - beta_return[returner_id[i], surface_id[i]] 
                          + intercepts[surface_id[i]];
    }
    p ~ binomial_logit(n, match_skills);
    }
}
generated quantities {
    matrix[n_players, n_surfaces] predicted_serve_probs;
    matrix[n_players, n_surfaces] predicted_return_probs;
    
    for (i in 1:n_players) {
        for (j in 1:n_surfaces) {
            predicted_serve_probs[i, j] = inv_logit(serve_skills[i] + beta_serve[i, j] + intercepts[j]);
            predicted_return_probs[i, j] = 1 - inv_logit(intercepts[j] - return_skills[i] - beta_return[i, j]);
    }
    }
    
}
