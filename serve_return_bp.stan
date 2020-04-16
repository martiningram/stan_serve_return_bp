data {
    int n_matches;
    int n_players;
    int n_normal[n_matches];
    int p_normal[n_matches];
    int n_bp[n_matches];
    int p_bp[n_matches];

    int server_id[n_matches];
    int returner_id[n_matches];
}
parameters {
    real intercept;
    real bp_intercept;

    real<lower=0> serve_sd;
    real<lower=0> return_sd;

    real<lower=0> serve_add_sd;
    real<lower=0> return_add_sd;

    vector[n_players] serve_skills_raw;
    vector[n_players] return_skills_raw;
    vector[n_players] serve_additions_bp_raw;
    vector[n_players] return_additions_bp_raw;
}
transformed parameters {
    vector[n_players] serve_skills = serve_skills_raw * serve_sd;
    vector[n_players] return_skills = return_skills_raw * return_sd;

    vector[n_players] serve_additions_bp = serve_additions_bp_raw * serve_add_sd;
    vector[n_players] return_additions_bp = return_additions_bp_raw * return_add_sd;
}
model {
    vector[n_matches] predictions;
    vector[n_matches] bp_predictions;

    serve_sd ~ normal(0, 1);
    return_sd ~ normal(0, 1);
    serve_add_sd ~ normal(0, 1);
    return_add_sd ~ normal(0, 1);

    serve_skills_raw ~ normal(0, 1);
    return_skills_raw ~ normal(0, 1);
    serve_additions_bp_raw ~ normal(0, 1);
    return_additions_bp_raw ~ normal(0, 1);

    predictions = serve_skills[server_id] - return_skills[returner_id] + intercept;
    bp_predictions = predictions + serve_additions_bp[server_id] - return_additions_bp[returner_id] + bp_intercept;

    p_normal ~ binomial_logit(n_normal, predictions);
    p_bp ~ binomial_logit(n_bp, bp_predictions);
}

