import numpy as np
from sklearn.preprocessing import LabelEncoder
from tdata.functional.sackmann import get_data
import tensorflow as tf
from svgp.tf.quadrature import expectation
from functools import partial
from ml_tools.flattening import flatten_and_summarise_tf, reconstruct_tf
from ml_tools.tensorflow import lo_tri_from_elements
from svgp.tf.kl import mvn_kl
from svgp.tf.mogp import create_ls
from ml_tools.lin_alg import pos_def_mat_from_vector
from ml_tools.normals import covar_to_corr
from scipy.optimize import minimize
from ml_tools.flattening import reconstruct_np


def log_binomial_lik(p, logits, n):
    # TODO: Check!
    return (p * tf.math.log_sigmoid(logits)
            + (n - p) * tf.math.log_sigmoid(-logits))


@tf.function
def compute_objective(elts_serve, elts_return, elts_prior_serve,
                      elts_prior_return, intercept, n, p, n_surfaces,
                      server_ids, returner_ids, surf_ids,
                      mean_surface_skills_serve, mean_surface_skills_return):

    prior_L_serve = lo_tri_from_elements(elts_prior_serve, n_surfaces)
    prior_L_return = lo_tri_from_elements(elts_prior_return, n_surfaces)

    prior_cov_serve = tf.einsum(
        'ik,kl->il', prior_L_serve, tf.transpose(prior_L_serve)) + \
        tf.eye(n_surfaces) * 1e-6
    prior_cov_return = tf.einsum(
        'ik,kl->il', prior_L_return, tf.transpose(prior_L_return)) + \
        tf.eye(n_surfaces) * 1e-6

    L_serve = create_ls(elts_serve, n_surfaces, elts_serve.shape[0])
    L_return = create_ls(elts_return, n_surfaces, elts_return.shape[0])

    cov_serve = tf.einsum(
        'ijk,ikl->ijl', L_serve, tf.transpose(L_serve, (0, 2, 1))) \
        + tf.eye(n_surfaces) * 1e-6

    cov_return = tf.einsum(
        'ijk,ikl->ijl', L_return, tf.transpose(L_return, (0, 2, 1))) + \
        tf.eye(n_surfaces) * 1e-6

    mean_serve_surface_skills = tf.gather_nd(mean_surface_skills_serve,
                                             tf.stack([server_ids, surf_ids],
                                                      axis=1))
    mean_return_surface_skills = tf.gather_nd(mean_surface_skills_return,
                                              tf.stack([returner_ids,
                                                        surf_ids], axis=1))

    var_serve_surface_skills = tf.gather_nd(
        cov_serve, tf.stack([server_ids, surf_ids, surf_ids], axis=1))

    var_return_surface_skills = tf.gather_nd(
        cov_return, tf.stack([returner_ids, surf_ids, surf_ids], axis=1))

    kl_serve = tf.reduce_sum(
        mvn_kl(mean_surface_skills_serve, cov_serve,
               tf.zeros(mean_surface_skills_return.shape[1]), prior_cov_serve,
               is_batch=True))

    kl_return = tf.reduce_sum(
        mvn_kl(mean_surface_skills_return, cov_return,
               tf.zeros(mean_surface_skills_return.shape[1]), prior_cov_return,
               is_batch=True))

    lik = partial(log_binomial_lik, n=n)

    pred_mean = (mean_serve_surface_skills - mean_return_surface_skills +
                 intercept)
    pred_var = var_serve_surface_skills + var_return_surface_skills

    expected_lik = expectation(p, pred_var, pred_mean, lik)

    return expected_lik - (kl_serve + kl_return)


# df = get_data(
#     '/Users/ingramm/Projects/tennis/tennis-data/data/sackmann/tennis_atp/')
df = get_data('/home/martiningram/data/tennis_atp')

to_use = df[df['tourney_date'].dt.year >= 1990]

to_use = to_use[to_use['surface'] != 'None']

to_use = to_use.dropna(subset=['w_1stWon', 'w_2ndWon', 'w_svpt', 'l_svpt'])

to_use = to_use[(to_use['w_svpt'] > 20) & (to_use['l_svpt'] > 20)]

winner_serve_won = to_use['w_1stWon'] + to_use['w_2ndWon']
winner_serve_hit = to_use['w_svpt']

loser_serve_won = to_use['l_1stWon'] + to_use['l_2ndWon']
loser_serve_hit = to_use['l_svpt']

winner_names = to_use['winner_name'].values
loser_names = to_use['loser_name'].values

server_names = np.concatenate([winner_names, loser_names])
returner_names = np.concatenate([loser_names, winner_names])
n = tf.constant(np.concatenate([winner_serve_hit.values,
                                loser_serve_hit.values]).astype(np.float32))
p = tf.constant(np.concatenate([winner_serve_won.values,
                                loser_serve_won.values]).astype(np.float32))

encoder = LabelEncoder()

server_ids = tf.constant(encoder.fit_transform(server_names))
returner_ids = tf.constant(encoder.transform(returner_names))

n_players = len(encoder.classes_)


surf_enc = LabelEncoder()

surf_ids = surf_enc.fit_transform(to_use['surface'])
surf_ids = tf.constant(np.concatenate([surf_ids, surf_ids]))


n_players, n_surfaces = len(encoder.classes_), len(surf_enc.classes_)
n_matches = len(surf_ids)

init_cov = np.eye(n_surfaces)
init_L = np.linalg.cholesky(init_cov)
init_elts = init_L[np.tril_indices_from(init_L)].astype(np.float32)

elts_serve = tf.tile([init_elts], (n_players, 1))
elts_return = tf.tile([init_elts], (n_players, 1))

elts_prior_serve = init_elts.copy()
elts_prior_return = init_elts.copy()

mean_surface_skills_serve = tf.zeros((n_players, n_surfaces))
mean_surface_skills_return = tf.zeros((n_players, n_surfaces))

init_theta = {
        'elts_serve': elts_serve,
        'elts_return': elts_return,
        'elts_prior_serve': elts_prior_serve,
        'elts_prior_return': elts_prior_return,
        'mean_surface_skills_serve': mean_surface_skills_serve,
        'mean_surface_skills_return': mean_surface_skills_return,
        'intercept': tf.constant(0.5)
}

flat_theta, summary = flatten_and_summarise_tf(**init_theta)


def to_optimise(flat_theta):

    flat_theta = tf.cast(tf.constant(flat_theta), tf.float32)

    with tf.GradientTape() as tape:

        tape.watch(flat_theta)

        theta = reconstruct_tf(flat_theta, summary)

        obj = -compute_objective(n=n, p=p, n_surfaces=n_surfaces,
                                 server_ids=server_ids,
                                 returner_ids=returner_ids, surf_ids=surf_ids,
                                 **theta)

        grad = tape.gradient(obj, flat_theta)

        print(obj, np.linalg.norm(grad.numpy()))

    print(np.round(covar_to_corr(pos_def_mat_from_vector(
        theta['elts_prior_serve'], n_surfaces)), 2))

    return obj.numpy().astype(np.float64), grad.numpy().astype(np.float64)


result = minimize(to_optimise, flat_theta.numpy().astype(np.float64),
                  method='L-BFGS-B', jac=True)

np.savez('surface_model_1990', players=encoder.classes_,
         surfaces=surf_enc.classes_, **reconstruct_np(result.x, summary))
