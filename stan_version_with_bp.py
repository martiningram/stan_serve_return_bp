import sys
from tdata.functional.sackmann import get_data
import numpy as np
from sklearn.preprocessing import LabelEncoder
from ml_tools.stan import load_stan_model_cached

df = get_data('/home/martiningram/data/tennis_atp')

start_year = int(sys.argv[1])
target_dir = '/data/cephfs/punim0592/tennis/stan_fits_serve'

to_use = df[df['tourney_date'].dt.year >= start_year]

to_use = to_use.dropna(subset=['w_1stWon', 'w_2ndWon', 'w_svpt', 'l_svpt'])

to_use = to_use[(to_use['w_svpt'] > 20) & (to_use['l_svpt'] > 20)]
to_use = to_use[(to_use['l_bpSaved'] >= 0) & (to_use['w_bpSaved'] >= 0)]

winner_serve_won = to_use['w_1stWon'] + to_use['w_2ndWon']
winner_serve_hit = to_use['w_svpt']

loser_serve_won = to_use['l_1stWon'] + to_use['l_2ndWon']
loser_serve_hit = to_use['l_svpt']

winner_bp_faced = to_use['w_bpFaced']
winner_bp_won = to_use['w_bpSaved']

loser_bp_faced = to_use['l_bpFaced']
loser_bp_won = to_use['l_bpSaved']

winner_non_bp_serve_won = winner_serve_won - winner_bp_won
winner_non_bp_serve_hit = winner_serve_hit - winner_bp_faced

loser_non_bp_serve_won = loser_serve_won - loser_bp_won
loser_non_bp_serve_hit = loser_serve_hit - loser_bp_faced

winner_names = to_use['winner_name'].values
loser_names = to_use['loser_name'].values

server_names = np.concatenate([winner_names, loser_names])
returner_names = np.concatenate([loser_names, winner_names])
n_normal = np.concatenate([winner_non_bp_serve_hit.values,
                           loser_non_bp_serve_hit.values]).astype(np.float32)
p_normal = np.concatenate([winner_non_bp_serve_won.values,
                           loser_non_bp_serve_won.values]).astype(np.float32)

n_bp = np.concatenate([winner_bp_faced.values,
                       loser_bp_faced.values]).astype(np.float32)
p_bp = np.concatenate([winner_bp_won.values,
                       loser_bp_won.values]).astype(np.float32)

encoder = LabelEncoder()

server_ids = encoder.fit_transform(server_names)
returner_ids = encoder.transform(returner_names)

n_players = len(encoder.classes_)

# Fit a model in Stan to compare
model = load_stan_model_cached('serve_return_bp.stan')

model_data = {
    'n_matches': server_ids.shape[0],
    'n_players': len(encoder.classes_),
    'n_normal': n_normal.astype(int),
    'p_normal': p_normal.astype(int),
    'n_bp': n_bp.astype(int),
    'p_bp': p_bp.astype(int),
    'server_id': server_ids + 1,
    'returner_id': returner_ids + 1
}

stan_fit_result = model.sampling(data=model_data)
print(stan_fit_result, file=open(target_dir +
                                 f'/stan_results_{start_year}.txt', 'w'))
np.savez(target_dir + f'/stan_fit_{start_year}.npz',
         player_names=encoder.classes_, **stan_fit_result.extract())
