from tdata.functional.sackmann import get_data
import sys
import numpy as np
from os.path import join
from sklearn.preprocessing import LabelEncoder
from ml_tools.stan import load_stan_model_cached


start_year = int(sys.argv[1])
tour = sys.argv[2]
target_dir = '/data/cephfs/punim0592/tennis/stan_fits_surface'

assert tour in ['atp', 'wta']

# df = get_data(
#     '/Users/ingramm/Projects/tennis/tennis-data/data/sackmann/tennis_atp/')
df = get_data(f'/home/martiningram/data/tennis_{tour}', tour=tour)

to_use = df[df['tourney_date'].dt.year >= start_year]

to_use = to_use[to_use['surface'] != 'None']

# TODO: Could reconsider dropping carpet
to_use = to_use[to_use['surface'] != 'Carpet']

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
n = np.concatenate([winner_serve_hit.values, loser_serve_hit.values])
p = np.concatenate([winner_serve_won.values, loser_serve_won.values])

encoder = LabelEncoder()

server_ids = encoder.fit_transform(server_names)
returner_ids = encoder.transform(returner_names)

n_players = len(encoder.classes_)

surf_enc = LabelEncoder()

surf_ids = surf_enc.fit_transform(to_use['surface'])
surf_ids = np.concatenate([surf_ids, surf_ids])

model = load_stan_model_cached('surface_model.stan')

model_data = {
    'n_matches': server_ids.shape[0],
    'n_players': len(encoder.classes_),
    'n_surfaces': len(surf_enc.classes_),
    'n': n.astype(int),
    'p': p.astype(int),
    'server_id': server_ids + 1,
    'returner_id': returner_ids + 1,
    'surface_id': surf_ids + 1
}

fit_results = model.sampling(data=model_data)

print(fit_results,
      file=open(join(
          target_dir, f'stan_surface_model_results_{start_year}_{tour}.txt'),
          'w'))

np.savez(join(target_dir, f'stan_samples_{start_year}_{tour}.npz'),
         **fit_results.extract(), player_names=encoder.classes_,
         surface_names=surf_enc.classes_)
