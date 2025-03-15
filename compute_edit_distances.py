import json
import Levenshtein
import os


seed = 21

sweep_dir = os.path.join('logs', f'sweep-{seed}')
with open(os.path.join(sweep_dir, 'stats.json')) as f:
    stats = json.load(f)

with open('example_games.json') as f:
    example_games = json.load(f)

for cot in [0, 1]:
    for fewshot in [0, 1]:
        key = f"fewshot-{fewshot}_cot-{cot}"
        for game_i, game_stat in enumerate(stats[key]):
            code = game_stat['code']
            dists = []
            for ex_game in example_games:
                edit_distance = Levenshtein.distance(code, ex_game)
                print(edit_distance)
                dists.append(edit_distance)
            game_min_dist = min(dists)
            stats[key][game_i]['min_dist'] = game_min_dist

with open(os.path.join(sweep_dir, 'stats_and_dists.json'), 'w') as f:
    json.dump(stats, f, indent=4)
