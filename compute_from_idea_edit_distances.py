import json
import Levenshtein
import os


seed = 12
max_gen_attempts = 20

sweep_dir = os.path.join('logs', f'sweep-{seed}', 'fromIdea')

with open(os.path.join(sweep_dir, 'stats.json')) as f:
    stats = json.load(f)

with open('example_games.json') as f:
    example_games = json.load(f)

fewshot = 1
for cot in [0, 1]:
    from_idea = 1
    key = ("fromIdea-1_" if from_idea else "") + f"fewshot-{fewshot}_cot-{cot}"
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
