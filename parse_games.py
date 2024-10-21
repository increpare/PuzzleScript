import json
import os

from utils import num_tokens_from_string


if __name__ == '__main__':
    example_games = []
    for game_path in os.listdir('src/demo'):
        with open(f'src/demo/{game_path}', 'r', encoding='utf-8') as f:
            game_code = f.read()
        if not game_path.endswith('.txt'):
            print(f'Skipping {game_path}')
            continue
        n_tokens = num_tokens_from_string(game_code, 'gpt-4o')
        game_name = os.path.basename(game_path)
        example_games.append(game_code)
        print(f'{game_path}: {n_tokens} tokens')
    json.dump(example_games, open('example_games.json', 'w'), indent=2)
