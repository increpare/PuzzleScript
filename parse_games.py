import json
import os
import re

from utils import num_tokens_from_string

def extract_sprites(content):
    # Regex to find the OBJECTS section
    objects_section = re.search(r'OBJECTS\s*=+\s*(.*?)\s*=+\s*LEGEND', content, re.DOTALL).group(1).strip()

    # Regex to extract individual object definitions
    object_definitions = re.findall(r'(\w+)\n([^\n]+)\n((?:[.\d]+\n)+)', objects_section)

    objects = {}
    # object_strs = set({})
    for name, color, pattern in object_definitions:
        pattern_lines = pattern.strip().split("\n")
        objects[name] = {
            "color": color,
            "pattern": pattern_lines
        }
        # object_str = f"{name}\n{color}\n{pattern}"
        # object_strs.add(object_str)

    return objects


if __name__ == '__main__':
    example_games = []
    example_sprites = {}
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
        new_sprites = extract_sprites(game_code)
        for new_sprite in new_sprites:
            if new_sprite not in example_sprites:
                example_sprites[new_sprite] = [new_sprites[new_sprite],]
            else:
                # print(f'Warning: Sprite {new_sprite} already exists. Going with: ' +\
                #       f'{example_sprites[new_sprite]}\nover: {new_sprites[new_sprite]}')
                example_sprites[new_sprite].append(new_sprites[new_sprite])

    json.dump(example_games, open('example_games.json', 'w'), indent=2)

    json.dump(example_sprites, open('example_sprites.json', 'w'), indent=2)
    example_sprite_names ='\n'.join(list(example_sprites.keys()))
    with open('example_sprite_names.txt', 'w', encoding='utf-8') as f:
        f.write(example_sprite_names)
    
    n_tokens = num_tokens_from_string(example_sprite_names, 'gpt-4o')
    print(f'example_sprite_names: {n_tokens} tokens')
