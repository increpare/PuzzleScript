import argparse
import json
import os

parser = argparse.ArgumentParser(description='Parse PuzzleScript files')
parser.add_argument('--overwrite', '-o', action='store_true', help='Overwrite existing parsed_games.txt')
args = parser.parse_args()

from lark import Lark, Transformer, Tree, Token
import numpy as np

with open("syntax.lark", "r") as file:
    puzzlescript_grammar = file.read()

# Initialize the Lark parser with the PuzzleScript grammar
parser = Lark(puzzlescript_grammar, start="ps_game")

class GameTransformer(Transformer):
    pass
    # def ps_game(self, items):
    #     return {"game": items}
    
    # def prelude(self, items):
    #     return {"prelude": prelude}
    
    # def prelude_data(self, items):
    #     keyword, value, _newline = items
    #     return {keyword: value}
    
    def objects_section(self, items):
        return [obj for obj in items if type(obj) == Tree and obj.data == "object_data"]
    
    # def rules_section(self, items):
    #     return {"rules": items}
    
    # def legend_section(self, items):
    #     return {"legend": items}
    
    # def collision_layers_section(self, items):
    #     return {"collision_layers": items}
    
    # def sounds_section(self, items):
    #     return {"sounds": items}
    
    # def levels_section(self, items):
    #     return {"levels": items}
    
    # def win_conditions_section(self, items):
    #     return {"win_conditions": items}
    
    # def object_data(self, items):
    #     name, *rest = items
    #     return {"object": name.children[0].value, "properties": rest}
    
    def sprite(self, items):
        # Assuming we want to return the sprite as a grid
        sprite_grid = []
        current_row = []
        for item in items:
            if item.type == "SPRITE_PIXEL":
                current_row.append(item.value)
            elif item.type == "NEWLINE":  # NEWLINE token means row change
                sprite_grid.append(current_row)
                current_row = []
        if current_row:
            sprite_grid.append(current_row)
        return {"sprite": np.array(sprite_grid)}
    
    def levels_section(self, items):
        breakpoint()
        return [level for level in items if 
                (type(level) == Tree
                and level.data == "level_data" 
                and level.children[0].data == "levellines")]

    def levellines(self, items):
        level_grid, level_row = [], []
        for item in items:
            if isinstance(item, Token) and item.type == "NEWLINE":
                continue
            if item.type == "LEVELLINE":
                level_grid.append(np.array(item.value.strip()))
        level_grid = np.array(level_grid)
        return {"levellines": level_grid} 

    def levelline(self, items):
        level_row = []
        breakpoint()
        for item in items:
            if isinstance(item, Token) and item.type == "PIXEL":
                level_row.append(item.value)
        return level_row
    # def legend_data(self, items):
    #     # Check if comment
    #     if isinstance(items[0], Token) and items[0].type == "COMMENT":
    #         return {"comment": items[0].value}
    #     legend_key, obj_name, *rest = items
    #     return {"key": legend_key.children[0].value, "object": obj_name.children[0].value}

    # def rule_data(self, items):
    #     rule_parts = []
    #     for part in items:
    #         if isinstance(part, Tree) and part.data == "rule_part":
    #             rule_content = []
    #             for subpart in part.children:
    #                 if isinstance(subpart, Token):
    #                     rule_content.append(subpart.value)
    #                 elif isinstance(subpart, Tree):
    #                     rule_content.append(subpart.children[0].value)
    #             rule_parts.append(rule_content)
    #     return {"rule": rule_parts}



# Parse a PuzzleScript file
def parse_puzzlescript_file(filename):
    with open(filename, "r") as file:
        content = file.read()
    
    content = preprocess_ps_txt(content)
    
    # try:
    # Parse the content of the file
    parse_tree = parser.parse(content)
    # Optionally transform the parse tree to a more usable form
    # transformed_tree = GameTransformer().transform(parse_tree)
    
    # Print or return the transformed parse tree
    # print(parse_tree.pretty())
    return parse_tree

    # except Exception as e:
    #     print(f"Error parsing file: {e}")

games_to_skip = set({'easyenigma.txt'})

def preprocess_ps_txt(txt):
    # If the file does not end with a newline, add one
    if not txt.endswith("\n"):
        txt += "\n"
    return txt

# Usage example
if __name__ == "__main__":
    # Replace 'your_puzzlescript_file.txt' with the path to your PuzzleScript file
    demo_games_dir = os.path.join('script-doctor','games')
    parsed_games_filename = "parsed_games.txt"
    if args.overwrite or not os.path.exists(parsed_games_filename):
        with open(parsed_games_filename, "w") as file:
            file.write("")
    with open(parsed_games_filename, "r") as file:
        # Get the set of all lines from this text file
        parsed_games = set(file.read().splitlines())
    for i, filename in enumerate(['blank.txt'] + os.listdir(demo_games_dir)):
        if filename in parsed_games or filename in games_to_skip:
            print(f"Skipping {filename}")
            continue
        if filename.endswith('.txt'):
            print(f"Processing {filename}")
            parse_tree = parse_puzzlescript_file(os.path.join(demo_games_dir, filename))
            print(parse_tree.pretty())
            # parse_tree = GameTransformer().transform(parse_tree)
            # prelude, objects, legend, sounds, collision_layers, rules, win_conditions, levels = parse_tree.children
            # [print(obj.pretty()) for obj in objects]
            # [print(level.pretty()) for level in levels]
            # breakpoint()
            print(f"    Parsed game {i} successfully:")
            with open("parsed_games.txt", "a") as file:
                file.write(filename + "\n")

