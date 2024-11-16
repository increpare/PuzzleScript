import json
import os

from lark import Lark, Transformer, Tree, Token

with open("syntax.lark", "r") as file:
    puzzlescript_grammar = file.read()

# Initialize the Lark parser with the PuzzleScript grammar
parser = Lark(puzzlescript_grammar, start="ps_game")

class GameTransformer(Transformer):
    def ps_game(self, items):
        return {"game": items}
    
    def prelude(self, items):
        return {"prelude": items}
    
    def prelude_data(self, items):
        keyword, value, _newline = items
        return {keyword: value}
    
    def objects_section(self, items):
        return {"objects": items}
    
    def rules_section(self, items):
        return {"rules": items}
    
    def legend_section(self, items):
        return {"legend": items}
    
    def collision_layers_section(self, items):
        return {"collision_layers": items}
    
    def sounds_section(self, items):
        return {"sounds": items}
    
    def levels_section(self, items):
        return {"levels": items}
    
    def win_conditions_section(self, items):
        return {"win_conditions": items}
    
    def object_data(self, items):
        name, *rest = items
        return {"object": name.children[0].value, "properties": rest}
    
    def sprite(self, items):
        # Assuming we want to return the sprite as a grid
        sprite_grid = []
        current_row = []
        for item in items:
            if isinstance(item, Tree) and item.data == "sprite_pixel":
                current_row.append(item.children[0].value)
            elif isinstance(item, Token) and item.type == "NEWLINE":  # NEWLINE token means row change
                sprite_grid.append(current_row)
                current_row = []
        if current_row:
            sprite_grid.append(current_row)
        return {"sprite": sprite_grid}

    def legend_data(self, items):
        # Check if comment
        if isinstance(items[0], Token) and items[0].type == "COMMENT":
            return {"comment": items[0].value}
        legend_key, obj_name, *rest = items
        return {"key": legend_key.children[0].value, "object": obj_name.children[0].value}

    def rule_data(self, items):
        rule_parts = []
        for part in items:
            if isinstance(part, Tree) and part.data == "rule_part":
                rule_content = []
                for subpart in part.children:
                    if isinstance(subpart, Token):
                        rule_content.append(subpart.value)
                    elif isinstance(subpart, Tree):
                        rule_content.append(subpart.children[0].value)
                rule_parts.append(rule_content)
        return {"rule": rule_parts}



# Parse a PuzzleScript file
def parse_puzzlescript_file(filename):
    with open(filename, "r") as file:
        content = file.read()
    
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

# Usage example
if __name__ == "__main__":
    # Replace 'your_puzzlescript_file.txt' with the path to your PuzzleScript file
    demo_games_dir = os.path.join('script-doctor','games')
    parsed_games_filename = "parsed_games.txt"
    if not os.path.exists(parsed_games_filename):
        with open(parsed_games_filename, "w") as file:
            file.write("")
    with open(parsed_games_filename, "r") as file:
        # Get the set of all lines from this text file
        parsed_games = set(file.read().splitlines())
    for i, filename in enumerate(['blank.txt'] + os.listdir(demo_games_dir)):
        if filename in parsed_games:
            print(f"Skipping {filename}")
            continue
        if filename.endswith('.txt'):
            print(f"Processing {filename}")
            parse_tree = parse_puzzlescript_file(os.path.join(demo_games_dir, filename))
            breakpoint()
            print(f"    Parsed game {i} successfully:")
            with open("parsed_games.txt", "a") as file:
                file.write(filename + "\n")

