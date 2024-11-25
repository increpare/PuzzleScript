import argparse
import json
import os
import re

import lark
from lark.reconstruct import Reconstructor

parser = argparse.ArgumentParser(description='Parse PuzzleScript files')
parser.add_argument('--overwrite', '-o', action='store_true', help='Overwrite existing parsed_games.txt')
args = parser.parse_args()

from lark import Lark, Transformer, Tree, Token, Visitor
import numpy as np

with open("syntax.lark", "r") as file:
    puzzlescript_grammar = file.read()
with open("syntax_generate.lark", "r") as file:
    min_puzzlescript_grammar = file.read()

class StripPuzzleScript(Transformer):
    """
    Reduces the parse tree to a minimal functional version of the grammar.
    """
    def message(self, items):
        return None

    def strip_newlines_data(self, items, data_name):
        """Remove any instances of section data that are newlines/comments"""
        items = [item for item in items if not (isinstance(item, Token) and item.type == "NEWLINES_OR_COMMENTS")]
        if len(items) > 0:
            return Tree(data_name, items)

    def strip_section_items(self, items, data_name):
        """Remove any empty section items (e.g. resulting from returning None above, when encountering a datum that is all newlines/comments)"""
        return [item for item in items if isinstance(item, Tree) and item.data == data_name]        

    def ps_game(self, items):
        items = [item for item in items if type(item) == Tree]
        return Tree('ps_game', items)

    def objects_section(self, items):
        return Tree('objects_section', self.strip_section_items(items, 'object_data'))

    def legend_section(self, items):
        return Tree('legend_section', self.strip_section_items(items, 'legend_data'))

    def levels_section(self, items):
        items = self.strip_section_items(items, 'level_data')
        items = [i for i in items if i]
        return Tree('levels_section', items)

    def winconditions_section(self, items):
        return Tree('winconditions_section', self.strip_section_items(items, 'condition_data'))
    
    def collisionlayers_section(self, items):
        return Tree('collisionlayers_section', self.strip_section_items(items, 'layer_data'))

    def rules_section(self, items):
        return Tree('rules_section', self.strip_section_items(items, 'rule_block'))

    def sounds_section(self, items):
        return

    def prelude(self, items):
        return

    def object_data(self, items):
        return self.strip_newlines_data(items, 'object_data')

    def level_data(self, items):
        return self.strip_newlines_data(items, 'level_data')
        # Remove any Tokens

    def legend_data(self, items):
        return self.strip_newlines_data(items, 'legend_data')
    
    def rule_data(self, items):
        return self.strip_newlines_data(items, 'rule_data')

    # def rule_block(self, items):
    #     return items[0]
    
    # def rule_part(self, items):
    #     return items[0]

    def rule_block_once(self, items):
        return Tree('rule_block_once', [i for i in items if i])

    def rule_block_loop(self, items):
        return Tree('rule_block_loop', [i for i in items if i])
    
    def condition_data(self, items):
        return self.strip_newlines_data(items, 'condition_data')

    def layer_data(self, items):
        return self.strip_newlines_data(items, 'layer_data')

    def shape_2d(self, items):
        # Create a 2D array of the items
        grid = []
        row = []
        for s in items:
            # If we encounter a newline, start a new row
            if s == "\n":
                if len(row) > 0:
                    grid.append(row)
                    row = []
            else:
                row.append(s.value)
        row_lens = [len(r) for r in grid]
        if len(set(row_lens)) > 1:
            breakpoint()
        grid = np.array(grid)
        return grid

    def sprite(self, items):
        # Remote any item that is a message
        items = [i for i in items if not (isinstance(i, Token) and i.type == 'COMMENT')]
        return Tree('sprite', self.shape_2d(items))

    def levellines(self, items):
        grid = []
        level_str = items[0].value
        # Split the level string into lines
        level_lines = level_str.strip().split("\n")        
        grid = [list(line) for line in level_lines]
        # padd all rows with empty tiles
        max_len = max(len(row) for row in grid)
        for row in grid:
            row += ['.'] * (max_len - len(row))
        return np.array(grid)


def array_2d_to_str(arr):
    s = ""
    for row in arr:
        s += "".join(row) + "\n"
    print(s)
    return s


class PrintPuzzleScript(Transformer):
    def ps_game(self, items):
        return "\n\n".join(items)

    def prefix(self, items):
        return items[0].value

    def colors(self, items):
        return ' '.join(items)

    def color(self, items):
        return items[0].value

    def rule_data(self, items):
        return ' '.join([item for item in items if item])

    def cell_border(self, items):
        return " | "

    def object_name(self, items):
        return items[0].value

    def command_keyword(self, items):
        return ' '.join(items)

    def sound(self, items):
        return ''

    def command(self, items):
        return ' '.join(items)

    def legend_operation(self, items):
        if len(items) == 1:
            return items[0]
        else:
            return ' or '.join(items)

    def legend_key(self, items):
        return items[0].value

    def object_name(self, items):
        return items[0].value

    def sprite(self, arr):
        return array_2d_to_str(arr)

    def level_data(self, items):
        return '\n'.join(items)

    def levellines(self, arr):
        return array_2d_to_str(arr)
    
    def object_data(self, items):
        return ''.join(items)
    
    def legend_data(self, items):
        return items[0].value + ' = ' + ' or '.join(items[1:])
    
    def layer_data(self, items):
        return ', '.join(items)

    def rule_part(self, items):
        return '[ ' + ''.join(items) + ' ]'

    def rule_block(self, items):
        return items[0]

    def rule_block_once(self, items):
        return ''.join([item for item in items if item])

    def rule_block_loop(self, items):
        return ''.join(items)

    def rule_content(self, items):
        return ' '.join(items)

    def levels_section(self, items):
        return 'LEVELS\n\n' + '\n'.join([i for i in items if i])

    def objects_section(self, items):
        return 'OBJECTS\n\n' + '\n'.join(items)
    
    def legend_section(self, items):
        return 'LEGEND\n\n' + '\n'.join(items)
    
    def rules_section(self, items):
        return 'RULES\n\n' + ''.join(items)

    def condition_id(self, items):
        return items[0].value

    def condition_data(self, items):
        return ' '.join(items)
    
    def winconditions_section(self, items):
        return 'WINCONDITIONS\n\n' + '\n'.join(items)
    
    def collisionlayers_section(self, items):
        return 'COLLISIONLAYERS\n\n' + '\n'.join(items)

    def level_data(self, arr):
        if arr[0] is not None:
            return array_2d_to_str(arr[0])



# Parse a PuzzleScript file
def parse_puzzlescript_file(parser, filename):
    with open(filename, "r") as file:
        content = file.read()
    
    content = strip_misc(content)
    content = strip_comments(content)
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

def strip_misc(text):
    # remove double newlines
    text = re.sub(r'\n\n+', '\n\n', text)
    """Remove any lines that are only `=+`."""
    return "\n".join(line for line in text.split("\n") if not re.match(r"^[=+]+$", line))

def strip_comments(text):
    """Remove any `(comment)` from the text"""
    n_open_parens = 0
    new_text = ""
    for c in text:
        if c == "(":
            n_open_parens += 1
        elif c == ")":
            n_open_parens -= 1
        elif n_open_parens == 0:
            new_text += c
    return new_text

# Usage example
if __name__ == "__main__":
    # Initialize the Lark parser with the PuzzleScript grammar
    parser = Lark(puzzlescript_grammar, start="ps_game", maybe_placeholders=False)
    min_parser = Lark(min_puzzlescript_grammar, start="ps_game")

    # Replace 'your_puzzlescript_file.txt' with the path to your PuzzleScript file
    # demo_games_dir = os.path.join('script-doctor','games')
    demo_games_dir = 'scraped_puzzles'

    parsed_games_filename = "parsed_games.txt"
    min_grammar = os.path.join('syntax_generate.lark')
    if args.overwrite or not os.path.exists(parsed_games_filename):
        with open(parsed_games_filename, "w") as file:
            file.write("")
    with open(parsed_games_filename, "r") as file:
        # Get the set of all lines from this text file
        parsed_games = set(file.read().splitlines())
    for i, filename in enumerate(os.listdir(demo_games_dir)):
        if filename in parsed_games or filename in games_to_skip:
            print(f"Skipping {filename}")
            continue
        if filename.endswith('.txt'):
            print(f"Processing {filename}")
            parse_tree = parse_puzzlescript_file(parser, os.path.join(demo_games_dir, filename))
            min_parse_tree = StripPuzzleScript().transform(parse_tree)
            pretty_parse_tree_str = min_parse_tree.pretty()
            pretty_tree_filename = os.path.join('pretty_trees', filename)
            with open(pretty_tree_filename, "w") as file:
                file.write(pretty_parse_tree_str)
            # print(min_parse_tree.pretty())
            ps_str = PrintPuzzleScript().transform(min_parse_tree)
            min_filename = os.path.join('min_games', filename)
            with open(min_filename, "w") as file:
                file.write(ps_str)

            print(f"    Parsed game {i} successfully:")
            with open("parsed_games.txt", "a") as file:
                file.write(filename + "\n")

