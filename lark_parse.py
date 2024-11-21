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

with open("syntax_loose.lark", "r") as file:
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
        items = [item for item in items if not (isinstance(item, Tree) and item.data == "newlines_or_comments")]
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

    def levelline(self, items):
        return items[0]

    def levellines(self, items):
        grid = []
        level_lines = items
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
def preprocess_ps(parser, filename):
    with open(filename, "r") as file:
        content = file.read()

    ## Preprocess the content of the file, stripping any comments
    content = strip_comments(content)
    # any double newlines
    content = content.replace('\n\n\n', '\n\n')
    # and any lines that are just `=+`
    content = re.sub(r'^=+\n', '', content, flags=re.MULTILINE)

    txt = txt.replace('\u00A0', ' ')
    # If the file does not end with a newline, add one
    if not txt.endswith("\n"):
        txt += "\n"
    return txt

 

    # except Exception as e:
    #     print(f"Error parsing file: {e}")

def strip_comments(text):
    new_text = ""
    n_open_brackets = 0
    # Move through the text, keeping track of how deep we are in brackets
    for c in text:
        if c == "(":
            n_open_brackets += 1
        elif c == ")":
            # we ignore unmatched closing brackets if we are outside
            n_open_brackets = max(0, n_open_brackets - 1)
        elif n_open_brackets == 0:
            new_text += c
    return new_text

games_to_skip = set({'easyenigma.txt'})

# Usage example
if __name__ == "__main__":
    # Initialize the Lark parser with the PuzzleScript grammar
    parser = Lark(puzzlescript_grammar, start="ps_game", maybe_placeholders=False)
    # min_parser = Lark(min_puzzlescript_grammar, start="ps_game")

    # games_dir = os.path.join('script-doctor','games')
    games_dir = 'scraped_games'

    parsed_games_filename = "parsed_games.txt"
    # min_grammar = os.path.join('syntax_generate.lark')
    if args.overwrite or not os.path.exists(parsed_games_filename):
        with open(parsed_games_filename, "w") as file:
            file.write("")
    with open(parsed_games_filename, "r", encoding='utf-8') as file:
        # Get the set of all lines from this text file
        parsed_games = set(file.read().splitlines())
    if not os.path.isdir('pretty_trees'):
        os.mkdir('pretty_trees')
    # for i, filename in enumerate(['blank.txt'] + os.listdir(demo_games_dir)):
    game_files = os.listdir(games_dir)
    # sort them alphabetically
    game_files.sort()
    for i, filename in enumerate(game_files):
        if filename in parsed_games or filename in games_to_skip:
            print(f"Skipping {filename}")
            print(f"Processing {filename}")
            # try:

            # Now save the simpolified version of the file
            simp_filename = filename.strip('.txt') + '_simplified.txt'
            print(simp_filename)
            if not os.path.exists(simp_filename):
                content = preprocess_ps(os.path.join(games_dir, filename))
                with open(simp_filename, "w") as file:
                    file.write(content)
            else:
                with open(simp_filename, "r") as file:
                    content = file.read()
            parse_tree = parser.parse(content)

            # except Exception as e:
                # breakpoint()
                # with open("parsed_games.txt", "a") as file:
                #     file.write(filename + "\n")
                # continue
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

