import argparse
import json
import os
import pickle
import re
import traceback
import signal  # Add this import
import contextlib
from typing import Optional

import lark
from lark.reconstruct import Reconstructor

parser = argparse.ArgumentParser(description='Parse PuzzleScript files')
parser.add_argument('--overwrite', '-o', action='store_true', help='Overwrite existing parsed_games.txt')
args = parser.parse_args()
games_to_skip = set({'easyenigma', 'A_Plaid_Puzzle'})

# test_games = ['blockfaker', 'sokoban_match3', 'notsnake', 'sokoban_basic']
test_games = ['blockfaker', 'sokoban_basic', 'sokoban_match3', 'notsnake']

from lark import Lark, Transformer, Tree, Token, Visitor
import numpy as np


@contextlib.contextmanager
def timeout_handler(seconds: int):
    """Context manager for handling timeouts using signals"""
    def signal_handler(signum, frame):
        raise TimeoutError("Parsing timed out")
        
    # Store previous handler
    previous_handler = signal.getsignal(signal.SIGALRM)
    
    try:
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        yield
    finally:
        signal.alarm(0)  # Disable alarm
        signal.signal(signal.SIGALRM, previous_handler)  # Restore previous handler


with open("syntax.lark", "r", encoding='utf-8') as file:
    puzzlescript_grammar = file.read()
with open("syntax_generate.lark", "r", encoding='utf-8') as file:
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
        items = [item for item in items if not (isinstance(item, Token) and (item.type == "NEWLINES" or item.type == "NEWLINE"))]
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

    def prelude_data(self, items):
        return self.strip_newlines_data(items, 'prelude_data')

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
        line = [str(i) for i in items]
        assert line[-1] == "\n"
        return line[:-1]

    def levellines(self, items):
        grid = []
        level_lines = items
        grid = [line for line in level_lines[:-1]]
        # pad all rows with empty tiles
        max_len = max(len(row) for row in grid)
        for row in grid:
            row += ['.'] * (max_len - len(row))
        grid = np.array(grid)
        return grid


class RepairPuzzleScript(Transformer):
    def object_data(self, items):
        # If we're missing colors, add random ones
        child_trees = [i for i in items if isinstance(i, Tree)]
        child_tree_names = [i.data for i in child_trees]
        colors = [i for i in child_trees if i.data == 'colors'][0]
        n_colors = len(colors.children)
        if 'sprite' in child_tree_names:
            sprite = [i for i in child_trees if i.data == 'sprite'][0]
            sprite_arr = sprite.children
            unique_sprite_pixels = set(np.unique(sprite_arr))
            unique_sprite_pixels.discard('.')
            n_unique_pixels = len(unique_sprite_pixels)
            while n_unique_pixels - 1 > n_colors:
                colors.children.append(Token('COLOR', f'#{np.random.randint(0, 0xFFFFFF):06X}'))
                n_colors += 1
                breakpoint()
        return Tree('object_data', items)


def array_2d_to_str(arr):
    s = ""
    for row in arr:
        s += "".join(row) + "\n"
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

    def prelude(self, items):
        return '\n'.join(items)

    def prelude_data(self, items):
        return ' '.join(items)

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

    def legend_data(self, items):
        # Omitting final NEWLINES
        return str(items[0])[:-1].strip() + ' = ' + ' or '.join(items[1:])

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
        return array_2d_to_str(arr)[:-2]

    def level_data(self, items):
        return '\n'.join(items)

    def levellines(self, arr):
        return array_2d_to_str(arr)
    
    def object_data(self, items):
        return '\n'.join(items) + '\n'

    def object_line(self, items):
        return ' '.join(items)

    def color_line(self, items):
        return ' '.join(items)
     
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


def add_empty_sounds_section(txt):
    txt = re.sub(r'COLLISIONLAYERS', 'SOUNDS\n\nCOLLISIONLAYERS', txt)
    return txt


def preprocess_ps(txt):
    # Remove whitespace at end of any line
    txt = re.sub(r'[ \t]+$', '', txt, flags=re.MULTILINE)

    # Remove any lines that are just `=+`
    txt = re.sub(r'^=+\n', '', txt, flags=re.MULTILINE)

    txt = txt.replace('\u00A0', ' ')
    # If the file does not end with 2 newlines, fix this
    for i in range(2):
        if not txt.endswith("\n\n"):
            txt += "\n"

    # Remove any lines beginning with "message" (case insensitive)
    txt = re.sub(r'^message.*\n', '', txt, flags=re.MULTILINE | re.IGNORECASE)

    # Truncate lines ending with "message"
    txt = re.sub(r'message.*\n', '\n', txt, flags=re.MULTILINE | re.IGNORECASE)

    ## Strip any comments
    txt = strip_comments(txt)

    # Remove any lines that are just whitespace
    txt = re.sub(r'^\s*\n', '\n', txt, flags=re.MULTILINE)

    # any more-than-double newlines should be replaced by a single newline
    txt = re.sub(r'\n{3,}', '\n\n', txt)

    # Remove everything until "objects" (case insensitive)
    # txt = re.sub(r'^.*OBJECTS', 'OBJECTS', txt, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)

    return txt.lstrip()


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

data_dir = 'data'
games_dir = os.path.join(data_dir, 'scraped_games')
min_games_dir = os.path.join(data_dir, 'min_games')
simpd_dir = os.path.join(data_dir, 'simplified_games')
trees_dir = os.path.join(data_dir, 'game_trees')
pretty_trees_dir = os.path.join(data_dir, 'pretty_trees')
parsed_games_filename = os.path.join(data_dir, "parsed_games.txt")

# Usage example
if __name__ == "__main__":
    # Initialize the Lark parser with the PuzzleScript grammar
    parser = Lark(puzzlescript_grammar, start="ps_game", maybe_placeholders=False)
    # min_parser = Lark(min_puzzlescript_grammar, start="ps_game")

    # games_dir = os.path.join('script-doctor','games')

    os.makedirs(trees_dir, exist_ok=True)
    os.makedirs(pretty_trees_dir, exist_ok=True)
    os.makedirs(min_games_dir, exist_ok=True)
    # min_grammar = os.path.join('syntax_generate.lark')
    if args.overwrite or not os.path.exists(parsed_games_filename):
        with open(parsed_games_filename, "w") as file:
            file.write("")
    with open(parsed_games_filename, "r", encoding='utf-8') as file:
        # Get the set of all lines from this text file
        parsed_games = set(file.read().splitlines())
    # for i, filename in enumerate(['blank.txt'] + os.listdir(demo_games_dir)):
    game_files = os.listdir(games_dir)
    # sort them alphabetically
    game_files.sort()
    test_game_files = [f"{test_game}.txt" for test_game in test_games]
    game_files = test_game_files + game_files

    if not os.path.isdir(simpd_dir):
        os.mkdir(simpd_dir)
    scrape_log_dir = 'scrape_logs'
    if not os.path.isdir(scrape_log_dir):
        os.makedirs(scrape_log_dir)
    simpd_games = set(os.listdir(simpd_dir))
    for i, filename in enumerate(game_files):
        filepath = os.path.join(games_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            ps_text = f.read()
        simp_filename = filename[:-4] + '_simplified.txt' 
        if filename in parsed_games or os.path.basename(filename) in games_to_skip:
            print(f"Skipping {filepath}")
            continue

        print(f"Parsing game {filepath} ({i+1}/{len(game_files)})")
        simp_filepath = os.path.join(simpd_dir, simp_filename)
        if args.overwrite or not (simp_filename in simpd_games):
            # Now save the simplified version of the file
            content = preprocess_ps(ps_text)
            with open(simp_filepath, "w", encoding='utf-8') as file:
                file.write(content)
        else:
            with open(simp_filepath, "r", encoding='utf-8') as file:
                content = file.read()
        print(f"Parsing {simp_filepath}")

        log_filename = os.path.join(scrape_log_dir, filename + '.log')

        # This timeout functionality only works on Unix
        if os.name != 'nt':
            def parse_attempt_fn():
                with timeout_handler(10):
                    parse_tree = parser.parse(content)
                return parse_tree
        else:
            def parse_attempt_fn():
                return parser.parse(content)

        try:
            parse_tree = parse_attempt_fn()

        except TimeoutError:
            with open(log_filename, 'w') as file:
                file.write("timeout")
            print(f"Timeout parsing {simp_filepath}")
            with open(parsed_games_filename, 'a') as file:
                file.write(filename + "\n")
            continue
        except Exception as e:
            with open(log_filename, 'w') as file:
                traceback.print_exc(file=file)

            print(f"Error parsing {simp_filepath}:\n{e}")
            with open(parsed_games_filename, 'a', encoding='utf-8') as file:
                file.write(filename + "\n")
            continue


        min_parse_tree = StripPuzzleScript().transform(parse_tree)
        min_tree_path = os.path.join(trees_dir, filename[:-3] + 'pkl')
        with open(min_tree_path, "wb") as f:
            pickle.dump(min_parse_tree, f)
        pretty_parse_tree_str = min_parse_tree.pretty()
        pretty_tree_filename = os.path.join(pretty_trees_dir, filename)
        print(f"Writing pretty tree to {pretty_tree_filename}")
        with open(pretty_tree_filename, "w", encoding='utf-8') as file:
            file.write(pretty_parse_tree_str)
        # print(min_parse_tree.pretty())
        ps_str = PrintPuzzleScript().transform(min_parse_tree)
        min_filename = os.path.join(min_games_dir, filename)
        print(f"Writing minified game to {min_filename}")
        with open(min_filename, "w", encoding='utf-8') as file:
            file.write(ps_str)

        with open(parsed_games_filename, 'a') as file:
            file.write(filename + "\n")

    # Count the number of games in `min_gmes`
    n_min_games = len(min_games_dir)
    print(f"Number of minified games: {n_min_games}")