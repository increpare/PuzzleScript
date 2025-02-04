import os
import pickle
from typing import List, Dict, Optional, Any, Set, Tuple
import copy
import random

from lark import Token, Transformer, Tree
import numpy as np

from ps_game import PSGame, PSObject, Rule

class GenPSTree(Transformer):
    """
    Reduces the parse tree to a minimal functional version of the grammar.
    """
    def object_data(self, items):
        name_line = items[0]
        name = str(name_line.children[0].children[0])
        colors = []
        color_line = items[1]
        legend_key = str(name_line.children[1].children[0]) if len(name_line.children) > 1 else None
        for color in color_line.children:
            colors.append(str(color.children[0]))
        if len(items) < 3:
            sprite = None
        else:
            sprite = np.array([c for c in items[2].children])

        return PSObject(
            name=name,
            colors=colors,
            sprite=sprite,
        )

    def rule_content(self, items):
        return ' '.join(items)

    def rule_part(self, items):
        cells = []
        breakpoint()
        # for item in items:
        #     if isinstance(items[1], Tree):
        #         return 
        return items

    def rule_data(self, items):
        ps = []
        l = []
        for i, item in enumerate(items):
            if isinstance(item, Token) and item.type == 'RULE':
                breakpoint()
            if isinstance(item, Token) and item.type == 'THEN':
                r = items[i+1:]
                break
            l.append(item)
        rule = Rule(
            left_patterns = l,
            right_patterns = r,
        )
        print(rule)
        breakpoint()
        return rule

    def return_items_lst(self, items):
        return items

    def objects_section(self, items: List[PSObject]):
        return {ik.name: ik for ik in items}

    level_data = legend_section = levels_section = return_items_lst

    def ps_game(self, items):
        breakpoint()
        return PSGame()

data_dir = 'data'
from parse_lark import trees_dir
import glob

if __name__ == '__main__':
    tree_paths = glob.glob(os.path.join(trees_dir, '*'))
    trees = []
    for tree_path in tree_paths:
        print(tree_path)
        og_game_path = os.path.join(data_dir, 'scraped_games', os.path.basename(tree_path)[:-3] + 'txt')
        print(f"Parsing {og_game_path}")
        with open(tree_path, 'rb') as f:
            tree = pickle.load(f)
        trees.append(tree)

        tree = GenPSTree().transform(tree)