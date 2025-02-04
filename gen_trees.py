import os
import pickle
import re
from typing import List, Dict, Optional, Any, Set, Tuple
import copy
import random

from lark import Token, Transformer, Tree
import numpy as np

from ps_game import LegendEntry, PSGame, PSObject, Rule, RuleBlock, WinCondition

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

    def cell_border(self, items):
        return '|'

    def rule_block_once(self, items):
        # One RuleBlock with possible nesting (right?)
        # assert len(items) == 1
        return RuleBlock(looping=False, rules=items)

    def rule_block_loop(self, items):
        return RuleBlock(looping=True, rules=items)

    def rule_part(self, items):
        cells = []
        cell = []
        for item in items:
            if item != '|':
                cell.append(item)
            else:
                cells.append(cell)
                cell = []
        cells.append(cell)
        return cells

    def legend_data(self, items):
        key = items[0]
        key = re.search(r'(.+)=', key).groups()[0]
        # The first entry in this legend key's mapping is just an object name
        assert len(items[1].children) == 1
        obj_names = [str(items[1].children[0])]
        # Every subsequent item is preceded by an AND or OR legend operator.
        # They should all be the same
        operator = None
        for it in items[2:]:
            obj_name = str(it.children[1].children[0])
            obj_names.append(obj_name)
            new_op = str(it.children[0])
            if operator is not None:
                assert operator == new_op
            else:
                operator = new_op

        return LegendEntry(key=key, obj_names=obj_names, operator=operator)

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
        return rule
    
    def return_items_lst(self, items):
        return items

    def objects_section(self, items: List[PSObject]):
        return {ik.name: ik for ik in items}

    def legend_section(self, items: List[LegendEntry]):
        return {it.key: it for it in items}
    
    def layer_data(self, items):
        obj_names = [str(it.children[0]) for it in items]
        return obj_names

    def condition_data(self, items):
        quant = str(items[0])
        src_obj = str(items[1].children[0])
        trg_obj = None
        if len(items) > 2:
            trg_obj = str(items[3].children[0])
        return WinCondition(quantifier=quant, src_obj=src_obj, trg_obj=trg_obj)

    level_data = levels_section = collisionlayers_section = rule_block = rules_section \
        = winconditions_section = return_items_lst

    def ps_game(self, items):
        prelude_items = items[0].children
        title, author, homepage = None, None, None
        flickscreen = False
        verbose_logging = False
        for pi in prelude_items:
            pi_items = pi.children
            keyword = pi_items[0].lower()
            value = None
            print(pi_items)
            if len(pi_items) > 1:
                value = str(pi_items[1])
            if keyword == 'title':
                title = value
            elif keyword == 'author':
                author = value
            elif keyword == 'homepage':
                homepage = value
            elif keyword == 'flickscreen':
                flickscreen = True
            elif keyword == 'verbose_logging':
                verbose_logging = value
        # assert title is not None
        return PSGame(
            title=title,
            objects = items[0],
            flickscreen=flickscreen,
            verbose_logging=verbose_logging,
            legend=items[2],
            collision_layers=items[3],
            rules=items[4],
            win_conditions=items[5],
            levels=items[6],
        )

data_dir = 'data'
from parse_lark import trees_dir
import glob

if __name__ == '__main__':
    tree_paths = glob.glob(os.path.join(trees_dir, '*'))
    trees = []
    tree_paths = sorted(tree_paths, reverse=True)
    test_games = ['Soko-bine']
    test_game_paths = [os.path.join(trees_dir, tg + '.pkl') for tg in test_games]
    tree_paths = test_game_paths + tree_paths
    for tree_path in tree_paths:
        print(tree_path)
        og_game_path = os.path.join(data_dir, 'scraped_games', os.path.basename(tree_path)[:-3] + 'txt')
        print(f"Parsing {og_game_path}")
        with open(tree_path, 'rb') as f:
            tree = pickle.load(f)
        trees.append(tree)

        tree = GenPSTree().transform(tree)