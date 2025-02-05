import os
import pickle
import re
from typing import List, Dict, Optional, Any, Set, Tuple
import copy
import random

import cv2
from einops import rearrange
from lark import Token, Transformer, Tree
import numpy as np
from PIL import Image

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
            objects = items[1],
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


# TODO: double check these colors
color_hex_map = {
    "black": "#000000",
    "white": "#FFFFFF",
    "lightgray": "#D3D3D3",
    "lightgrey": "#D3D3D3",
    "gray": "#808080",
    "grey": "#808080",
    "darkgray": "#A9A9A9",
    "darkgrey": "#A9A9A9",
    "red": "#FF0000",
    "darkred": "#8B0000",
    "lightred": "#FF6666",
    "brown": "#A52A2A",
    "darkbrown": "#5C4033",
    "lightbrown": "#C4A484",
    "orange": "#FFA500",
    "yellow": "#FFFF00",
    "green": "#008000",
    "darkgreen": "#006400",
    "lightgreen": "#90EE90",
    "blue": "#0000FF",
    "lightblue": "#ADD8E6",
    "darkblue": "#00008B",
    "purple": "#800080",
    "pink": "#FFC0CB",
    "transparent": "#00000000"  # Transparent in RGBA format
}

def replace_bg_tiles(x):
    if x == '.':
        return 0
    else:
        return int(x) + 1

def render_sprite(colors, sprite):
    sprite = np.vectorize(replace_bg_tiles)(sprite)
    colors = np.array(['transparent'] + colors)
    colors_vec = np.zeros((len(colors), 4), dtype=np.uint8)
    for i, c in enumerate(colors):
        c = c.lower()
        if c in color_hex_map:
            alpha = 255
            if c == 'transparent':
                alpha = 0
            c = color_hex_map[c]
        c = hex_to_rgba(c, alpha)
        colors_vec[i] = np.array(c)
    im = colors_vec[sprite]
    return im

def level_to_multihot(level):
    pass

def assign_vecs_to_objs(collision_layers):
    n_lyrs = len(collision_layers)
    n_objs = sum([len(lyr) for lyr in collision_layers])
    coll_masks = np.zeros((n_lyrs, n_objs))
    objs_to_idxs = {}
    # vecs = np.eye(n_objs, dtype=np.uint8)
    # obj_vec_dict = {}
    j = 0
    for i, layer in enumerate(collision_layers):
        for obj in layer:
            # obj_vec_dict[obj] = vecs[i]
            objs_to_idxs[obj] = j
            coll_masks[i, j] = 1
            j += 1
    return objs_to_idxs, coll_masks

def hex_to_rgba(hex_code, alpha):
    """Converts a hex color code to RGB values."""

    hex_code = hex_code.lstrip('#')
    rgb = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
    return (*rgb, alpha)

def process_legend(legend):
    char_legend = {}
    meta_tiles = {}
    conjoined_tiles = {}
    for k, v in legend.items():
        v: LegendEntry
        if v.operator is None:
            assert len(v.obj_names) == 1
            char_legend[v.obj_names[0]] = k.strip()
        elif v.operator == 'or':
            meta_tiles[k.strip()] = v.obj_names
        elif v.operator == 'and':
            conjoined_tiles[k.strip()] = v.obj_names
        else: raise Exception('Invalid LegendEntry operator.')
    return char_legend, meta_tiles, conjoined_tiles

def expand_collision_layers(collision_layers, meta_tiles):
    # Preprocess collision layers to replace joint objects with their sub-objects
    for i, l in enumerate(collision_layers):
        j = 0
        for o in l:
            if o in meta_tiles:
                subtiles = meta_tiles[o]
                l = l[:j] + subtiles + l[j+1:]
                collision_layers[i] = l
                # HACK: we could do this more efficiently
                expand_collision_layers(collision_layers)
                j += len(subtiles)
            else:
                j += 1
    return collision_layers

class PSEnv:
    def __init__(self, tree: PSGame):
        self.title = tree.title
        self.levels = tree.levels
        obj_legend, meta_tiles, joint_tiles = process_legend(tree.legend)
        collision_layers = expand_collision_layers(tree.collision_layers, meta_tiles)
        self.obj_to_idxs, coll_masks = assign_vecs_to_objs(collision_layers)
        sprite_stack = []
        for obj_name in self.obj_to_idxs:
            obj = tree.objects[obj_name]
            im = render_sprite(obj.colors, obj.sprite)
            sprite_stack.append(im)
        self.sprite_stack = np.array(sprite_stack)
        n_objs = len(self.obj_to_idxs)
        char_legend = {v: k for k, v in obj_legend.items()}
        self.obj_vecs = np.eye(n_objs, dtype=np.uint8)
        joint_obj_vecs = []
        self.chars_to_idxs = {obj_legend[k]: v for k, v in self.obj_to_idxs.items()}
        for jo, subobjects in joint_tiles.items():
            vec = np.zeros(n_objs, dtype=np.uint8)
            for so in subobjects:
                vec += self.obj_vecs[self.obj_to_idxs[so]]
            assert jo not in self.chars_to_idxs
            self.chars_to_idxs[jo] = len(self.chars_to_idxs)
            self.obj_vecs = np.concatenate((self.obj_vecs, vec[None]), axis=0)

    def char_level_to_multihot(self, level):
        int_level = np.vectorize(lambda x: self.chars_to_idxs[x])(level)
        multihot_level = self.obj_vecs[int_level]
        bg_idx = self.obj_to_idxs['Background']
        multihot_level = rearrange(multihot_level, "h w c -> c h w")
        multihot_level[bg_idx] = 1
        return multihot_level

    def render(self, multihot_level):
        level_height, level_width = multihot_level.shape[1:]
        sprite_height, sprite_width = self.sprite_stack.shape[1:3]
        im = np.zeros((level_height * sprite_height, level_width * sprite_width, 4), dtype=np.uint8)
        im_lyrs = []
        for i, sprite in enumerate(self.sprite_stack):
            sprite_stack_i = np.stack(
                (np.zeros_like(sprite), sprite)
            )
            lyr = multihot_level[i]
            im_lyr = sprite_stack_i[lyr]
            im_lyr = rearrange(im_lyr, "lh lw sh sw c -> (lh sh) (lw sw) c")
            im_lyr_im = Image.fromarray(im_lyr)
            lyr_path = os.path.join(scratch_dir, f'lyr_{i}.png')
            im_lyr_im.save(lyr_path)
            overwrite_mask = im_lyr[:, :, 3] == 255
            im = np.where(np.repeat(overwrite_mask[:, :, None], 4, 2), im_lyr, im)

        return im

def human_loop(env):
    for i, level in enumerate(env.levels):
        level = level[0]
        multihot_level = env.char_level_to_multihot(level)
        im = env.render(multihot_level)
        im_im = Image.fromarray(im)
        # im_path = os.path.join(scratch_dir, f'im_{i}.png')
        # im_im.save(im_path)
        cv2.imshow(env.title, im_im)
        breakpoint()


scratch_dir = 'scratch'
os.makedirs(scratch_dir, exist_ok = True)
if __name__ == '__main__':
    tree_paths = glob.glob(os.path.join(trees_dir, '*'))
    trees = []
    tree_paths = sorted(tree_paths, reverse=True)
    test_games = ['sokoban_basic']
    test_game_paths = [os.path.join(trees_dir, tg + '.pkl') for tg in test_games]
    tree_paths = test_game_paths + tree_paths
    for tree_path in tree_paths:
        print(tree_path)
        og_game_path = os.path.join(data_dir, 'scraped_games', os.path.basename(tree_path)[:-3] + 'txt')
        print(f"Parsing {og_game_path}")
        with open(tree_path, 'rb') as f:
            tree = pickle.load(f)
        trees.append(tree)

        tree: PSGame = GenPSTree().transform(tree)

        env = PSEnv(tree)
        human_loop(env)