from dataclasses import dataclass
from functools import partial
import os
import pickle
import re
import time
from typing import List, Dict, Optional, Any, Set, Tuple
import copy
import random

import cv2
from einops import rearrange
import jax
from lark import Token, Transformer, Tree
import numpy as np
from PIL import Image

from parse_lark import test_games
from ps_game import LegendEntry, PSGame, PSObject, Rule, RuleBlock, WinCondition

class GenPSTree(Transformer):
    """
    Reduces the parse tree to a minimal functional version of the grammar.
    """
    def object_data(self, items):
        name_line = items[0]
        name = str(name_line.children[0].children[0]).lower()
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
        key = items[0].lower()
        # The first entry in this legend key's mapping is just an object name
        assert len(items[1].children) == 1
        obj_names = [str(items[1].children[0]).lower()]
        # Every subsequent item is preceded by an AND or OR legend operator.
        # They should all be the same
        operator = None
        for it in items[2:]:
            obj_name = str(it.children[1].children[0]).lower()
            obj_names.append(obj_name)
            new_op = str(it.children[0])
            if operator is not None:
                assert operator == new_op
            else:
                operator = new_op

        return LegendEntry(key=key, obj_names=obj_names, operator=operator)

    def prefix(self, items):
        out = str(items[0])
        if out.lower().startswith('sfx'):
            return None

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
        l = [it for it in l if it is not None]
        r = [it for it in r if it is not None]
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
        obj_names = [str(it.children[0]).lower() for it in items]
        return obj_names

    def condition_data(self, items):
        quant = str(items[0])
        src_obj = str(items[1].children[0]).lower()
        trg_obj = None
        if len(items) > 2:
            trg_obj = str(items[3].children[0]).lower()
        return WinCondition(quantifier=quant, src_obj=src_obj, trg_obj=trg_obj)

    levels_section = collisionlayers_section = rule_block = rules_section \
        = winconditions_section = return_items_lst

    def level_data(self, items):
        return np.vectorize(lambda x: str(x).lower())(items)

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

def render_solid_color(color):
    color = color.lower()
    if color in color_hex_map:
        c = color_hex_map[color]
    else:
        c = '#000000'
    c = hex_to_rgba(c, 255)
    im = np.zeros((5, 5, 4), dtype=np.uint8)
    im[:, :, :] = np.array(c)
    return im

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
    coll_masks = np.zeros((n_lyrs, n_objs), dtype=np.uint8)
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
        k = k.split(' ')[0]
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

def expand_meta_tiles(tile_list, obj_to_idxs, meta_tiles):
    expanded_meta_tiles = []
    for mt in tile_list:
        if mt in obj_to_idxs:
            expanded_meta_tiles.append(mt)
        elif mt in tile_list:
            expanded_meta_tiles += expand_meta_tiles(meta_tiles[mt], obj_to_idxs, meta_tiles)
    return expanded_meta_tiles

def gen_check_win(win_conditions, obj_to_idxs):
    def check_all(lvl, src, trg):
        return np.sum((lvl[src] == 1) * (lvl[trg] == 0)) == 0

    def check_some(lvl, src, trg):
        return np.sum((lvl[src] == 1) * (lvl[trg] == 1)) > 0

    def check_none(lvl, src):
        return np.sum(lvl[src] == 1) == 0

    funcs = []
    for win_condition in win_conditions:
        src, trg = win_condition.src_obj, win_condition.trg_obj
        src = obj_to_idxs[src]
        if trg is not None:
            trg = obj_to_idxs[trg]
        if win_condition.quantifier == 'all':
            func = partial(check_all, src=src, trg=trg)
        elif win_condition.quantifier == 'some':
            func = partial(check_some, src=src, trg=trg)
        elif win_condition.quantifier == 'no':
            func = partial(check_none, src=src)
        else:
            raise Exception('Invalid quantifier.')
        funcs.append(func)

    def check_win(lvl):
        return all([func(lvl) for func in funcs])
    return check_win


def gen_subrule(rule, n_objs, obj_to_idxs, meta_tiles):
    assert len(rule.prefixes) == 0
    lp = rule.left_patterns
    rp = rule.right_patterns
    n_kernels = len(lp)
    assert n_kernels == 1
    assert len(rp) == n_kernels
    lp = lp[0]
    rp = rp[0]
    n_cells = len(lp)
    assert len(rp) == n_cells

    lr_in = np.zeros((n_objs + (n_objs * 4), 1, n_cells), dtype=np.int8)
    rr_in = np.zeros_like(lr_in)
    ur_in = np.zeros((n_objs + (n_objs * 4), n_cells, 1), dtype=np.int8)
    dr_in = np.zeros_like(ur_in)
    lr_out = np.zeros_like(lr_in)
    rr_out = np.zeros_like(rr_in)
    ur_out = np.zeros_like(ur_in)
    dr_out = np.zeros_like(dr_in)
    for i, (l_cell, r_cell) in enumerate(zip(lp, rp)):
        # Left cell
        if len(l_cell) == 1:
            l_cell = l_cell[0]
            l_cell = l_cell.split(' ')
        l_force = False
        l_no = False
        for obj in l_cell:
            obj = obj.lower()
            if obj in meta_tiles:
                return None
            if obj in obj_to_idxs:
                fill_val = 1
                if l_no:
                    fill_val = -1
                obj_i = obj_to_idxs[obj]
                lr_in[obj_i, 0, i] = fill_val
                rr_in[obj_i, 0, n_cells-1-i] = fill_val
                ur_in[obj_i, i, 0] = fill_val
                dr_in[obj_i, n_cells-1-i, 0] = fill_val
                if l_force:
                    lr_in[n_objs + (obj_i * 4) + 2, 0, i] = 1
                    rr_in[n_objs + (obj_i * 4) + 0, 0, n_cells-1-i] = 1
                    ur_in[n_objs + (obj_i * 4) + 1, i, 0] = 1
                    dr_in[n_objs + (obj_i * 4) + 3, n_cells-1-i, 0] = 1
                l_force, l_no = False, False
            elif obj == '>':
                l_force = True
            elif obj == 'no':
                l_no = True
            else: raise Exception(f'Invalid object `{obj}` in rule.')
        # Right cell
        if len(r_cell) == 1:
            r_cell = r_cell[0]
            r_cell = r_cell.split(' ')
        r_force = False
        r_no = False
        for obj in r_cell:
            obj = obj.lower()
            # NOTE: Should never have a meta-tile here if we didn't already see one on the left side (?)
            if obj in obj_to_idxs:
                fill_val = 1
                if r_no:
                    fill_val = -1
                obj_i = obj_to_idxs[obj]
                lr_out[obj_i, 0, i] = fill_val
                rr_out[obj_i, 0, n_cells-1-i] = fill_val
                ur_out[obj_i, i, 0] = fill_val
                dr_out[obj_i, n_cells-1-i, 0] = fill_val
                if r_force:
                    lr_out[n_objs + (obj_i * 4) + 2, 0, i] = 1
                    rr_out[n_objs + (obj_i * 4) + 0, 0, n_cells-1-i] = 1
                    ur_out[n_objs + (obj_i * 4) + 1, i, 0] = 1
                    dr_out[n_objs + (obj_i * 4) + 3, n_cells-1-i, 0] = 1
                r_force, r_no = False, False
            elif obj == '>':
                r_force = True
            elif obj == 'no':
                r_no = True
            else: raise Exception('Invalid object in rule.')
    lr_out = lr_out - np.clip(lr_in, 0, 1)
    rr_out = rr_out - np.clip(rr_in, 0, 1)
    ur_out = ur_out - np.clip(ur_in, 0, 1)
    dr_out = dr_out - np.clip(dr_in, 0, 1)
    lr_rule = np.stack((lr_in, lr_out), axis=0)
    rr_rule = np.stack((rr_in, rr_out), axis=0)
    ur_rule = np.stack((ur_in, ur_out), axis=0)
    dr_rule = np.stack((dr_in, dr_out), axis=0)
    # TODO: If horizontal/vertical etc. has been specified, filter out unnecessary rules here
    rules = [lr_rule, rr_rule, ur_rule, dr_rule]
    rule_names = [f"{rule}_{dir}" for dir in enumerate(['lr', 'rr', 'ur', 'dr'])]
    rule_fns = [partial(apply_rule, move_rule=rule, rule_name=rule_name) 
                    for rule, rule_name in zip(rules, rule_names)]
    return rule_fns


@dataclass
class ObjFnReturn:
    # detected object/force indices
    force_idx: int = None
    obj_idx: int = None


@dataclass
class CellFnReturn:
    # A list of indices of objects that were detected
    # Return these so that we can remove them in output cells (before projecting output pattern)
    # (this doesn't include detecting `no` i.e. the absence of an object)
    detected: np.ndarray
    # Return the force index so that, in default, non-direction-specific rules, we can project the correct force to output cells
    force_idx: np.ndarray
    # A dictionary of the objects detected, mapping meta-object names to sub-object indices
    meta_objs: dict


def gen_subrules_meta(rule, n_objs, obj_to_idxs, meta_tiles):
    assert len(rule.prefixes) == 0

    # def detect_all_objs_in_cell(objs_vec, m_cell):
    #     # for detecting overlapping objects (not necessary?)
    #     return np.all(objs_vec[:, None, None] == m_cell), []

    ### Functions for detecting regular atomic objects
    def detect_obj_in_cell(obj_idx, m_cell):
        return m_cell[obj_idx] == 1, ObjFnReturn(obj_idx=obj_idx)

    def detect_no_obj_in_cell(obj_idx, m_cell):
        return m_cell[obj_idx] == 0, ObjFnReturn()

    def detect_force_on_obj(obj_idx, force_idx, m_cell):
        obj_is_present = m_cell[obj_idx] == 1
        force_is_present = m_cell[n_objs + (obj_idx * 4) + force_idx] == 1
        # force_idx = np.argwhere(m_cell[n_objs + (obj_idx * 4):n_objs + (obj_idx * 4) + 4] == 1)
        # assert len(force_idx) <= 1
        is_force_on_obj = obj_is_present and force_is_present
        if is_force_on_obj:
            return is_force_on_obj, ObjFnReturn(
                # force_idx=force_idx[0], 
                obj_idx=obj_idx)
        else:
            return is_force_on_obj, ObjFnReturn()

    ### Functions for detecting meta-objects
    def detect_any_objs_in_cell(objs_vec, m_cell):
        """Given a multi-hot vector indicating a set of objects, return the index of the object contained in this cell."""
        detected_vec_idxs = np.argwhere(np.sum(objs_vec[:, None, None] * m_cell, axis=0) > 0)[0]
        if len(detected_vec_idxs) == 0:
            return False, {}
        else:
            obj_idx = detected_vec_idxs[0]

            return True, {'obj_idx': obj_idx}

    def detect_no_objs_in_cell(objs_vec, m_cell):
        detected = np.zeros(m_cell.shape, dtype=np.int8)
        return np.sum(objs_vec[:, None, None] * m_cell) == 0, ObjFnReturn

    def detect_force_on_meta(obj_idxs, force_idx, m_cell):
        force_obj_vecs = []
        for obj_idx in obj_idxs:
            force_obj_vec = np.zeros(n_objs + n_objs * 4, dtype=np.int8)
            force_obj_vec[obj_idx] = 1
            force_obj_vec[n_objs + obj_idx * 4 + force_idx] = 1
            force_obj_vecs.append(force_obj_vec)
        obj_activations = np.sum(np.array(force_obj_vecs)[:, None, None] * m_cell, axis=0) 
        if np.all(obj_activations < 2):
            return False, ObjFnReturn()
        obj_idx = np.argwhere(obj_activations == 2)[0][0]
        force_idx = np.argwhere(m_cell[n_objs + (obj_idx * 4):n_objs + (obj_idx * 4) + 4] == 1)[0]
        detected = np.zeros_like(m_cell)
        detected[obj_idx] = 1
        detected[force_idx] = 1
        return True, ObjFnReturn(
            # force_idx=force_idx,
            obj_idx=obj_idx)

    ### Function for projecting onto cells
    def project_cell(detect_out: CellFnReturn, m_cell):
        detected = detect_out.detected
        force_idx = detect_out.force_idx
        meta_objs = detect_out.meta_objs
    
        assert np.all(detected <= 1) and np.all(detected >= 0)
        m_cell -= detected

    def gen_cell_detection_fn(l_cell, force_idx):
        fns = []
            # l_cell = l_cell[0]
        l_cell = l_cell.split(' ')
        no, force = False, False
        obj_names = []
        for obj in l_cell:
            obj = obj.lower()
            if obj == 'no':
                no = True
            elif obj == '>':
                force = True
            else:
                obj_names.append(obj)
                if obj in obj_to_idxs:
                    obj_idx = obj_to_idxs[obj]
                    if no:
                        fns.append(partial(detect_no_obj_in_cell, obj_idx))
                    elif force:
                        fns.append(partial(detect_force_on_obj, obj_idx, force_idx))
                    else:
                        fns.append(partial(detect_obj_in_cell, obj_idx))
                elif obj in meta_tiles:
                    sub_objs = expand_meta_tiles([obj], obj_to_idxs, meta_tiles)
                    sub_obj_idxs = [obj_to_idxs[so] for so in sub_objs]
                    sub_objs_vec = np.zeros((n_objs + n_objs * 4), dtype=np.int8)
                    sub_objs_vec[sub_obj_idxs] = 1
                    if no:
                        fns.append(partial(detect_no_objs_in_cell, sub_objs_vec))
                    elif force:
                        fns.append(partial(detect_force_on_meta, sub_obj_idxs, force_idx))
                    else:
                        fns.append(partial(detect_any_objs_in_cell, sub_objs_vec))
                else:
                    raise Exception(f'Invalid object `{obj}` in rule.')
        
        def cell_detection_fn(m_cell):
            # TODO: can vmap this
            fn_outs = [fn(m_cell) for fn in fns]
            activated = all([f[0] for f in fn_outs])
            fn_outs: List[ObjFnReturn] = [f[1] for f in fn_outs]
            detected = np.zeros(m_cell.shape, dtype=np.int8)
            for i, f in enumerate(fn_outs):
                if f.obj_idx is not None:
                    detected[f.obj_idx] = 1
                if f.force_idx is not None:
                    detected[f.force_idx] = 1
            force_idxs = [np.array([f.force_idx for f in fn_outs if f.force_idx is not None])]
            force_idx = force_idxs[0] if len(force_idxs) > 0 else None
            meta_objs = {k: np.array([f.obj_idx for f in fn_outs if f.obj_idx is not None]) for k in obj_names}
            ret = CellFnReturn(
                detected=detected,
                force_idx=force_idx,
                meta_objs=meta_objs,
            )
            return activated, ret

        return cell_detection_fn

    def disambiguate_meta(obj, meta_objs):
        return meta_objs[obj].item()
        # if obj in meta_objs:
        #     return obj_to_idxs[meta_objs[obj]]

    def project_obj(m_cell, obj, meta_objs):
        obj_idx = disambiguate_meta(obj, meta_objs)
        m_cell[obj_idx] = 1
        return m_cell

    def project_no_obj(m_cell, obj, meta_objs):
        obj_idx = disambiguate_meta(obj, meta_objs)
        m_cell[obj_idx] = 0
        return m_cell

    def project_force_on_obj(m_cell, detect_out, obj, force_idx):
        meta_objs = detect_out.meta_objs
        obj_idx = disambiguate_meta(obj, meta_objs)
        m_cell = m_cell.at[obj_idx].set(1)
        m_cell = m_cell.at[n_objs + (obj_idx * 4) + force_idx].set(1)
        return m_cell

    def gen_cell_projection_fn(r_cell, force_idx):
        fns = []
        if len(r_cell) == 1:
            r_cell = r_cell[0]
            r_cell = r_cell.split(' ')
        no, force = False, False
        for obj in r_cell:
            obj = obj.lower()
            if obj == 'no':
                no = True
            elif obj == '>':
                force = True
            elif obj in obj_to_idxs:
                if no:
                    fns.append(partial(project_no_obj, obj))
                elif force:
                    fns.append(partial(project_force_on_obj, obj=obj, force_idx=force_idx))
                else:
                    fns.append(partial(project_obj, obj))
        
        def cell_projection_fn(m_cell, detect_out):
            m_cell -= detect_out.detected
            # vmap
            for proj_fn in fns:
                m_cell = proj_fn(m_cell, detect_out)
            return m_cell

        return cell_projection_fn


    def gen_rule_fn(lp, rp, rot):
        lp = np.array(lp)
        force_idx = [1, 0, 2, 3][rot]
        is_horizontal = lp.shape[0] == 1
        is_vertical = lp.shape[1] == 1
        assert is_horizontal ^ is_vertical
        in_patch_shape = lp.shape
        # TODO: kernels. We assume just 1 here.
        if is_horizontal:
            lp = lp[0, :]
        elif is_vertical:
            lp = lp[:, 0]
        cell_detection_fns = []
        cell_projection_fns = []
        for i, l_cell in enumerate(lp):
            cell_detection_fns.append(gen_cell_detection_fn(l_cell, force_idx))
        for i, r_cell in enumerate(rp):
            cell_projection_fns.append(gen_cell_projection_fn(r_cell, force_idx))

        def rule_fn(lvl):
            n_chan = lvl.shape[1]
            patches = jax.lax.conv_general_dilated_patches(
                lvl, in_patch_shape, window_strides=(1, 1), padding='VALID',
            )
            assert patches.shape[0] == 1
            patches = patches[0]
            patches = rearrange(patches, "c h w -> h w c")
            # TODO vmap across the patches
            patch_activations, detect_outs  = [], []
            for xi, in_patch_row in enumerate(patches):
                patch_activations_row = []
                cell_outs_row = []
                for yi, in_patch in enumerate(in_patch_row):
                    pattern_activated = True
                    cell_outs_patch = []
                    for i, cell_fn in enumerate(cell_detection_fns):
                        in_patch = in_patch.reshape((n_chan, *in_patch_shape))
                        if is_vertical:
                            m_cell = in_patch[:, i, 0]
                        if is_horizontal:
                            m_cell = in_patch[:, 0, i]
                        cell_activated, outs = cell_fn(m_cell)
                        if not cell_activated:
                            pattern_activated = False
                            # break
                        cell_outs_patch.append(outs)
                    patch_activations_row.append(pattern_activated)
                    cell_outs_row.append(cell_outs_patch)
                patch_activations.append(patch_activations_row)
                detect_outs.append(cell_outs_row)

            # eliminate all but one activation

            # return patch_activations, cell_outs
            patch_activations = np.array(patch_activations)
            if patch_activations.sum() > 0:
                print(lp, rp, force_idx)

                # Mask out everything but the position of the "first" activation
                first_a = np.argwhere(patch_activations == 1)[0]
                patch_activations = np.zeros_like(patch_activations)
                patch_activations[*first_a] = 1
                detect_outs = detect_outs[first_a[0]][first_a[1]]

                out_cell_idxs = np.indices(in_patch_shape)
                for idxs in out_cell_idxs:
                    idxs = idxs[0] + first_a
                    map_cells = patches[*idxs]
                    map_cells = map_cells.reshape((n_chan, -1))
                    out_cells = []
                    for i, (detect_out, cell_projection_fn) in enumerate(zip(detect_outs, cell_projection_fns)):
                        out_cell = map_cells[:, i]
                        out_cell = cell_projection_fn(out_cell, detect_out)
                        out_cells.append(out_cell)
                    out_cells = np.array(out_cells).reshape(-1)
                    patches = patches.at[*idxs].set(out_cells)

                # Now, to reconstruct the level, we create an identity kernel (using `in_patch_shape`) and perform a transposed convolution
                # TODO


            return lvl

        return rule_fn
    
    rule_fns = []
    for lp, rp in zip(rule.left_patterns, rule.right_patterns):
        # for rot in [0,1,0,0]:
        for rot in [0]:
            # rotate the patterns
            lp = np.rot90(lp, rot)
            rp = np.rot90(rp, rot)
            rule_fns.append(gen_rule_fn(lp, rp, rot))

    return rule_fns
        
        
def gen_rules(obj_to_idxs, coll_mat, tree_rules, meta_tiles):
    n_objs = len(obj_to_idxs)
    rule_fns = []
    meta_tile_rules = []
    if len(tree_rules) == 0:
        pass
    elif len(tree_rules) == 1:
        rule_blocks = tree_rules[0]
        for rule_block in rule_blocks:
            assert rule_block.looping == False
            for rule in rule_block.rules:
                # TODO: rule-block and loop logics
                sub_rule_fns = gen_subrules_meta(rule, n_objs, obj_to_idxs, meta_tiles)
                rule_fns += sub_rule_fns

    def rule_fn(lvl):
        lvl = lvl[None].astype(np.int8)
        for rule_fn in rule_fns:
            lvl = rule_fn(lvl)
        return lvl[0]

    return rule_fn

def gen_move_rules(obj_to_idxs, coll_mat):
    n_objs = len(obj_to_idxs)
    rule_fns = []
    for obj, idx in obj_to_idxs.items():
        if obj == 'Background':
            continue
        coll_vector = coll_mat[idx]

        left_rule_in = np.zeros((n_objs + (n_objs * 4), 1, 2), dtype=np.int8)
        # object must be present in right cell
        left_rule_in[idx, 0, 1] = 1
        # leftward force
        left_rule_in[n_objs + (idx * 4) + 0, 0, 1] = 1
        # absence of collidable tiles in the left cell
        left_rule_in[:n_objs, 0, 0] = -coll_vector
        left_rule_out = np.zeros_like(left_rule_in)
        # object moves to left cell
        left_rule_out[idx, 0, 0] = 1
        # remove anything that was present in the input (i.e. the object and force)
        left_rule_out -= np.clip(left_rule_in, 0, 1)
        left_rule = np.stack((left_rule_in, left_rule_out), axis=0)

        # FIXME: Actually this is the down rule and vice versa, and we've relabelled actions accordingly. Hack. Can't figure out what's wrong here.
        # Something to do with flipping output kernel?
        up_rule_in = np.zeros((n_objs + (n_objs * 4), 2, 1), dtype=np.int8)
        # object must be present in lower cell
        up_rule_in[idx, 0] = 1
        # upward force
        up_rule_in[n_objs + (idx * 4) + 1, 0] = 1
        # absence of collidable tiles in the upper cell
        up_rule_in[:n_objs, 1, 0] = -coll_vector
        up_rule_out = np.zeros_like(up_rule_in)
        # object moves to upper cell
        up_rule_out[idx, 1] = 1
        # remove anything that was present in the input
        up_rule_out -= np.clip(up_rule_in, 0, 1)
        up_rule = np.stack((up_rule_in, up_rule_out), axis=0)

        right_rule_in = np.zeros((n_objs + (n_objs * 4), 1, 2), dtype=np.int8)
        # object must be present in left cell
        right_rule_in[idx, 0, 0] = 1
        # rightward force
        right_rule_in[n_objs + (idx * 4) + 2, 0, 0] = 1
        # absence of collidable tiles in the right cell
        right_rule_in[:n_objs, 0, 1] = -coll_vector
        right_rule_out = np.zeros_like(right_rule_in)
        # object moves to right cell
        right_rule_out[idx, 0, 1] = 1
        # remove anything that was present in the input
        right_rule_out -= np.clip(right_rule_in, 0, 1)
        right_rule = np.stack((right_rule_in, right_rule_out), axis=0)

        down_rule_in = np.zeros((n_objs + (n_objs * 4), 2, 1), dtype=np.int8)
        # object must be present in upper cell
        down_rule_in[idx, 1] = 1
        # downward force
        down_rule_in[n_objs + (idx * 4) + 3, 1] = 1
        # absence of collidable tiles in the lower cell
        down_rule_in[:n_objs, 0, 0] = -coll_vector
        down_rule_out = np.zeros_like(down_rule_in)
        # object moves to lower cell
        down_rule_out[idx, 0] = 1
        # remove anything that was present in the input
        down_rule_out -= np.clip(down_rule_in, 0, 1)
        down_rule = np.stack((down_rule_in, down_rule_out), axis=0)

        # rules += [left_rule, right_rule, up_rule, down_rule]
        rules = [left_rule, right_rule, up_rule, down_rule]
        rule_names = [f"{obj}_move_{j}" for j in ['left', 'right', 'up', 'down']]
        rule_fns += [partial(apply_rule, move_rule=rule, rule_name=rule_name)
                     for rule, rule_name in zip(rules, rule_names)]
    return rule_fns

def apply_rule(lvl, move_rule, rule_name):
    lvl = lvl[None].astype(np.int8)
    inp = move_rule[0]
    ink = inp[None]
    activations = jax.lax.conv(lvl, ink, window_strides=(1,1), padding='VALID')
    thresh_act = np.sum(np.clip(inp, 0, 1))
    bin_activations = (activations == thresh_act).astype(np.int8)

    non_zero_activations = np.argwhere(bin_activations != 0)
    if non_zero_activations.size > 0:
        print(f"Non-zero activations detected: {non_zero_activations}. Rule_i: {rule_name}")
        # if rule_i == 1:
        #     breakpoint()

    outp = move_rule[1]
    outk = outp[:, None]
    # flip along width and height
    outk = np.flip(outk, axis=(2, 3))
    out = jax.lax.conv_transpose(bin_activations, outk, (1, 1), padding='VALID',
                                        dimension_numbers=('NCHW', 'OIHW', 'NCHW'))
    # if out.sum() != 0:
    #     print('lvl')
    #     print(lvl[0,2])
    #     print('out')
    #     print(out[0,2])
    #     breakpoint()
    lvl += out

    return lvl[0]

    

@dataclass
class PSState:
    multihot_level: np.ndarray
    win: bool

class PSEnv:
    def __init__(self, tree: PSGame):
        self.title = tree.title
        self.levels = tree.levels
        obj_legend, meta_tiles, joint_tiles = process_legend(tree.legend)
        collision_layers = expand_collision_layers(tree.collision_layers, meta_tiles)
        self.obj_to_idxs, coll_masks = assign_vecs_to_objs(collision_layers)
        self.n_objs = len(self.obj_to_idxs)
        coll_mat = np.einsum('ij,ik->jk', coll_masks, coll_masks, dtype=np.uint8)
        rule_fns = gen_rules(self.obj_to_idxs, coll_mat, tree.rules, meta_tiles)
        self.rule_fns = [rule_fns] + gen_move_rules(self.obj_to_idxs, coll_mat)
        self.check_win = gen_check_win(tree.win_conditions, self.obj_to_idxs)
        self.player_idx = self.obj_to_idxs['player']
        sprite_stack = []
        for obj_name in self.obj_to_idxs:
            obj = tree.objects[obj_name]
            if obj.sprite is not None:
                im = render_sprite(obj.colors, obj.sprite)
            else:
                assert len(obj.colors) == 1
                im = render_solid_color(obj.colors[0])
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
        bg_idx = self.obj_to_idxs['background']
        multihot_level = rearrange(multihot_level, "h w c -> c h w")
        multihot_level[bg_idx] = 1
        return multihot_level

    def render(self, state: PSState, cv2=True):
        lvl = state.multihot_level
        level_height, level_width = lvl.shape[1:]
        sprite_height, sprite_width = self.sprite_stack.shape[1:3]
        im = np.zeros((level_height * sprite_height, level_width * sprite_width, 4), dtype=np.uint8)
        im_lyrs = []
        for i, sprite in enumerate(self.sprite_stack):
            sprite_stack_i = np.stack(
                (np.zeros_like(sprite), sprite)
            )
            lyr = lvl[i]
            im_lyr = sprite_stack_i[lyr]
            im_lyr = rearrange(im_lyr, "lh lw sh sw c -> (lh sh) (lw sw) c")
            im_lyr_im = Image.fromarray(im_lyr)
            lyr_path = os.path.join(scratch_dir, f'lyr_{i}.png')
            im_lyr_im.save(lyr_path)
            overwrite_mask = im_lyr[:, :, 3] == 255
            im = np.where(np.repeat(overwrite_mask[:, :, None], 4, 2), im_lyr, im)

        if cv2:
            # swap the red and blue channels
            im = im[:, :, [2, 1, 0, 3]]

        return im

    def reset(self, lvl_i):
        lvl = self.get_level(lvl_i)
        return PSState(
            multihot_level=lvl,
            win = False,
        )

    def apply_player_force(self, action, state: PSState):
        multihot_level = state.multihot_level
        player_pos = np.argwhere(multihot_level[self.player_idx] == 1)[0]
        force_map = np.zeros((4 * multihot_level.shape[0], *multihot_level.shape[1:]), dtype=np.uint8)
        if action >= 4:
            # Apply action
            pass
        else:
            action = [0, 1, 2, 3][action]
            force_map[self.player_idx * 4 + action, player_pos[0], player_pos[1]] = 1
        lvl = np.concatenate((multihot_level, force_map), axis=0)
        return lvl

    def step(self, action, state: PSState):
        lvl = self.apply_player_force(action, state)
        lvl_changed = True
        n_apps = 0
        while lvl_changed and n_apps < 100:
            lvl_changed = False
            lvl, lvl_changed = substep(lvl, self.rule_fns)
            n_apps += 1

        multihot_level = lvl[:self.n_objs]

        win = self.check_win(multihot_level)
        if win:
            print("You win!")
        else:
            print("You not win yet!")
        return PSState(
            multihot_level=multihot_level,
            win=win,
        )

    def get_level(self, level_idx):
        level = self.levels[level_idx][0]
        # Convert the level to a multihot representation and render it
        multihot_level = env.char_level_to_multihot(level)
        return multihot_level


def substep(lvl, rule_fns):
    lvl_changed = False
    for i, rule_fn in enumerate(rule_fns):
        # Detect input activations
        new_lvl = rule_fn(lvl)
        new_lvl = np.clip(new_lvl, 0, 1)
        if not np.array_equal(new_lvl, lvl):
            lvl = new_lvl
            lvl_changed = True
            print("Rule applied")
    return lvl, lvl_changed


def human_loop(env: PSEnv):
    lvl_i = 0 
    state = env.reset(lvl_i)
    im = env.render(state)
    
    # Resize the image by a factor of 5
    new_h, new_w = tuple(np.array(im.shape[:2]) * 10)
    im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    state_hist = []
    
    # Display the image in an OpenCV window
    cv2.imshow(env.title, im)
    print("Press an arrow key or 'x' (ESC to exit).")
    
    # Loop waiting for key presses
    while True:
        # cv2.waitKey(0) waits indefinitely until a key is pressed.
        # Mask with 0xFF to get the lowest 8 bits (common practice).
        key = cv2.waitKey(0)
        action = None
        do_reset = False

        # If the user presses ESC (ASCII 27), exit the loop.
        if key == 27:
            break
        elif key == ord('x'):
            print("x")
            action = 4
        elif key == 97:
            print("left arrow")
            action = 0
        elif key == 119:
            print("up arrow")
            action = 3
        elif key == 100:
            print("right arrow")
            action = 2
        elif key == 115:
            print("down arrow")
            action = 1
        elif key == ord('r'):
            print("Restarting level...")
            do_reset = True
        elif key == ord('n'):
            print("Advancing level...")
            lvl_i += 1
            do_reset = True
        elif key == ord('z'):
            print("Undoing last action...")
            if len(state_hist) > 1:
                state_hist.pop()
                state = state_hist[-1]
            im = env.render(state)
            im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(env.title, im)

        else:
            print("Other key pressed:", key)

        if lvl_i >= len(env.levels):
            print("No more levels!")
            break

        elif do_reset:
            state = env.reset(lvl_i)
            state_hist.append(state)
            im = env.render(state)
            im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(env.title, im)
    
        elif action is not None:
            lvl = env.apply_player_force(action, state)
            vis_lvl = lvl[:env.n_objs]
            lvl_changed = True
            n_vis_apps = 0
            while lvl_changed:
                lvl_changed = False
                vis_lvl_changed = False
                lvl, lvl_changed = substep(lvl, env.rule_fns)
                new_vis_lvl = lvl[:env.n_objs]
                if not np.array_equal(new_vis_lvl, vis_lvl):
                    vis_lvl = new_vis_lvl
                    vis_lvl_changed = True
                    if n_vis_apps > 1 and vis_lvl_changed:
                        print('sleep')
                        cv2.waitKey(1)  # waits for 1000 ms (1 second)
                    n_vis_apps += 1
                    print('Rule visibly applied')
                state.multihot_level = vis_lvl
                im = env.render(state)
                im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                cv2.imshow(env.title, im)
                # Add a short waitKey here to allow the window to update.
                cv2.waitKey(1)  # 1 ms delay; adjust as necessary
            win = env.check_win(vis_lvl)
            if win:
                print("You win!")
            else:
                print("You not win yet!")
            state = PSState(
                multihot_level=vis_lvl,
                win=win,
            )
            state_hist.append(state)

        if state.win:
            lvl_i += 1
            state = env.reset(lvl_i)
            im = env.render(state)
            im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(env.title, im)
    # Close the image window
    cv2.destroyAllWindows()

scratch_dir = 'scratch'
os.makedirs(scratch_dir, exist_ok = True)
if __name__ == '__main__':
    tree_paths = glob.glob(os.path.join(trees_dir, '*'))
    trees = []
    tree_paths = sorted(tree_paths, reverse=True)
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