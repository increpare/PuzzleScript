import os
import pickle
from typing import List, Dict, Optional, Any, Set, Tuple, Type, Union
import copy
import random


class PSObject:
    def __init__(self, name: str, colors: List[str], sprite: Optional[List[List[str]]] = None):
        self.name = name
        self.colors = colors  # e.g. ["BLUE", "LIGHTBLUE"]
        self.sprite = sprite  # e.g. 2D array of '0','1','.'
    
    def __repr__(self):
        return f"PSObject(name={self.name}, colors={self.colors}, sprite={self.sprite})"

class LegendEntry:
    """
    Example of a 'legend': 
        "Target = Target1 or Target2 or Target3"
    stored as something like: 
        main_name = "Target"
        components = [("Target1"), ("Target2"), ("Target3")]
    """
    def __init__(self, key: str, obj_names: List[str], operator: Optional[str]):
        self.key = key
        self.obj_names = obj_names
        self.operator = operator

    def __repr__(self):
        return f"{self.key} = {f' {self.operator} '.join(self.obj_names)}"

class RuleBlock:
    def __init__(self, rules: List[Union["Rule", "RuleBlock"]], looping=False):
        self.looping = looping
        self.rules = rules

    def __repr__(self):
        return (f"RuleBlock -- {('once' if not self.looping else 'loop')}\n"\
                + "\n".join([str(r) for r in self.rules]) +
                "\n end block"
                )

class Rule:
    """
    A simplistic representation of a rule like:
      [ > Player | BOUNDARY | Box ] -> [ > Player | > Box ]
    or possibly including flags like 'late', 'checkpoint', etc.
    """
    def __init__(self,
                 left_patterns: List[List[str]],
                 right_patterns: List[List[str]],
                 prefixes: Optional[List[str]] = None):
        # left_patterns, right_patterns: each is a list of "rule parts",
        # each "rule part" is a list of object/directional tokens in that cell.
        self.left_patterns = left_patterns
        self.right_patterns = right_patterns
        self.prefixes = prefixes if prefixes else []

    def __repr__(self):
        return (f"Rule(prefixes={self.prefixes}, "
                f"left={self.left_patterns}, "
                f"right={self.right_patterns})")

class WinCondition:
    """
    For example: 'All Box on Target'
    """
    def __init__(self, quantifier: str, src_obj: str, trg_obj: str):
        self.quantifier = quantifier  # e.g. "All", "Some", "No"
        self.src_obj = src_obj
        self.trg_obj = trg_obj
    
    def __repr__(self):
        return f"{self.quantifier} {self.src_obj}" + (f" on {self.trg_obj}" if self.trg_obj is not None else "")

class PSGame:
    def __init__(self,
                 title: str,
                 flickscreen: Optional[str],
                 verbose_logging: bool,
                 objects: Dict[str, PSObject],
                 legend: List[LegendEntry],
                 collision_layers: List[List[str]],
                 rules: List[Rule],
                 win_conditions: List[WinCondition],
                 levels: List[List[List[str]]]):
        self.title = title
        self.flickscreen = flickscreen
        self.verbose_logging = verbose_logging

        self.objects = objects               # Dict[object_name -> PSObject]
        self.legends = legend               # List[LegendEntry]
        self.collision_layers = collision_layers   # List of Lists of object_names
        self.rules = rules                   # List[Rule]
        self.win_conditions = win_conditions # List[WinCondition]
        self.levels = levels                 # Each level is a 2D array of strings
    
    def copy(self) -> 'PSGame':
        """
        Return a deep copy of this game instance.
        """
        return copy.deepcopy(self)
    
    def __repr__(self):
        return (f"PSGame(title={self.title!r}, objects={self.objects}, "
                f"legends={self.legends}, layers={self.collision_layers}, "
                f"rules={self.rules}, wcs={self.win_conditions}, "
                f"levels=...)")

