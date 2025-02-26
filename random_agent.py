import glob
import os
import pickle
import random

from gen_trees import GenPSTree, PSEnv
from parse_lark import trees_dir, test_games, data_dir
from ps_game import PSGame


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
        state = env.reset(0)

        for i in range(100):
            print(i)
            action = random.randint(0, 5)
            state = env.step(action, state)
