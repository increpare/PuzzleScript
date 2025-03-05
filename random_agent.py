import glob
import os
import pickle
import random

import imageio
import jax
import jax.numpy as jnp
import numpy as np

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

        # 0 - left
        # 1 - down
        # 2 - right
        # 3 - up

        key = jax.random.PRNGKey(0)

        actions = jax.random.randint(key, (100,), 0, 5)
        # actions = jnp.array([0, 3, 0])

        state = env.reset(0)

        def step_env(state, action):
            state = env.step(action, state)
            return state, state

        _, state_v = jax.lax.scan(step_env, state, actions)

        # Use jax tree map to add the initial state
        state_v = jax.tree_map(lambda x, y: jnp.concatenate([x[None], y]), state, state_v)

        frames = jax.vmap(env.render, in_axes=(0, None))(state_v, None)
        frames = frames.astype(np.uint8)

        traj_dir = os.path.join('vids', os.path.basename(tree_path)[:-3])
        frames_dir = os.path.join(traj_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            imageio.imsave(os.path.join(frames_dir, f'rand_{i:03d}.png'), frame)

        # Make a gif out of the frames
        gif_path = os.path.join(traj_dir, 'rand.gif')
        imageio.mimsave(gif_path, frames, duration=0.1)
        # exit()