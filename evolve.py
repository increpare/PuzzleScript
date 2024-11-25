import base64
import os
import json
import random
import shutil

from flask import jsonify
import io
from javascript import require
from PIL import Image
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore

from game_gen import gen_game, gen_game_from_plan
from prompts import *
ps = require('./script-doctory-py/node_modules/puzzlescript/lib/index.js')


@dataclass
class EvoConfig:
    pop_size: int = 3
    n_gens: int = 20
    exp_seed: int = 1
    overwrite: bool = False

cs = ConfigStore.instance()
cs.store(name="config", node=EvoConfig)

class GameIndividual:
    def __init__(self, code, fitness, compiledIters, solvedIters, skipped):
        self.code = code
        self.fitness = fitness
        self.compiledIters = compiledIters
        self.solvedIters = solvedIters
        self.skipped = skipped

def load_game_from_file(game):
    game_path = os.path.join('src', 'demo', f'{game}.txt')
    with open(game_path, 'r') as f:
        code = f.read()
    return code

def solve_level_bfs(level, engine):
    sol, n_search_iters = engine.bfs(level)
    return sol, n_search_iters



def log_gen_results(save_dir, sols, n_iter, console_text, solver_text, gif_urls):
    save_dir = os.path.join('logs', save_dir)
    if console_text:
        compile_output_path = os.path.join(save_dir, f'{n_iter}j_compile.txt')
        if not os.path.isfile(compile_output_path):
            with open(compile_output_path, 'w') as f:
                f.write(console_text)
        solver_output_path = os.path.join(save_dir, f'{n_iter}k_solver.txt')
        if solver_text and not os.path.isfile(solver_output_path):
            with open(solver_output_path, 'w') as f:
                f.write(solver_text)

    sols_path = os.path.join(save_dir, f'{n_iter}l_sols.json')
    if sols:
        with open(sols_path, 'w') as f:
            json.dump(sols, f, indent=4)
    
    for gif_url, level_i in gif_urls:
        # Download the gif from the url
        gif_data = base64.b64decode(gif_url.split(',')[1])
        gif_path = os.path.join(save_dir, f'{n_iter}m_sol_level-{level_i}.gif')
        
        with open(gif_path, 'wb') as f:
            f.write(gif_data)
        
        # Convert the gif to a sequence of pngs
        with Image.open(io.BytesIO(gif_data)) as img:
            img.seek(0)
            frame_number = 0
            png_dir = os.path.join(save_dir, f'{n_iter}m_sol_level-{level_i}')
            os.makedirs(png_dir, exist_ok=True)
            
            while True:
                frame_path = os.path.join(png_dir, f'frame_{frame_number}.png')
                img.save(frame_path, 'PNG')
                frame_number += 1
                try:
                    img.seek(frame_number)
                except EOFError:
                    break
    
    return jsonify({})



# async def gen_game_wrapper(gen_mode, parents, save_dir, exp_seed, fewshot, cot, from_idea=False, idea='', from_plan=False, max_gen_attempts=10):
def gen_game_wrapper(gen_mode, parents, save_dir, exp_seed, fewshot, cot, from_idea=False, idea='', from_plan=False, max_gen_attempts=10):
    console_text = ''
    n_gen_attempts = 0
    code = ''
    compilation_success = False
    solvable = False
    solver_text = ''
    compiled_iters = []
    solved_iters = []

    while n_gen_attempts < max_gen_attempts and (n_gen_attempts == 0 or not compilation_success or not solvable):
        print(f"Game {save_dir}, attempt {n_gen_attempts}.")
        if from_plan and n_gen_attempts == 0:
            code, sols, skip = gen_game_from_plan(exp_seed, save_dir, idea, n_gen_attempts)
        else:
            code, sols, skip = gen_game(exp_seed, fewshot, cot, save_dir, gen_mode, parents, code, from_idea, idea, console_text, solver_text, compilation_success, n_gen_attempts)

        if skip:
            return GameIndividual(code, -1, [], [], True)

        if not compilation_success:
            compilation_success = True
            solvable = True
            engine = ps.GameEngine(ps.Parser.parse(code).data, ps.EmptyGameEngineHandler())
            for level_i in range(len(engine.data['levels'])):
                sol, n_search_iters = solve_level_bfs(level_i, engine)
                if sol:
                    solver_text += f"Found solution for level {level_i} in {n_search_iters} iterations: {sol}.\n"
                    if len(sol) < 10:
                        solver_text += "Solution is very short. Please make it a bit more complex.\n"
                        solvable = False
                    else:
                        solved_iters.append(n_gen_attempts)
                else:
                    solvable = False
                    solver_text += f"Level {level_i} is not solvable. Please repair it.\n"
        n_gen_attempts += 1

        log_gen_results(save_dir, sols, n_gen_attempts, console_text, solver_text, [])

    return GameIndividual(code, n_search_iters, compiled_iters, solved_iters, False)

# async def evolve():
def evolve(cfg: EvoConfig):
    pop_size = cfg.pop_size
    n_gens = cfg.n_gens
    exp_seed = cfg.exp_seed
    pop = []
    gen = 0

    evo_dir = f"evo-{exp_seed}"

    if cfg.overwrite:
        shutil.rmtree(evo_dir, ignore_errors=True)

    for ind_idx in range(pop_size * 2):
        save_dir = f"{evo_dir}/gen{gen}/game{ind_idx}"
        # game_i = await gen_game_wrapper('init', [], save_dir, exp_seed, fewshot=True, cot=True)
        game_i = gen_game_wrapper('init', [], save_dir, exp_seed, fewshot=True, cot=True)
        pop.append(game_i)

    for gen in range(1, n_gens):
        pop.sort(key=lambda x: x.fitness, reverse=True)
        ancestors = pop[:pop_size]
        new_pop = []

        for ind_idx in range(pop_size):
            do_crossover = random.random() < 0.5
            if do_crossover:
                gen_mode = 'crossover'
                parent1 = random.choice(ancestors)
                remaining_ancestors = [parent for parent in ancestors if parent != parent1]
                parent2 = random.choice(remaining_ancestors)
                parents = [parent1, parent2]
            else:
                gen_mode = 'mutate'
                parents = [random.choice(ancestors)]

            save_dir = f"evo-{exp_seed}/gen{gen}/game{ind_idx}"
            # new_pop.append(await gen_game_wrapper('mutate', parents, save_dir, exp_seed, fewshot=True, cot=True))
            new_pop.append(gen_game_wrapper('mutate', parents, save_dir, exp_seed, fewshot=True, cot=True))

        pop.extend(new_pop)

@hydra.main(config_name="config", version_base="1.3")
def main(cfg: EvoConfig):
    evolve(cfg)

if __name__ == '__main__':
    main()
    # import asyncio
    # asyncio.run(evolve())