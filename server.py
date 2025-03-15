import base64
import binascii as ba
from dataclasses import dataclass
from enum import IntEnum
import io
import inspect
import itertools
import json
import os
import random
import re
import webbrowser  # Add this import
import atexit  # Add this import
import threading  # Add this import
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service as ChromeService
# from selenium import webdriver

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
import hydra
from hydra.core.config_store import ConfigStore
from PIL import Image
import openai
import requests

import game_gen
from parse_lark import PrintPuzzleScript, RepairPuzzleScript, StripPuzzleScript, add_empty_sounds_section, preprocess_ps
from prompts import *
from utils import extract_ps_code, gen_fewshot_examples, llm_text_query, num_tokens_from_string, save_prompts, truncate_str_to_token_len


@dataclass
class Config:
    port: int = 8000
    sweep: str = 'models'

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


load_dotenv()
openai_client = None
app = Flask(__name__)


@app.route('/')
def serve_doctor():
    return send_from_directory('src', 'doctor.html')

# Route to serve JavaScript files dynamically
@app.route('/<path:filename>')
def serve_js(filename):
    return send_from_directory('src', filename)


@app.route('/save_sweep_stats', methods=['POST'])
def save_sweep_stats():
    data = request.json
    stats = data['results']
    save_dir = os.path.join('logs', data['save_dir'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)
    concise_stats = {}
    for hyp_settings in stats:
        concise_stats[hyp_settings] = []
        for val in stats[hyp_settings]:
            val.pop('code')
            concise_stats[hyp_settings].append(val)
    with open(os.path.join(save_dir, 'concise_stats.json'), 'w') as f:
        json.dump(concise_stats, f, indent=4)

    return jsonify({})


@app.route('/load_ideas', methods=['POST'])
def load_ideas():
    data = request.json
    brainstorm_seed = data['brainstorm_seed']
    ideas_path = os.path.join('logs', f'brainstorm_s-{brainstorm_seed}.json')
    with open(ideas_path, 'r') as f:
        ideas = json.load(f)
    return ideas


@app.route('/load_game_from_file', methods=['POST'])
def load_game_from_file():
    data = request.json
    game = data['game']
    # game_path = os.path.join('misc', game)
    # game_path = os.path.join('src', 'demo', f'{game}.txt')
    game_path = os.path.join('scraped_games', f'{game}.txt')
    with open(game_path, 'r') as f:
        code = f.read()
    return code


@app.route('/log_gen_results', methods=['POST'])
def log_gen_results():
    data = request.json
    console_text = data['console_text']
    solver_text = data['solver_text']
    save_dir = os.path.join('logs', data['save_dir'])
    n_iter = data['n_iter']
    if console_text:
        compile_output_path = os.path.join(save_dir, f'{n_iter}j_compile.txt')
        if not os.path.isfile(compile_output_path):
            with open(compile_output_path, 'w') as f:
                f.write(console_text)
        solver_output_path = os.path.join(save_dir, f'{n_iter}k_solver.txt')
        if solver_text and not os.path.isfile(solver_output_path):
            with open(solver_output_path, 'w') as f:
                f.write(solver_text)

    sols = data['sols']
    n_iter = data['n_iter']
    sols_path = os.path.join(save_dir, f'{n_iter}l_sols.json')
    if sols:
        with open(sols_path, 'w') as f:
            json.dump(sols, f, indent=4)
    gif_urls = data['gif_urls']
    
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

import lark
lark_parser = lark.Lark.open('syntax.lark', start='ps_game')

@app.route('/gen_game', methods=['POST'])
def gen_game():
    data = request.json
    seed = data['seed']
    fewshot = data['fewshot']
    cot = data['cot']
    from_idea = data['from_idea']
    game_idea = data['game_idea']
    cot_prompt_text = cot_prompt if cot else ''
    log_dir = 'logs'
    save_dir = os.path.join(log_dir, data['save_dir'])
    os.makedirs(save_dir, exist_ok=True)
    code = data['code']
    gen_mode = data['gen_mode']
    parents = data['parents']
    parents = [] if parents == 'None' else parents
    compilation_success = data['compilation_success']
    console_text = data['console_text']
    solver_text = data['solver_text']
    lark_error = data['lark_error']
    n_iter = data['n_iter']
    gen_game_output_path = os.path.join(save_dir, f'{n_iter}b_code.txt')
    gen_game_code_output_path = os.path.join(save_dir, f'{n_iter}b_code.txt')
    print(f"Saving code at {gen_game_code_output_path}")
    if not os.path.isfile(gen_game_code_output_path):
        gen_game_prompt_output_path = os.path.join(save_dir, f'{n_iter}a_prompt.txt')
        system_prompt = game_gen_system_prompt
        from_idea_prompt_i = from_idea_prompt.format(game_idea=game_idea) if from_idea else ''
        if n_iter == 0:
            parents_text = '/n/n'.join([p['code'] for p in parents])
            if gen_mode == 'init':
                prompt = gen_game_prompt.format(cot_prompt=cot_prompt_text, from_idea_prompt=from_idea_prompt_i)
            elif gen_mode == 'mutate':
                prompt = game_mutate_prompt.format(parents=parents_text, cot_prompt=cot_prompt_text)
            elif gen_mode == 'crossover':
                prompt = game_crossover_prompt.format(parents=parents_text, cot_prompt=cot_prompt_text)    
        elif not compilation_success:
            if lark_error is None:
                lark_error_prompt = ''
            else:
                lark_error_prompt = f"""{(f"It also resulted in the following error when we attempted to parse the code as a context free grammar using lark:\n```\n{lark_error}\n```\n" if lark_error is not None else "")}"""
            prompt = game_compile_repair_prompt.format(code=code, console_text=console_text, cot_prompt=cot_prompt_text,
                                                       game_idea=game_idea, lark_error_prompt=lark_error_prompt,
                                                       from_idea_repair_prompt=from_idea_prompt_i)
        else:
            prompt = game_solvability_repair_prompt.format(code=code, solver_text=solver_text,
                                                           from_idea_repair_prompt=from_idea_prompt_i, cot_prompt=cot_prompt)
        # if not gen_mode == GenModes.ZERO_SHOT:
        if fewshot:
            system_prompt += gen_fewshot_examples(system_prompt, prompt, max_tokens=exp_config.context_len)
        save_prompts(system_prompt, prompt, gen_game_prompt_output_path)
        text = llm_text_query(system_prompt, prompt, seed, model=exp_config.model)
        with open(gen_game_code_output_path, 'w', encoding='utf-8') as f:
            f.write(text)
    else:
        with open(gen_game_code_output_path, 'r', encoding='utf-8') as f:
            text = f.read()
    code, plaintext = extract_ps_code(text)
    if 'randomDir' in code:
        skip = True
    else:
        skip = False
    # If the code we just generated has already been solved, pass it to the client so it doesn't apply the solver to it
    sols_path = os.path.join(save_dir, f'{n_iter}l_sols.json')
    if os.path.exists(sols_path):
        with open(sols_path, 'r') as f:
            sols = json.load(f)
    else:
        sols = {}

    simp_filepath = os.path.join(save_dir, f'{n_iter}b1_code_simplified.txt')
    # Now save the simplified version of the file
    simp_code = preprocess_ps(code)
    with open(simp_filepath, "w", encoding='utf-8') as file:
        print(f"Writing to {simp_filepath}")
        file.write(simp_code)
    successful_lark_parse = False
    min_code = None
    lark_error = None
    try:
        parse_tree = lark_parser.parse(simp_code)
        successful_lark_parse = True
    except lark.exceptions.ParseError as e:
        print(f"Faile to parse code with lark: {e}")
        lark_error = str(e)
    except lark.exceptions.UnexpectedCharacters as e:
        print(f"Failed to parse code with lark, UnexpectedCharacters: {e}")
        lark_error = str(e)
    if successful_lark_parse:
        lark_error = None
        min_parse_tree = StripPuzzleScript().transform(parse_tree)
        pretty_parse_tree_str = min_parse_tree.pretty()
        pretty_tree_filename = os.path.join(save_dir, f'{n_iter}b2_code_tree.txt')
        print(f"Writing pretty tree to {pretty_tree_filename}")
        with open(pretty_tree_filename, "w") as file:
            file.write(pretty_parse_tree_str)
        repaired_parse_tree = RepairPuzzleScript().transform(min_parse_tree)
        min_code = PrintPuzzleScript().transform(repaired_parse_tree)
        min_code = add_empty_sounds_section(min_code)
        min_filename = os.path.join(save_dir, f'{n_iter}b3_code_min.txt')
        with open(min_filename, "w") as file:
            file.write(min_code)

    return jsonify({
        'code': code,
        'min_code': min_code,
        'text': plaintext,
        'sols': sols,
        'skip': skip,
        'lark_error': lark_error,
    })


@app.route('/gen_game_from_plan', methods=['POST'])
def gen_game_from_plan():
    data = request.json
    seed = data['seed']
    # fewshot = data['fewshot']
    fewshot = True
    # from_idea = data['from_idea']
    game_idea = data['game_idea']
    cot_prompt_text = cot_prompt
    log_dir = 'logs'
    save_dir = os.path.join(log_dir, data['save_dir'])
    os.makedirs(save_dir, exist_ok=True)
    plan_output_path = os.path.join(save_dir, f'0b_plan.txt')
    print(f"Saving plan at: {plan_output_path}")
    if not os.path.isfile(plan_output_path):
        plan_prompt_output_path = os.path.join(save_dir, f'0a_prompt.txt')
        plan_system_prompt = game_gen_system_prompt
        from_idea_prompt_i = from_idea_prompt.format(game_idea=game_idea)
        prompt = plan_game_prompt.format(from_idea_prompt=from_idea_prompt_i)
        if fewshot:
            plan_system_prompt += gen_fewshot_examples(plan_system_prompt, prompt, max_tokens=exp_config.context_len)
        save_prompts(plan_system_prompt, prompt, plan_prompt_output_path)
        game_plan = llm_text_query(plan_system_prompt, prompt, seed)
        with open(plan_output_path, 'w', encoding='utf-8') as f:
            f.write(game_plan)
    else:
        with open(plan_output_path, 'r', encoding='utf-8') as f:
            game_plan = f.read()
    sprites_system_prompt = game_gen_system_prompt
    if fewshot:
        sprites_system_prompt += gen_fewshot_examples(sprites_system_prompt, '', max_tokens=exp_config.context_len) 
    with open('example_sprite_names.txt', 'r', encoding='utf-8') as f:
        sprite_names = f.read()
    with open('example_sprites.json', 'r') as f:
        sprites_library = json.load(f)
    sprites_prompt = gen_sprites_prompt.format(game_plan=game_plan, sprites_library=sprite_names)
    sprites_output_path = os.path.join(save_dir, f'0c_sprites.txt')
    if not os.path.isfile(sprites_output_path):
        sprites_prompt_output_path = os.path.join(save_dir, f'0b_sprites_prompt.txt')
        save_prompts(sprites_system_prompt, sprites_prompt, sprites_prompt_output_path)
        sprites = llm_text_query(sprites_system_prompt, sprites_prompt, seed)
        with open(sprites_output_path, 'w', encoding='utf-8') as f:
            f.write(sprites)
        
    else:
        with open(sprites_output_path, 'r', encoding='utf-8') as f:
            sprites = f.read()

    match = re.search(r'OBJECTS\s*=+\s*(.*?)\s*=+\s*LEGEND\s*=+\s*(.*)```', sprites, re.DOTALL)
    objects = match.group(1)
    objects_list = objects.split('\n\n')
    # Find any objects that are just a name (a single line with a single word)
    for i, obj in enumerate(objects_list):
        if len(obj.split('\n')) == 1:
            obj = obj.strip()
            assert obj in sprites_library, f"Object {obj} not found in sprite library"
            objects_list[i] = random.choice(sprites_library[obj])
    object_legend = match.group(2)

    rules_system_prompt = game_gen_system_prompt
    if fewshot:
        rules_system_prompt += gen_fewshot_examples(rules_system_prompt, '', max_tokens=exp_config.context_len)
    rules_prompt = gen_rules_prompt.format(game_plan=game_plan, object_legend=object_legend)
    rules_output_path = os.path.join(save_dir, f'0e_rules.txt')
    if not os.path.isfile(rules_output_path):
        rules_prompt_output_path = os.path.join(save_dir, f'0d_rules_prompt.txt')
        save_prompts(sprites_system_prompt, rules_prompt, rules_prompt_output_path)
        rules = llm_text_query(sprites_system_prompt, rules_prompt, seed)
        with open(rules_output_path, 'w', encoding='utf-8') as f:
            f.write(rules)
    else:
        with open(rules_output_path, 'r', encoding='utf-8') as f:
            rules = f.read()
        
    match = re.search(r'COLLISIONLAYERS\s*=+\s*(.*?)\s*=+\s*RULES\s*=+\s*((?:(?!\n=+).)*)\s*=+\s*WINCONDITIONS\s*=+\s*((?:(?!===).)*?)\s*=*\s*```', rules, re.DOTALL)
    collision_layers = match.group(1)
    rules = match.group(2)
    win_conditions = match.group(3)
    partial_code = (
        "========\nLEGEND\n========\n"
        + object_legend
        + "\n========\nCOLLISIONLAYERS\n========\n"
        + collision_layers
        + "\n========\nRULES\n========\n"
        + rules
        + "\n========\nWINCONDITIONS\n========\n"
        + win_conditions
    )

    levels_system_prompt = game_gen_system_prompt
    if fewshot:
        levels_system_prompt += gen_fewshot_examples(levels_system_prompt, '', max_tokens=exp_config.context_len)
    levels_prompt = gen_levels_prompt.format(game_plan=game_plan, code=partial_code)
    levels_output_path = os.path.join(save_dir, f'0g_levels.txt')
    if not os.path.isfile(levels_output_path):
        levels_prompt_output_path = os.path.join(save_dir, f'0f_levels_prompt.txt')
        save_prompts(levels_system_prompt, levels_prompt, levels_prompt_output_path)
        levels = llm_text_query(levels_system_prompt, levels_prompt, seed)
        with open(levels_output_path, 'w', encoding='utf-8') as f:
            f.write(levels)
    else:
        with open(levels_output_path, 'r', encoding='utf-8') as f:
            levels = f.read()
    
    levels = re.search(r'```plaintext\s*(?:\s*=+\s*LEVELS\s*=+\s*)\s*(.*)```', levels, re.DOTALL).group(1)
    
    partial_code = (
        "========\nOBJECTS\n========\n"
        + objects
        + "\n"
        + partial_code
        + "\n========\nLEVELS\n========\n"
        + levels
    )

    finalize_system_prompt = game_gen_system_prompt
    if fewshot:
        finalize_system_prompt += gen_fewshot_examples(finalize_system_prompt, '', max_tokens=exp_config.context_len)
    finalize_prompt = gen_finalize_prompt.format(game_plan=game_plan, code=partial_code)
    finalize_output_path = os.path.join(save_dir, f'0i_code.txt')
    if not os.path.isfile(finalize_output_path):
        finalize_prompt_output_path = os.path.join(save_dir, f'0h_code_prompt.txt')
        save_prompts(finalize_system_prompt, finalize_prompt, finalize_prompt_output_path)
        code = llm_text_query(finalize_system_prompt, finalize_prompt, seed)
        with open(finalize_output_path, 'w', encoding='utf-8') as f:
            f.write(code)
    else:
        with open(finalize_output_path, 'r', encoding='utf-8') as f:
            code = f.read()
    
    code = extract_ps_code(code)[0]

    # Add an empty SOUNDS section to the code, after the LEGEND
    code_a, code_b = code.split('\n========\nCOLLISIONLAYERS')
    code = code_a + '\n========\nSOUNDS\n========\n========\nCOLLISIONLAYERS' + code_b
    
    if 'randomDir' in code:
        skip = True
    else:
        skip = False
    # If the code we just generated has already been solved, pass it to the client so it doesn't apply the solver to it
    sols_path = os.path.join(save_dir, f'{0}l_sols.json')
    if os.path.exists(sols_path):
        with open(sols_path, 'r') as f:
            sols = json.load(f)
    else:
        sols = {}
    return jsonify({
        'code': code,
        # 'text': plaintext,
        'sols': sols,
        'skip': skip,
    })


TRANSITIONS_DIR = 'transitions'

@app.route('/list_scraped_games')
def list_scraped_games():
    games = []
    for filename in os.listdir('scraped_games'):
        if filename.endswith('.txt'):
            games.append(filename[:-4])
    return jsonify(games)

@app.route('/save_init_state', methods=['POST'])
def save_init_state():
    data = request.json

    game_hash = data['game_hash']
    game_level = data['game_level']
    state_hash = data['state_hash']
    im_data = data['im_data']
    im_data = base64.b64decode(im_data.split(',')[1])
    im_dir = os.path.join(TRANSITIONS_DIR, game_hash, str(game_level), 'images')
    os.makedirs(im_dir, exist_ok=True)
    im_path = os.path.join(im_dir, f'{state_hash}.png')
    if not os.path.isfile(im_path):
        with open(im_path, 'wb') as f:
            f.write(im_data)
    return jsonify({'status': 'success'})


@app.route('/save_transition', methods=['POST'])
def save_transition():
    data = request.json

    game_hash = data['game_hash']
    game_level = data['game_level']
    trans_dir = os.path.join(TRANSITIONS_DIR, game_hash, str(game_level))
    im_dir = os.path.join(trans_dir, 'images')
    
    # Create directories
    os.makedirs(trans_dir, exist_ok=True)
    os.makedirs(im_dir, exist_ok=True)
    
    # Save images
    # for i, img in enumerate([data['state1_img'], data['state2_img']]):
    img_2 = data['state2_img']
    img_data_2 = base64.b64decode(img_2.split(',')[1])
    img_hash_1 = data[f'state1_hash']
    img_hash_2 = data[f'state2_hash']
    img_path_1 = os.path.join(im_dir, f'{img_hash_1}.png')
    img_path_2 = os.path.join(im_dir, f'{img_hash_2}.png')

    if not os.path.exists(img_path_2):
        with open(img_path_2, 'wb') as f:
            f.write(img_data_2)
    
    # Save transition data
    transition = {
        'state1': data['state1_hash'],
        'state2': data['state2_hash'],
        'action': data['action']
    }
    
    with open(os.path.join(trans_dir, 'transitions.json'), 'a') as f:
        f.write(json.dumps(transition) + '\n')
        
    return jsonify({'status': 'success'})

sweep_i = 21
hypers_i = 0
hypers_ks, hypers_lst = [], []

class ExpConfig:
    model = 'gpt-4o'
    game_i = 0
    cot = True
    fewshot = True
    from_idea = False
    context_len = 30_000

class Sweep:
    game_i = list(range(10))

class ModelSweep(Sweep):
    model = ['gpt-4o', 'o1', 'o3-mini']
    context_len = [10_000]

exp_config = ExpConfig()

all_hypers = {
    'models': ModelSweep()
}

@app.route('/get_sweep_args', methods=['GET'])
def get_sweep_args():
    global hypers_i, exp_config
    save_dir = f'sweep-{sweep_i}'

    done = hypers_i == len(hypers_lst) - 1
    hyper_vals = hypers_lst[hypers_i] if not done else None
    hyper_dict = {k: v for k, v in zip(hypers_ks, hyper_vals)}
    for k in hyper_dict:
        setattr(exp_config, k, hyper_dict[k])

    if exp_config.model in ['o1', 'o3-mini']:
        exp_config.cot = False

    exp_dir = os.path.join(
        save_dir, 
        (f'fromIdea-{int(exp_config.from_idea)}_' if exp_config.from_idea else '') + \
        f'fewshot-{int(exp_config.fewshot)}_cot-{int(exp_config.cot)}' + \
        (f'_{exp_config.model}' if exp_config.model != 'gpt-4o' else '') + \
        (f'_ctx-{exp_config.context_len}' if exp_config.context_len != 30_000 else '')
    )
    game_dir = os.path.join(exp_dir, f'game-{exp_config.game_i}')

    hypers_i += 1
    return jsonify({
        'hypers': hyper_dict,
        'done': done,
        'gameStr': game_dir,
        'fewshot': exp_config.fewshot,
        'cot': exp_config.cot,
        'gameIdx': exp_config.game_i,
        'fromIdea': exp_config.from_idea,
        'fromPlan': False,
    })

def instance_to_dict(instance):
    return {k: v for k, v in inspect.getmembers(instance) if not k.startswith('__') and not inspect.ismethod(k)}

@hydra.main(config_name="config", version_base="1.3")
def main(cfg: Config):
    hypers = all_hypers[cfg.sweep]
    global hypers_ks, hypers_lst
    sweep_dict = instance_to_dict(hypers)
    hypers_ks = list(sweep_dict)
    hypers_lst = list(itertools.product(*sweep_dict.values()))
    app.run(port=cfg.port)


if __name__ == '__main__':
    main()
