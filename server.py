import base64
import binascii as ba
from enum import IntEnum
import io
import json
import os
import random
import re

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
import openai
import requests

from utils import num_tokens_from_string, truncate_str_to_token_len


load_dotenv()
openai_client = openai.Client(api_key=os.getenv('OPENAI_API_KEY'))
app = Flask(__name__)


@app.route('/')
def serve_doctor():
    return send_from_directory('src', 'doctor.html')

# Route to serve JavaScript files dynamically
@app.route('/<path:filename>')
def serve_js(filename):
    return send_from_directory('src', filename)

comments_note = "Any comments must be in the form `(comment)`."
formatting_prompt = \
    """Return your code in full, inside a ```plaintext code block."""
game_gen_system_prompt = (
    "You are a game designer, familiar with the PuzzleScript game description language. "
)
fewshow_examples_prompt = (
    "Here are some example games, for inspiration (do not reproduce these games exactly):"""
)
gen_game_prompt = (
    """Output the code for a complete PuzzleScript game. {from_idea_prompt} {cot_prompt}"""
    + formatting_prompt
)
gen_game_from_idea_prompt = (
    """Create a simplified `demake` of the following game idea in PuzzleScript: {game_idea}. {cot_prompt}"""
    + formatting_prompt
)
cot_prompt = (
    """First, reason about your task and determine the best plan of action. Then, write your code. """
)
game_mutate_prompt = (
    """Consider the code for the following PuzzleScript game:\n\n{parents}\n\n"""
    """Create a variation on this game, making it more complex. {cot_prompt}"""
    + formatting_prompt
)
game_crossover_prompt = (
    """Consider the code for the following PuzzleScript games:\n\n{parents}\n\n"""
    """Create a new, more complex game by combining elements of these games. {cot_prompt}"""
    + formatting_prompt
)
game_compile_repair_prompt = (
    """The following PuzzleScript game code:\n```plaintext\n{code}\n```\n"""
    """produced the following console output:\n{console_text}\n"""
    """Return a repaired version of the code that addresses these errors. {cot_prompt} {from_idea_prompt}"""
    + formatting_prompt
)
from_idea_prompt = """The game should be a simplified `demake` of the following game idea: {game_idea}"""
game_solvability_repair_prompt = (
    """The following PuzzleScript game code:\n```plaintext\n{code}\n```\n"""
    """compiled, but a solvability check returned the following error:\n{solver_text}\n"""
    """{from_idea_prompt}"""
    + formatting_prompt
)
# plan_game_prompt = (
#     "Generate a plan for a PuzzleScript game. Describe the game's story/theme, the necessary sprites, "
#     "the mechanics, and the levels that are needed to complete your vision. "
#     "Try to come up with a novel idea, which has not been done before, but which is still feasible "
#     "to implement in PuzzleScript. "
# )
plan_game_prompt = (
    "Generate a high-level plan for a PuzzleScript game. {from_idea_prompt} Think about the necessary objects, "
    "game mechanics and levels that will be needed to complete your vision."
)
gen_sprites_prompt = (
    "Consider the following PuzzleScript game development plan:\n```\n{game_plan}\n```\n"
    ". Select "
    "or generate "
    "the full set of sprites that will be needed to implement this plan. "
    "Here is the existing library of sprites which you may draw from: {sprites_library}.\n\n"
    "Output your response as a list of sprites in a ```plaintext code block, beginning with a header of the form:\n\n"
    "========\nOBJECTS\n========\n\n"
    "To generate a new sprite, define it like this:\n\n"
    "Player\n"
    "black orange white blue\n"
    ".000.\n"
    ".111.\n"
    "22222\n"
    ".333.\n"
    ".3.3.\n\n"
    ", where the first line is the sprite's name, the second line is the colors used in the sprite, "
    "and the following lines are the sprite's pixels, in a 5x54 grid, with indices (starting at 0), "
    "referring to the colors in the second line. "
    "(Colors can also be defined as hex codes, like #FF0000.) "
    "Note that you should favor re-using existing sprites. "
    "To select an existing sprite, simply list their names, like this:\n\n"
    "Player\n\n"
    # "ONLY list sprites that exist in the library above! "
    "After listing the sprites, create a legend that maps sprite names to single-character shorthands, like this:\n\n"
    "========\nLEGEND\n========\n\n"
    "P = Player\n\n"
)
gen_rules_prompt = (
    "Consider the following PuzzleScript game development plan:\n```\n{game_plan}\n```\n"
    ", with the following objects:\n\n{object_legend}\n\n"
    ". Generate the COLLISIONLAYERS, RULES, and WINCONDITIONS to implement the game logic. "
    "Format your output in a single ```plaintext code block, "
    "with each section underneath a header of the form, e.g.:\n"
    "========\nCOLLISIONLAYERS\n========\n"
    + comments_note
)
gen_levels_prompt = (
    "Consider the following PuzzleScript game development plan:\n```\n{game_plan}\n```\n"
    ", and the following partially complete PuzzleScript code:\n```plaintext\n{code}\n```\n"
    ". Generate a few levels to complete the design plan. "
    "Output your response as a list of levels in a ```plaintext code block. "
    "(You do not need to include the entire code, just the levels.)"
)
gen_finalize_prompt = (
    "Consider the following PuzzleScript game development plan:\n```\n{game_plan}\n```\n"
    ", and the following partially complete PuzzleScript code:\n```plaintext\n{code}\n```\n"
    ". Complete the code to implement the game. "
    + formatting_prompt
)

def save_prompts(sys_prompt, prompt, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(
            f"SYSTEM PROMPT:\n{sys_prompt}\n\nUSER PROMPT:\n{prompt}"
        )

        
def extract_ps_code(text):
    # User a regular expression to pull out the code block
    code_block = re.search(r'(.*)```plaintext\n(.*)```(.*)', text, re.DOTALL)
    if code_block:
        plaintext = code_block.group(1) + "..." + code_block.group(3)
        code = code_block.group(2)
        return code, plaintext
    else:
        breakpoint()


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


def gen_fewshot_examples(system_prompt, prompt):
    # Randomly add fewshot examples to the system prompt (within our token limit)
    with open('example_games.json', 'r') as f:
        example_games = json.load(f)
    n_tokens_avail = 10_000 - num_tokens_from_string(system_prompt, 'gpt-4o')
    fewshot_examples_prompt_i = fewshow_examples_prompt
    last_fewshot_examples_prompt_i = fewshot_examples_prompt_i
    while num_tokens_from_string(system_prompt + fewshot_examples_prompt_i + prompt, 'gpt-4o') < n_tokens_avail:
        last_fewshot_examples_prompt_i = fewshot_examples_prompt_i
        rand_example_i = random.randint(0, len(example_games) - 1)
        fewshot_examples_prompt_i += '\n\n' + example_games.pop(rand_example_i)
    fewshot_examples_prompt_i = last_fewshot_examples_prompt_i
    return fewshot_examples_prompt_i


def openai_text_query(system_prompt, prompt, seed):
    response = openai_client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt},
        ],
        seed=seed,
    )
    text = response.choices[0].message.content
    if text == '':
        breakpoint()
    return text


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
    n_iter = data['n_iter']
    gen_game_output_path = os.path.join(save_dir, f'{n_iter}b_code.txt')
    print(gen_game_output_path)
    if not os.path.isfile(gen_game_output_path):
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
            prompt = game_compile_repair_prompt.format(code=code, console_text=console_text, cot_prompt=cot_prompt_text,
                                                       from_idea_prompt=from_idea_prompt_i)
        else:
            prompt = game_solvability_repair_prompt.format(code=code, solver_text=solver_text,
                                                           from_idea_prompt=from_idea_prompt_i)
        # if not gen_mode == GenModes.ZERO_SHOT:
        if fewshot:
            system_prompt += gen_fewshot_examples(system_prompt, prompt)
        save_prompts(system_prompt, prompt, gen_game_prompt_output_path)
        text = openai_text_query(system_prompt, prompt, seed)
        with open(gen_game_output_path, 'w', encoding='utf-8') as f:
            f.write(text)
    else:
        with open(gen_game_output_path, 'r', encoding='utf-8') as f:
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
    return jsonify({
        'code': code,
        'text': plaintext,
        'sols': sols,
        'skip': skip,
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
    print(plan_output_path)
    if not os.path.isfile(plan_output_path):
        plan_prompt_output_path = os.path.join(save_dir, f'0a_prompt.txt')
        plan_system_prompt = game_gen_system_prompt
        from_idea_prompt_i = from_idea_prompt.format(game_idea=game_idea)
        prompt = plan_game_prompt.format(from_idea_prompt=from_idea_prompt_i)
        if fewshot:
            plan_system_prompt += gen_fewshot_examples(plan_system_prompt, prompt)   
        save_prompts(plan_system_prompt, prompt, plan_prompt_output_path)
        game_plan = openai_text_query(plan_system_prompt, prompt, seed)
        with open(plan_output_path, 'w', encoding='utf-8') as f:
            f.write(game_plan)
    else:
        with open(plan_output_path, 'r', encoding='utf-8') as f:
            game_plan = f.read()
    sprites_system_prompt = game_gen_system_prompt
    if fewshot:
        sprites_system_prompt += gen_fewshot_examples(sprites_system_prompt, '') 
    with open('example_sprite_names.txt', 'r', encoding='utf-8') as f:
        sprite_names = f.read()
    with open('example_sprites.json', 'r') as f:
        sprites_library = json.load(f)
    sprites_prompt = gen_sprites_prompt.format(game_plan=game_plan, sprites_library=sprite_names)
    sprites_output_path = os.path.join(save_dir, f'0c_sprites.txt')
    if not os.path.isfile(sprites_output_path):
        sprites_prompt_output_path = os.path.join(save_dir, f'0b_sprites_prompt.txt')
        save_prompts(sprites_system_prompt, sprites_prompt, sprites_prompt_output_path)
        sprites = openai_text_query(sprites_system_prompt, sprites_prompt, seed)
        with open(sprites_output_path, 'w', encoding='utf-8') as f:
            f.write(sprites)
        
    else:
        with open(sprites_output_path, 'r', encoding='utf-8') as f:
            sprites = f.read()

    match = re.search(r'OBJECTS\s*=+\s*(.*?)\s*=+\s*LEGEND\s*=+\s*(.*)```', sprites, re.DOTALL)
    objects = match.group(1)
    objects_list = objects.split('\n')
    # Find any objects that are just a name (a single line with a single word)
    for i, obj in enumerate(objects_list):
        if len(obj.split('\n')) == 1:
            obj = obj.strip()
            assert obj in sprites_library, f"Object {obj} not found in sprite library"
            objects_list[i] = random.choice(sprites_library[obj])
    object_legend = match.group(2)

    rules_system_prompt = game_gen_system_prompt
    if fewshot:
        rules_system_prompt += gen_fewshot_examples(rules_system_prompt, '')
    rules_prompt = gen_rules_prompt.format(game_plan=game_plan, object_legend=object_legend)
    rules_output_path = os.path.join(save_dir, f'0e_rules.txt')
    if not os.path.isfile(rules_output_path):
        rules_prompt_output_path = os.path.join(save_dir, f'0d_rules_prompt.txt')
        save_prompts(sprites_system_prompt, rules_prompt, rules_prompt_output_path)
        rules = openai_text_query(sprites_system_prompt, rules_prompt, seed)
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
        levels_system_prompt += gen_fewshot_examples(levels_system_prompt, '')
    levels_prompt = gen_levels_prompt.format(game_plan=game_plan, code=partial_code)
    levels_output_path = os.path.join(save_dir, f'0g_levels.txt')
    if not os.path.isfile(levels_output_path):
        levels_prompt_output_path = os.path.join(save_dir, f'0f_levels_prompt.txt')
        save_prompts(levels_system_prompt, levels_prompt, levels_prompt_output_path)
        levels = openai_text_query(levels_system_prompt, levels_prompt, seed)
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
        finalize_system_prompt += gen_fewshot_examples(finalize_system_prompt, '')
    finalize_prompt = gen_finalize_prompt.format(game_plan=game_plan, code=partial_code)
    finalize_output_path = os.path.join(save_dir, f'0i_code.txt')
    if not os.path.isfile(finalize_output_path):
        finalize_prompt_output_path = os.path.join(save_dir, f'0h_code_prompt.txt')
        save_prompts(finalize_system_prompt, finalize_prompt, finalize_prompt_output_path)
        code = openai_text_query(finalize_system_prompt, finalize_prompt, seed)
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


if __name__ == '__main__':
    app.run(debug=False, port=8000)
    # app.run(debug=True, port=8000)