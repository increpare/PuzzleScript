import binascii as ba
from enum import IntEnum
import json
import os
import random
import re

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
import openai
import requests

from utils import num_tokens_from_string, truncate_str_to_token_len


load_dotenv()
openai_client = openai.Client(api_key=os.getenv('OPENAI_API_KEY'))
app = Flask(__name__)


@app.route('/')
def serve_editor():
    return send_from_directory('src', 'editor.html')

# Route to serve JavaScript files dynamically
@app.route('/<path:filename>')
def serve_js(filename):
    return send_from_directory('src', filename)

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
from_idea_prompt = """The game should be a simplified `demake` of the following game idea: {game_idea}. """
game_solvability_repair_prompt = (
    """The following PuzzleScript game code:\n```plaintext\n{code}\n```\n"""
    """compiled, but a solvability check returned the following error:\n{solver_text}\n"""
    """{from_idea_prompt}"""
    + formatting_prompt
)
gen_game_plan_prompt = (
    "Generate a plan for a PuzzleScript game. Describe the game's story/theme, the necessary sprites, "
    "the mechanics, and the levels that are needed to complete your vision. "
    "Try to come up with a novel idea, which has not been done before, but which is still feasible "
    "to implement in PuzzleScript. "
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


@app.route('/save_sols', methods=['POST'])
def save_sols():
    data = request.json
    sols = data['sols']
    n_iter = data['n_iter']
    save_dir = os.path.join('logs', data['save_dir'])
    sols_path = os.path.join(save_dir, f'{n_iter}e_sols.json')
    with open(sols_path, 'w') as f:
        json.dump(sols, f, indent=4)
    gif_urls = data['gif_urls']
    for gif_url, level_i in gif_urls:
        # Download the gif from the url
        with open(os.path.join(save_dir, f'sol_level-{level_i}.gif'), 'wb') as f:
            f.write(ba.a2b_base64(gif_url.split(',')[1]))
    return jsonify({})


@app.route('/gen_game', methods=['POST'])
def gen_game():
    data = request.json
    seed = data['seed']
    fewshot = data['fewshot']
    cot = data['cot']
    from_idea = data['from_idea']
    game_idea = data['game_idea']
    cot_prompt_text = cot_prompt if cot else ''
    log_dir = os.path.join(
        'logs',
    #     (
    #         ('fewshot' if fewshot else 'zeroshot') + \
    #         ('_cot' if cot else '') + \
    #         f'_{seed}'
    #     )
    )
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
    if console_text:
        compile_output_path = os.path.join(save_dir, f'{data["n_iter"]-1}c_compile.txt')
        if not os.path.isfile(compile_output_path):
            with open(compile_output_path, 'w') as f:
                f.write(console_text)
        solver_output_path = os.path.join(save_dir, f'{data["n_iter"]-1}d_solver.txt')
        if solver_text and not os.path.isfile(solver_output_path):
            with open(solver_output_path, 'w') as f:
                f.write(solver_text)
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
            system_prompt += fewshot_examples_prompt_i
        save_prompts(system_prompt, prompt, gen_game_prompt_output_path)
        response = openai_client.chat.completions.create(
            model='gpt-4o',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt},
            ],
            seed=seed,
        )
        text = response.choices[0].message.content
        with open(gen_game_output_path, 'w', encoding='utf-8') as f:
            f.write(text)
    else:
        with open(gen_game_output_path, 'r', encoding='utf-8') as f:
            text = f.read()
    if text == '':
        breakpoint()
    code, plaintext = extract_ps_code(text)
    if 'randomDir' in code:
        skip = True
    else:
        skip = False
    # If the code we just generated has already been solved, pass it to the client so it doesn't apply the solver to it
    sols_path = os.path.join(save_dir, f'{n_iter}e_sols.json')
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


if __name__ == '__main__':
    # app.run(debug=False, port=8000)
    app.run(debug=True, port=8000)