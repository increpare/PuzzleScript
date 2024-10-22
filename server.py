from enum import IntEnum
import json
import os
import random
import re

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
import openai

from utils import num_tokens_from_string, truncate_str_to_token_len


class GenModes(IntEnum):
    ZERO_SHOT = 0
    FEWSHOT = 1
    MUTATE = 2


seed = 2
gen_mode = GenModes.FEWSHOT
if gen_mode == GenModes.ZERO_SHOT:
    exp_name = f'zero_shot_{seed}'
elif gen_mode == GenModes.FEWSHOT:
    exp_name = f'few_shot_{seed}'
elif gen_mode == GenModes.MUTATE:
    starter_game = 'sokoban_basic'
    exp_name = f'{starter_game}_{seed}'
log_dir = os.path.join('logs', exp_name)
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
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
game_gen_0shot_prompt = (
    """Output the code for a complete PuzzleScript game. """
    + formatting_prompt
)
game_mutate_prompt = (
    """Consider the following PuzzleScript game code:\n```plaintext\n{code}\n```\n"""
    """Create a variation on this game. """
    + formatting_prompt
)
game_compile_repair_prompt = (
    """The following PuzzleScript game code:\n```plaintext\n{code}\n```\n"""
    """produced the following console output:\n{console_text}\n"""
    """Return a repaired version of the code that addresses these errors. """
    + formatting_prompt
)
game_solvability_repair_prompt = (
    """The following PuzzleScript game code:\n```plaintext\n{code}\n```\n"""
    """compiled, but a solvability check returned the following error:\n{solver_text}\n"""
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
    

@app.route('/gen_game', methods=['POST'])
def gen_game():
    data = request.json
    code = data['code']
    compilation_success = data['compilation_success']
    console_text = data['console_text']
    solver_text = data['solver_text']
    n_iter = data['n_iter']
    if console_text:
        compile_output_path = os.path.join(log_dir, f'{data["n_iter"]-1}c_compile.txt')
        if not os.path.isfile(compile_output_path):
            with open(compile_output_path, 'w') as f:
                f.write(console_text)
        solver_output_path = os.path.join(log_dir, f'{data["n_iter"]-1}d_solver.txt')
        if solver_text and not os.path.isfile(solver_output_path):
            with open(solver_output_path, 'w') as f:
                f.write(solver_text)
    gen_game_output_path = os.path.join(log_dir, f'{n_iter}b_code.txt')
    if not os.path.isfile(gen_game_output_path):
        gen_game_prompt_output_path = os.path.join(log_dir, f'{n_iter}a_prompt.txt')
        system_prompt = game_gen_system_prompt
        if n_iter == 0:
            # prompt = game_gen_prompt
            if gen_mode == GenModes.ZERO_SHOT:
                prompt = game_gen_0shot_prompt
            elif gen_mode == GenModes.FEWSHOT:
                prompt = game_gen_0shot_prompt
            elif gen_mode == GenModes.MUTATE:
                with open(os.path.join('src', 'demo', f'{starter_game}.txt'), 'r') as f:
                    code = f.read()
                prompt = game_mutate_prompt.format(code=code)
        elif not compilation_success:
            prompt = game_compile_repair_prompt.format(code=code, console_text=console_text)
        else:
            prompt = game_solvability_repair_prompt.format(code=code, solver_text=solver_text)
        if not gen_mode == GenModes.ZERO_SHOT:
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
        if n_iter % 10 == 0:
            breakpoint()
        response = openai_client.chat.completions.create(
            model='gpt-4o',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt},
            ]
        )
        text = response.choices[0].message.content
        with open(gen_game_output_path, 'w') as f:
            f.write(text)
    else:
        with open(gen_game_output_path, 'r') as f:
            text = f.read()
    code, plaintext = extract_ps_code(text)
    return jsonify({
        'code': code,
        'text': plaintext,
    })


if __name__ == '__main__':
    app.run(debug=True, port=8000)