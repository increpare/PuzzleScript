import os
import re

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
import openai


starter_game = 'sokoban_basic'
log_dir = os.path.join('logs', f'{starter_game}_0')
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
game_gen_0shot_prompt = (
    """Output the code for a complete PuzzleScript game. """
    + formatting_prompt
)
game_mutate_prompt = (
    """Consider the following PuzzleScript game code:\n```plaintext\n{code}\n```\n"""
    """Create a variation on this game. """
    + formatting_prompt
)
game_repair_prompt = (
    """The following PuzzleScript game code:\n```plaintext\n{code}\n```\n"""
    """Produced the following console output:\n{console_text}\n"""
    """Return a repaired version of the code that addresses these errors. """
    + formatting_prompt
)

def save_prompts(sys_prompt, prompt, filename):
    with open(filename, 'w') as f:
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
    console_text = data['console_text']
    n_iter = data['n_iter']
    if console_text:
        compile_output_path = os.path.join(log_dir, f'{data["n_iter"]-1}c_compile.txt')
        if not os.path.isfile(compile_output_path):
            with open(compile_output_path, 'w') as f:
                f.write(console_text)
    gen_game_output_path = os.path.join(log_dir, f'{n_iter}b_code.txt')
    if not os.path.isfile(gen_game_output_path):
        breakpoint()
        gen_game_prompt_output_path = os.path.join(log_dir, f'{n_iter}a_prompt.txt')
        system_prompt = game_gen_system_prompt
        if code:
            prompt = game_repair_prompt.format(code=code, console_text=console_text)
        else:
            # prompt = game_gen_prompt
            with open(os.path.join('src', 'demo', f'{starter_game}.txt'), 'r') as f:
                code = f.read()
            prompt = game_mutate_prompt.format(code=code)
        save_prompts(system_prompt, prompt, gen_game_prompt_output_path)
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