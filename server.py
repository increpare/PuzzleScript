import os
import re

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
import openai

log_dir = os.path.join('logs', 'test_0')
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

game_gen_system_prompt = (
    "You are a game designer, familiar with the PuzzleScript game description language. "
)
game_gen_prompt = (
    """Output the code for a complete PuzzleScript game. """
    """Include the code in a "```plaintext" code block."""
)

def save_prompts(sys_prompt, prompt, filename):
    with open(filename, 'w') as f:
        f.write(
            f"SYSTEM PROMPT:\n {sys_prompt}\n\nUSER PROMPT:\n {prompt}"
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
    gen_game_output_path = os.path.join(log_dir, 'output.txt')
    if not os.path.isfile(gen_game_output_path):
        gen_game_prompt_output_path = os.path.join(log_dir, 'prompt.txt')
        save_prompts(game_gen_system_prompt, game_gen_prompt, gen_game_prompt_output_path)
        response = openai_client.chat.completions.create(
            model='gpt-4o',
            messages=[
                {'role': 'system', 'content': game_gen_system_prompt},
                {'role': 'user', 'content': game_gen_prompt}
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