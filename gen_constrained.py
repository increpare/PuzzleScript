import argparse
import json
import os
import random

from dotenv import load_dotenv
from lark import Lark
from outlines import models, generate, grammars

import transformers
import torch

parser = argparse.ArgumentParser(description='Generate PuzzleScript games')
parser.add_argument('--overwrite', '-o', action='store_true', help='Overwrite existing generated.txt')
args = parser.parse_args()

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

prompt = (
    "You are a game designer, familiar with the PuzzleScript game description language. "
)
fewshow_examples_prompt = (
    "Here are some example games, for inspiration (do not reproduce these games exactly):"""
)

# with open('example_games.json', 'r') as f:
    # example_games = json.load(f)
# list all files in min_games

example_games = []
for root, dirs, files in os.walk('min_games'):
    for file in files:
        if file.endswith('.txt'):
            with open(os.path.join(root, file), 'r') as f:
                example_games.append(f.read())

fewshot_examples_prompt_i = fewshow_examples_prompt
last_fewshot_examples_prompt_i = fewshot_examples_prompt_i
for i in range(3):
    last_fewshot_examples_prompt_i = fewshot_examples_prompt_i
    rand_example_i = random.randint(0, len(example_games) - 1)
    fewshot_examples_prompt_i += '\n\n' + example_games.pop(rand_example_i)
fewshot_examples_prompt_i = last_fewshot_examples_prompt_i
prompt += fewshot_examples_prompt_i
prompt += "\n\nNow, generate a PuzzleScript game."

prompt_filepath = os.path.join('temp', 'prompt.txt')
if not os.path.exists('temp'):
    os.makedirs('temp')
with open(prompt_filepath, 'w') as f:
    f.write(prompt)

with open('syntax_generate.lark', 'r') as f:
    grammar = f.read()

generated_filepath = os.path.join('temp', 'generated.txt')
if not os.path.isfile(generated_filepath) or args.overwrite:
    model = models.transformers(
        model_id,
        device='cuda:1',
        model_kwargs={'torch_dtype': torch.bfloat16},
        # device_map='auto'
    )

    generator = generate.cfg(model, grammar)
    print("Generating code...")
    # generated_code = generator(prompt, max_tokens=1024)
    # Stream the output
    stream = generator.stream(prompt, max_tokens=1024)
    all_generated_code = []
    with open(generated_filepath, 'w') as f:
        f.write('')
        for i in range(1024):
            generated_code = next(stream)
            all_generated_code.append(generated_code)
            print(generated_code)
            f.write(generated_code)
    generated_code = ''.join(all_generated_code)

    with open(generated_filepath, 'w') as f:
        f.write(generated_code)
else:
    with open(generated_filepath, 'r') as f:
        generated_code = f.read()

# Now parse the generated text
parser = Lark(grammar, start="ps_game")
parse_tree = parser.parse(generated_code)
print(parse_tree.pretty())
breakpoint()

