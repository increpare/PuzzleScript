import json
import os
import re

from dotenv import load_dotenv
import openai


load_dotenv()
openai_client = openai.Client(api_key=os.getenv('OPENAI_API_KEY'))
seed = 0

system_prompt = "You are an indie game designer, used to thinking outside of the box in order to come up with games that are fun to play while also relatively straightforward to implement."

prompt = "Come up with 10 unique and pithy game ideas for 2D tile-based indie games. Enumerate them as a list of the form '1. [Game idea]'. Each idea should be a single sentence long."

output_path = os.path.join('logs', f'brainstorm_s-{seed}')
output_path_text = output_path + '.txt'
output_path_json = output_path + '.json'

if not os.path.isfile(output_path_text):
    response = openai_client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt},
        ],
        seed=seed,
    )
    text = response.choices[0].message.content
    with open(output_path_text, 'w', encoding='utf-8') as f:
        f.write(text)

else:
    with open(output_path_text, 'r', encoding='utf-8') as f:
        text = f.read()
    # Convert to a list of strings with reges, where each entry is the string following a number and a period
    ideas = re.findall(r'\d+\.\s*(.*)', text)

    with open(output_path_json, 'w') as f:
        json.dump(ideas, f, indent=4)
