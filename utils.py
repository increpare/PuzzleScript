import json
import os
import random
import re
import time
import dotenv
from openai import AzureOpenAI
import requests
import tiktoken

from prompts import *

dotenv.load_dotenv()

def num_tokens_from_string(string: str, model_name: str) -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def truncate_str_to_token_len(string: str, model_name: str, n_tokens: int) -> str:
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(string)
    truncated_tokens = tokens[:n_tokens]
    truncated_str = encoding.decode(truncated_tokens)
    return truncated_str


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
    else:
        # Match the code block without the final ``` delimiter, in case the the block was never closed for some reason
        code_block = re.search(r'(.*)```plaintext\n(.*)$', text, re.DOTALL)
        plaintext = code_block.group(1)
    if code_block:
        code = code_block.group(2)
        return code, plaintext
    else:
        print("No code block found in text.")
        breakpoint()
        return None, None


def gen_fewshot_examples(system_prompt, prompt, max_tokens):
    # Randomly add fewshot examples to the system prompt (within our token limit)
    with open('example_games.json', 'r') as f:
        example_games = json.load(f)
    n_tokens_avail = max_tokens - num_tokens_from_string(system_prompt, 'gpt-4o')
    fewshot_examples_prompt_i = fewshow_examples_prompt
    last_fewshot_examples_prompt_i = fewshot_examples_prompt_i
    while num_tokens_from_string(system_prompt + fewshot_examples_prompt_i + prompt, 'gpt-4o') < n_tokens_avail:
        last_fewshot_examples_prompt_i = fewshot_examples_prompt_i
        rand_example_i = random.randint(0, len(example_games) - 1)
        fewshot_examples_prompt_i += '\n```\n' + example_games.pop(rand_example_i) + '\n```\n'
    fewshot_examples_prompt_i = last_fewshot_examples_prompt_i
    return fewshot_examples_prompt_i


GPT4V_ENDPOINT = "https://aoai-physics.openai.azure.com/openai/deployments/gpt4o/chat/completions?api-version=2024-02-15-preview"
GPT4V_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
headers = {
    "Content-Type": "application/json",
    "api-key": GPT4V_KEY,
}

o_endpoint = os.getenv("ENDPOINT_URL", "https://sc-pn-m898m3wl-eastus2.openai.azure.com/")
o_key = os.getenv("O3_MINI_KEY")

# client = AzureOpenAI(  
#     azure_endpoint=o_endpoint,  
#     api_key=o_key,  
#     api_version="2024-12-01-preview",
# )

def llm_text_query(system_prompt, prompt, seed, model):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    if model == 'gpt-4o':
        payload = {
            "messages": messages,
            "temperature": 0.7,
            "top_p": 0.95,
        }
        successful_query = False
        while not successful_query:
            try:
                print('Querying openai...')
                response = requests.post(GPT4V_ENDPOINT, headers=headers, json=payload)
                response.raise_for_status() # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
                successful_query = True
                print('Query completed.')
            except requests.RequestException as e:
                print(f"Failed to make the request. RequestException: {e}")
            except requests.HTTPError as e:
                print(f"HTTPError: {e}")
            time.sleep(5)

        return response.json()['choices'][0]['message']['content']

    else:
        assert model in ['o1', 'o3-mini']
        deployment = os.getenv('DEPLOYMENT_NAME', model)
        successful_query = False
        while not successful_query:
            print('Querying openai...')
            completion = client.chat.completions.create(  
                model=deployment,
                messages=messages,
                max_completion_tokens=100000,
                stop=None,  
                stream=False
            )
            successful_query = True
            # if completion.status_code == 200:
            #     successful_query = True
            #     print('Query completed.')
            # else:
            #     print(f"Failed to make the request. Status code: {completion.status_code}")
            #     time.sleep(5)
        return completion.choices[0].message.content
        

    global openai_client
    if openai_client is None:
        openai_client = openai.Client(api_key=os.getenv('OPENAI_API_KEY'))
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

