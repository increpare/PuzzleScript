import os

from dotenv import load_dotenv
from outlines import models, generate


load_dotenv()

with open('syntax.rsc', 'r') as f:
    grammar = f.read()

model = models.openai(
    'gpt-4o',
    api_key=os.getenv('OPENAI_API_KEY'),
)

generator = generate.cfg(model, grammar)
ret = generator.generate("Generate a PuzzleScript game.")
print(ret)
breakpoint()

