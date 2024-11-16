import os

from dotenv import load_dotenv
from outlines import models, generate


load_dotenv()

import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"


with open('Syntax.rsc', 'r') as f:
    grammar = f.read()

model = models.transformers(
    model_id,
)

generator = generate.cfg(model)
ret = generator.generate("Generate a PuzzleScript game.")
print(ret)
breakpoint()

