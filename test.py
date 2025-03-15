
import os  
import base64
from openai import AzureOpenAI  

import dotenv

dotenv.load_dotenv()

endpoint = os.getenv("ENDPOINT_URL", "https://sc-pn-m898m3wl-eastus2.openai.azure.com/")  
deployment = os.getenv("DEPLOYMENT_NAME", "o1")  
subscription_key = os.getenv("O3_MINI_KEY")
print(endpoint, subscription_key)

# Initialize Azure OpenAI Service client with key-based authentication    
client = AzureOpenAI(  
    azure_endpoint=endpoint,  
    api_key=subscription_key,  
    api_version="2024-12-01-preview",
)


# IMAGE_PATH = "YOUR_IMAGE_PATH"
# encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('ascii')

#Prepare the chat prompt 
chat_prompt = [
    {
        "role": "developer",
        "content": [
            {
                "type": "text",
                "text": "You are an AI assistant that helps people find information."
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Generate a novel PuzzleScript game. Output the full code."
            }
        ]
    }
] 

# Include speech result if speech is enabled  
messages = chat_prompt  

# Generate the completion  
completion = client.chat.completions.create(  
    model=deployment,
    messages=messages,
    max_completion_tokens=100000,
    stop=None,  
    stream=False
)

print(completion.to_json())  
 