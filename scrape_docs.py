import os
from bs4 import BeautifulSoup

def extract_documentation(html_content):
    # parse the html content
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # extract relevant documentation text
    # assuming the documentation text is within <p> tags or similar
    documentation_text = []
    body_tag = soup.find('body')
    container_tag = body_tag.find('div', {'class': 'container'}, recursive=False)
    if container_tag is not None:
        return container_tag.text

docs_dir = os.path.join('src', 'Documentation')
files_to_skip = set(['index', 'gifs', 'faq', 'privacypolicy', 'randomness', 'sounds',
                     'visual_debugger', 'leveleditor', 'keyboard_shortcuts', 'documentation',
                     'credits', 'about', 'extract_from_standalone', 'permanent_urls'])

all_documentation = []
for filename in os.listdir(docs_dir):
    if filename.endswith('.html'):
        # strip the .html extension
        if filename[:-5] in files_to_skip:
            continue
        file_path = os.path.join(docs_dir, filename)
        print(f'Extracting documentation from {file_path}')
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
            documentation = extract_documentation(html_content)
            if documentation is not None:
                all_documentation.append(documentation)

# all_documentation now contains the extracted documentation text from all HTML files in the directory
all_doc_str = '\n'.join(all_documentation)

# using regex, replace any 3+ newlines with 2 newlines
import re
all_doc_str = re.sub(r'\n{3,}', '\n\n', all_doc_str)

with open('all_documentation.txt', 'w', encoding='utf-8') as file:
    file.write(all_doc_str)