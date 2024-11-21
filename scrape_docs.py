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
    print(container_tag.text)
    breakpoint()
    return container_tag.text

# example usage
html_content = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>collision layers</title>
</head>
<body>
    <p>this is a sample documentation text.</p>
    <div>additional information can be found here.</div>
    <span>more details are provided in this section.</span>
</body>
</html>
"""

docs_dir = os.path.join('src', 'Documentation')

for filename in os.listdir(docs_dir):
    if filename.endswith('.html'):
        file_path = os.path.join(docs_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
            documentation = extract_documentation(html_content)
            print(documentation)
            breakpoint()