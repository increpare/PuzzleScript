import os
import requests
import re

from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

def return_keyval(d, key):
    if key in d:
        return d[key]
    for k, v in d.items():
        if isinstance(v, dict):
            return return_keyval(v, key)
    return None

ps_urls_path = "ps_urls.txt"
if not os.path.isfile(ps_urls_path):

    # URL of the JS file
    js_url = "https://pedros.works/puzzlescript/hyper/PGDGame.js"

    # Fetch the JS file content
    response = requests.get(js_url)
    response.raise_for_status()

    # Use regex to extract URLs
    game_links = re.findall(r'https?://\S+', response.text)
    print(response.text)

    # Display the game URLs
    [print(g) for g in game_links]
    print(f"Total links: {len(game_links)}")

    ps_links = [g for g in game_links if "puzzlescript.net/play" in g]
    ps_links = set(ps_links)
    print(f"Total PuzzleScript links: {len(ps_links)}")

    with open(ps_urls_path, "w") as f:
        f.write("\n".join(ps_links))

else:
    with open(ps_urls_path, "r") as f:
        ps_links = f.read().splitlines()
visited_ps_links_path = "visited_ps_links.txt"
if os.path.isfile(visited_ps_links_path):
    with open(visited_ps_links_path, "r") as f:
        visited_ps_links = set(f.read().splitlines())
else:
    with open(visited_ps_links_path, "w") as f:
        f.write("")
        visited_ps_links = []
if not os.path.isdir('scraped_games'):
    os.mkdir('scraped_games')
for link in ps_links:
    if link in visited_ps_links:
        print(f"Skipping {link}")
        continue
    if 'play' not in link:
        breakpoint()
    else:
        hack_link = link.replace('play.html?p=', 'editor.html?hack=')
        id = hack_link.split('hack=')[1].strip('"')
        script_path = os.path.join('scraped_games', f"{id}.txt")
        print(f'hack link: {hack_link}')
        git_url = f"https://api.github.com/gists/{id}"
        
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            "X-GitHub-Api-Version": "2022-11-28"
        }

        def add_to_visited():
            with open(visited_ps_links_path, "a", encoding='utf-8') as f:
                f.write(f"{link}\n")
        
        try:
            response = requests.get(git_url, headers=headers)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"Error: {e}")
            add_to_visited()
            continue
        
        gist = response.json()
        if 'script.txt' not in gist['files']:
            # Then look recursively thru dicts until we find a 'content'
            script = return_keyval(gist['files'], 'content')

        else:
            script =  gist['files']['script.txt']['content']

        with open(script_path, "w", encoding='utf-8') as f:
            f.write(script)
        add_to_visited()

# Count number of scripts
script_files = os.listdir("scraped_games")
print(f"Total scripts: {len(script_files)}")
