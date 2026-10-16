#!/usr/bin/env python3

"""Proxies a request for GitHub gist content.

This script handles fetching public gist content server-side using
a GitHub personal access token, so clients don't need authentication.

To use this script:
1. Generate a GitHub personal access token with 'gist' scope
2. Update GITHUB_TOKEN below with your token
3. Set it up as a CGI script on your web server
4. Add your domain to ORIGIN_LIST

Install python-requests:
    $ sudo apt-get install python3-pip
    $ sudo pip install requests
"""

import cgi
import json
import os
import requests
import sys

# Replace with your GitHub personal access token
#note: LOCATED in /usr/lib/cgi-bin on server with private key substituted, public address https://ded.increpare.com/cgi-bin/access_token.py (see auth.html)
GITHUB_TOKEN = "INSERT github_pat_blah private key here"

ORIGIN_LIST = [
    "www.puzzlescript.net",
    "www.increpare.com",
    "ded.increpare.com",
    "increpare.github.io",
    "sfiera.github.io",
    "www.flickgame.org",
    "www.tinychoice.net",
    "tinychoice.net",
    "www.plingpling.org",
    "plingpling.org",
    "www.flickgame.org",
    "flickgame.org",
]

GITHUB_API_URL = "https://api.github.com/gists/"
HEADERS = {
    "user-agent": "puzzlescript-gist-proxy",
    "accept": "application/vnd.github+json",
    "authorization": f"Bearer {GITHUB_TOKEN}",
    "x-github-api-version": "2022-11-28",
}

# Check origin
origin = os.environ.get("HTTP_ORIGIN", "")
if not origin.startswith("https://") or (origin[8:] not in ORIGIN_LIST):
    print("Content-type: application/json")
    print()
    json.dump({"error": "invalid origin"}, sys.stdout)
    sys.exit(0)

# Get gist ID from query parameters
form = cgi.FieldStorage()
gist_id = form.getfirst("id", "")

if not gist_id:
    print("Content-type: application/json")
    print("Access-Control-Allow-Origin: " + origin)
    print()
    json.dump({"error": "no gist id provided"}, sys.stdout)
    sys.exit(0)

# Clean the gist ID (remove any path characters for security)
gist_id = gist_id.replace("/", "").replace("\\", "")

try:
    # Fetch gist data from GitHub API
    response = requests.get(
        GITHUB_API_URL + gist_id,
        headers=HEADERS,
        timeout=10
    )
    
    if response.status_code == 200:
        gist_data = response.json()
        
        # Extract script.txt content
        files = gist_data.get("files", {})
        script_file = files.get("script.txt")
        
        if script_file and "content" in script_file:
            result = {"content": script_file["content"]}
        else:
            result = {"error": "script.txt not found in gist"}
    elif response.status_code == 404:
        result = {"error": "gist not found"}
    elif response.status_code == 403:
        result = {"error": "rate limit exceeded or access denied"}
    else:
        result = {"error": f"github api error: {response.status_code}"}
        
except requests.exceptions.Timeout:
    result = {"error": "request timeout"}
except requests.exceptions.RequestException as e:
    result = {"error": f"request failed: {str(e)}"}
except Exception as e:
    result = {"error": f"unexpected error: {str(e)}"}

print("Content-type: application/json")
print("Access-Control-Allow-Origin: " + origin)
print()
json.dump(result, sys.stdout)
