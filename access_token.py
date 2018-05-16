#!/usr/bin/env python3

"""Proxies a request for a GitHub access_token.

This script handles the server-side part of Github authentication.
PuzzleScript uses it to get an access token after a user gives
PuzzleScript permission to write gists on their behalf.

To use this script, register a Github OAuth application at
https://github.com/settings/developers and update the OAUTH_CLIENT and
OAUTH_SECRET values below to match. Add any allowed domains to
ORIGIN_LIST (they need to use HTTPS).

Install python-requests:
    $ sudo apt-get install python3-pip
    $ sudo pip install requests

Set it up as a cgi script on your web server. The server needs to
provide the HTTP_ORIGIN header.
"""

import cgi
import json
import os
import requests
import sys

OAUTH_CLIENT = "xxxxxxxxxxxxxxxxxxxx"
OAUTH_SECRET = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
ORIGIN_LIST = [
    "www.puzzlescript.net",
    "www.increpare.com",
    "increpare.github.io",
    "sfiera.github.io",
]

LOGIN_URL = "https://github.com/login/oauth/access_token"
LOGIN_HEADERS = {
    "user-agent": "puzzlescript",
    "accept": "application/json",
}

origin = os.environ.get("HTTP_ORIGIN", "")
if not origin.startswith("https://") or (origin[8:] not in ORIGIN_LIST):
    print("Content-type: text/plain")
    print()
    json.dump({"error": "invalid origin"}, sys.stdout)
    sys.exit(0)

form = cgi.FieldStorage()
code = form.getfirst("code", "")
state = form.getfirst("state", "")

try:
    data = requests.get(
            LOGIN_URL,
            headers=LOGIN_HEADERS,
            data={
                "client_id": OAUTH_CLIENT,
                "client_secret": OAUTH_SECRET,
                "code": code,
                "state": state,
            }).json()
except Exception as e:
    data = {"error": type(e).__name__}

print("Content-type: application/json")
print("Access-Control-Allow-Origin: " + origin)
print()
json.dump(data, sys.stdout)
