#!/usr/bin/env python3
"""Transient form-POST import of game text into the deployed PuzzleScript editor.

No dependencies or storage. See IMPORT_DEPLOYMENT.md for the Caddy/CGI adapter.
The 32 MiB limit applies to the encoded body (UTF-8 percent encoding can triple
its size). Accept any Origin, including local files' null Origin: this creates
an editor document, not a saved game or an authenticated state change.
"""
import html
from http import HTTPStatus
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import parse_qsl

MAX_BODY_BYTES = 32 * 1024 * 1024
EDITOR_PATH = Path('/home/locus/sites/puzzlescript/editor.html')
SOURCE_MARKER = '<script id="importedSource" type="application/json">null</script>'


def handle_request(environ, stream, editor_path=EDITOR_PATH):
    headers = {'Content-Type': 'text/html; charset=utf-8',
               'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer',
               'X-Content-Type-Options': 'nosniff',
               # Imported games compile using generated functions. Scripts come
               # from the deployed editor; source-derived event attributes must
               # never execute even if an existing diagnostic contains HTML.
               'Content-Security-Policy': "script-src 'self' 'unsafe-eval'; script-src-attr 'none'; object-src 'none'; base-uri 'self'"}

    def error(status, message):
        return status, headers, ('<!doctype html><html lang="en"><meta charset="utf-8">'
                                 '<title>Game import failed</title><h1>Game import failed</h1><p>'
                                 + html.escape(message) + '</p></html>')

    if environ.get('REQUEST_METHOD') != 'POST':
        headers['Allow'] = 'POST'
        return error(405, 'Open an exported game and use its Hack button to import its source.')
    if environ.get('CONTENT_TYPE', '').split(';')[0].strip().lower() != 'application/x-www-form-urlencoded':
        return error(415, 'Submit the game source using a UTF-8 URL-encoded form.')
    length_text = environ.get('CONTENT_LENGTH', '')
    if not re.fullmatch(r'[0-9]{1,10}', length_text):
        return error(400, 'Missing or invalid request length.')
    length = int(length_text)
    if length > MAX_BODY_BYTES:
        return error(413, 'The encoded game source exceeds the 32 MiB import limit.')
    try:
        chunks = []
        remaining = length
        while remaining:
            chunk = stream.read(min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        body = b''.join(chunks)
    except OSError:
        return error(400, 'The game source was not received completely. Please try again.')
    if len(body) != length:
        return error(400, 'The game source was not received completely.')
    try:
        text = body.decode('utf-8', errors='strict')
        if re.search(r'%(?![0-9a-fA-F]{2})', text):
            raise ValueError('Invalid percent escape')
        fields = parse_qsl(text, keep_blank_values=True, strict_parsing=True,
                           encoding='utf-8', errors='strict', max_num_fields=1)
        if len(fields) != 1 or fields[0][0] != 'source' or not fields[0][1]:
            raise ValueError('Expected one nonempty source field')
        source = fields[0][1]
    except (UnicodeError, ValueError):
        return error(400, 'Expected exactly one nonempty UTF-8 source field.')
    try:
        editor = Path(editor_path).read_text(encoding='utf-8')
        if editor.count(SOURCE_MARKER) != 1:
            raise ValueError('Missing editor import marker')
    except (OSError, UnicodeError, ValueError):
        return error(503, 'The editor is temporarily unavailable. Please try again later.')

    # JSON is inert text in a data script. Escape < to prevent HTML end tags;
    # ensure_ascii also protects Unicode line separators and round-trips UTF-8.
    data = json.dumps(source, ensure_ascii=True).replace('<', '\\u003c').replace('&', '\\u0026')
    bootstrap = '<base href="/"><script id="importedSource" type="application/json">' + data + '</script>'
    return 200, headers, editor.replace(SOURCE_MARKER, bootstrap, 1)


def handle_http(handler):
    """Adapter for the existing localhost CGI server, with real HTTP statuses."""
    handler.close_connection = True
    deadline = time.monotonic() + 20

    class DeadlineInput:
        def read(self, size):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Import request deadline exceeded')
            handler.connection.settimeout(remaining)
            return handler.rfile.read(size)

    environ = {'REQUEST_METHOD': handler.command,
               'CONTENT_TYPE': handler.headers.get('Content-Type', ''),
               'CONTENT_LENGTH': handler.headers.get('Content-Length', '')}
    status, headers, body = handle_request(environ, DeadlineInput(), EDITOR_PATH)
    handler.connection.settimeout(20)
    encoded = body.encode('utf-8')
    handler.send_response(status)
    for key, value in headers.items():
        handler.send_header(key, value)
    handler.send_header('Content-Length', str(len(encoded)))
    handler.send_header('Connection', 'close')
    handler.end_headers()
    if handler.command != 'HEAD':
        handler.wfile.write(encoded)


def main():
    status, headers, body = handle_request(os.environ, sys.stdin.buffer)
    output = ('Status: ' + str(status) + ' ' + HTTPStatus(status).phrase + '\r\n'
              + ''.join(key + ': ' + value + '\r\n' for key, value in headers.items())
              + '\r\n' + body)
    sys.stdout.buffer.write(output.encode('utf-8'))


if __name__ == '__main__':
    main()
