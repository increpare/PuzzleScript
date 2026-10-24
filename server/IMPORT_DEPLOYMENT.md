# Standalone Hack POST import

Prepared 2026-10-24; deployment authorization received after the initial read-only pass.

Exports contain a **Edit game** form that opens `https://www.puzzlescript.net/editor.html`
in a new tab with a UTF-8 URL-encoded `source` field. `nohack` in the prelude
omits the button; it does not remove source from the export. No spoiler prompt.
The endpoint accepts any submitting origin, including `Origin: null` from local
files. This is ordinary form navigation, so it needs no CORS headers.

Import responses also use a script policy allowing deployed same-origin scripts
and the engine’s existing generated functions, while blocking inline event
attributes. A focused parser fix escapes unknown-prelude text in diagnostics.
Ordinary GET responses retain their existing policy.

The receiver reads the current deployed editor HTML, replaces its inert
`importedSource` JSON marker, and supplies `<base href="/">` for assets, links,
and XHRs. The editor loads imported text before checking demos, gists, or saved
games. Responses use `Cache-Control: no-store` and `Referrer-Policy: no-referrer`.
Source is not written to server disk, logged, evaluated, or turned into a hosted game. Normal editor saves/autosaves in browser localStorage remain available.
The browser may retain it in the open document; resubmitting a POST on refresh
may show the browser's normal confirmation.

The encoded body limit is **33,554,432 bytes (32 MiB)**, both at Caddy and in
Python. Percent-encoded UTF-8 can take three times as many bytes as the source.
The receiver returns 413 for oversized bodies, 400 for incomplete/malformed
forms, 415 for unsupported content types, 405 for non-POST calls to the receiver, and 503 when the
editor is missing or hasn't been deployed with its import marker. The adapter
uses a 20-second absolute request-read deadline and closes the import connection.
It collects partial reads from the legacy CGI handler’s unbuffered input. It does not
change the existing OAuth/gist CGI origin policies or credentials.

## Verified live arrangement

Read-only SSH checks from the MacBook confirmed:

- Both `locus@ded.increpare.com` and `root@ded.increpare.com` accept the existing
  SSH credentials; `id` confirmed uid 1000 and uid 0 respectively.
- Caddy is running with `/etc/caddy/Caddyfile` (root-owned).
- `www.puzzlescript.net` serves `/home/locus/sites/puzzlescript`, using precompressed
  gzip assets. The bare domain redirects to `www`. Editor GET returned 200;
  `/import` returned 404.
- `/usr/lib/cgi-bin` and Apache configuration do not exist on this server.
- Legacy `/cgi-bin/*` on ded is proxied to `127.0.0.1:8081`.
- `cgi-bin.service` runs `/usr/bin/python3 /srv/cgi/cgiserver.py`, Python 3.11.2,
  as locus, with working directory `/srv/cgi` and read-only home protection.
- `/srv/cgi/cgiserver.py` is root-owned. It uses a single-threaded
  `CGIHTTPRequestHandler`, forwarding Origin to the existing scripts. Existing
  credential-bearing CGI scripts were not opened.

Python's legacy CGI handler sends HTTP 200 before a CGI script runs, so a plain
CGI `Status:` header cannot deliver the intended errors here. The supplied
`cgiserver-import.patch` dispatches only POST `/editor.html` to `import_source.handle_http`;
all other requests keep the current handler. Caddy routes only POST; GET continues serving the static editor at the same URL. No additional service, port, Python
dependency, or persistent source storage is needed. `import_source.py` also
supports conventional CGI servers through its `main()` entry point.

## Deployment steps

1. Run the checks below. A future full release uses `npm run build`. For this
   activation, the live site was older than the MacBook HEAD, so a full build
   was deliberately not uploaded. `prepare_live_import.js BASELINE OUTPUT`
   rebuilds only the affected live editor bundle, using its source-map file
   order and original `js/source/` files. It patches the deployed editor,
   export template, parser, autocomplete and documentation. The live player
   page and player bundle are unchanged. Review its `manifest.json` and deploy
   only those files, including their matching gzip companions. The editor marker
   must be present before enabling the route. The changed bundle URL is versioned
   to avoid old browser caches skipping the import initializer.
2. Stage the two server files (these commands do not enable the endpoint):

   ```sh
   scp server/import_source.py server/cgiserver-import.patch root@ded.increpare.com:/tmp/
   ```

3. On ded as root, save backups and install the receiver/adapter. Re-inspect
   the launcher before applying: the patch was made against the verified
   2026-10-24 contents and must fail rather than overwrite a changed launcher.

   ```sh
   stamp=$(date -u +%Y%m%dT%H%M%SZ)
   cp -a /srv/cgi/cgiserver.py /srv/cgi/cgiserver.py.backup.import-$stamp
   cp -a /etc/caddy/Caddyfile /etc/caddy/Caddyfile.backup.import-$stamp
   patch --dry-run --directory=/srv/cgi --strip=0 < /tmp/cgiserver-import.patch
   install -o root -g root -m 644 /tmp/import_source.py /srv/cgi/import_source.py
   patch --directory=/srv/cgi --strip=0 < /tmp/cgiserver-import.patch
   /usr/bin/python3 -m py_compile /srv/cgi/import_source.py /srv/cgi/cgiserver.py
   ```

4. In the existing `www.puzzlescript.net` block of `/etc/caddy/Caddyfile`, add
   the following before its `file_server` directive. Keep root, compression,
   precompressed gzip, and the existing header unchanged. Do not put this route
   on ded or the redirecting bare domain; the editor assets must share its origin.

   ```caddy
   @editor_import {
       path /editor.html
       method POST
   }
   handle @editor_import {
       request_body {
           max_size 33554432
       }
       reverse_proxy 127.0.0.1:8081
   }
   ```

5. Validate first, then restart the existing CGI process and reload Caddy:

   ```sh
   /usr/bin/caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
   systemctl restart cgi-bin.service
   systemctl reload caddy
   systemctl is-active cgi-bin.service caddy
   ```

6. Check the responses and browser flow using non-private test source:

   ```sh
   curl -i https://www.puzzlescript.net/editor.html
   curl -i --data-urlencode 'source=title Import check' https://www.puzzlescript.net/editor.html
   ```

   Expect 200 for both: GET is the normal static editor; POST is no-store. Inspect the POST HTML for
   the base URL and escaped JSON. Open a freshly exported valid game from
   `file://` and a hosted page, choose Edit game, and confirm exact source loads,
   assets/compilation work, and a pre-existing saved game doesn't replace it.
   Also export a game with `nohack` and confirm the button is absent. Check
   `journalctl -u cgi-bin.service` for errors and smoke-test the existing gist
   proxy/OAuth flow. A 5 MiB form-navigation test should be repeated against
   HTTPS after deployment; local receiver and HTTP-adapter tests cover it now.

Rollback: restore the saved Caddyfile and launcher, validate Caddy, restart
`cgi-bin.service`, and reload Caddy. The new Python module may remain unused.
Exports made with this feature will need the endpoint, so avoid distributing
those exports until the route is verified.

## Local checks

```sh
python3 -m unittest discover -s server -p 'test_import_source.py'
node src/tests/standalone_hack_test.js
npm test
npm run build
node src/tests/standalone_hack_test.js src/standalone_inlined.txt
node src/tests/standalone_hack_test.js bin/standalone_inlined.txt
```

The build increments `.build/buildnumber.txt` and regenerates
`src/standalone_inlined.txt`; preserve and review any pre-existing changes.
The normal static development server does not handle POST `/editor.html`.


## Activated deployment and verification

Activated 2026-10-24, with original rollback backup on ded at
`/root/puzzlescript-import-backups/20261024T013306Z/` (03:33 Berlin).
`revision2/` holds the intermediate receiver/assets and the pre-cache-version
editor HTML. The original `site-assets.tar.gz`, `manifest.json`,
`missing-assets.json`, `Caddyfile`, and `cgiserver.py` remain the complete
pre-feature rollback point. No Git staging, commits, pushes, or PRs were made.

Only 17 public files were uploaded:

- `editor.html` and `.gz`
- `standalone_inlined.txt` and `.gz`
- `Documentation/prelude.html` and `.gz`
- `js/scripts_compiled.js`, `.gz`, and `.map`
- `js/source/editor.js`, `buildStandalone.js`, `parser.js`, and
  `codemirror/anyword-hint.js`, each with its `.gz` companion

Server changes: `/srv/cgi/import_source.py`, the POST-only adapter in
`/srv/cgi/cgiserver.py`, and the POST `/editor.html` matcher in
`/etc/caddy/Caddyfile`. Existing CGI secrets, service unit, ports, normal GET
routes, player page/bundle, and unrelated website assets were unchanged.
The older live standalone template keeps its current player engine/layout;
its new Edit game form is in the existing footer on desktop and moves into
its existing touch menu. The source checkout uses modern `playerLinks` and
its already-established footer/menu placement for the next full release.

Verified: 785 existing engine tests; seven receiver tests including short reads
and a real localhost HTTP transfer; export/nohack, exact source round-trip,
metadata highlighting and autocomplete tests; full isolated release build;
live normal demo and gist loading (including Legend of Zokoban); live 5 MiB POST; Unicode and HTML/script-end text round-trip; 400/413/415 errors
with no-store; public nohack documentation. The coordinating Windows chat
independently exported from the live editor, opened the actual HTML via
file://, clicked Edit game, and confirmed exact source for both the small demo
and a 5,243,790-byte valid source, plus nohack omission and METADATA highlighting.
Mac Chrome additionally imported source containing harmless HTML event markers,
clicked RUN, and confirmed no marker execution. Unknown-prelude text rendered
escaped; the existing metadata-warning HTML was contained by the import-only
script policy. Imports populate the editor; the user still clicks RUN to compile.

Imported pages dispatch diagnostic and definition links through an external
console click listener that accepts only exact `jumpToLine(NUMBER)` calls
(including quoted positive integer targets) and safe integer line numbers. It
calls the known function directly, never evaluates handler strings, and prevents
the javascript: URL navigation. The script policy still blocks arbitrary inline
handlers. Ordinary GET editor links retain their existing behavior. Normal editor
browser saves/autosaves remain enabled. No source is stored on the server.

To roll back completely, remove the route by restoring the original Caddyfile,
restore the original CGI launcher, restore archived site files, and remove only
new public files listed in `missing-assets.json`; remove the new receiver module
separately, since it did not exist before. Validate Caddy, restart cgi-bin.service, and reload
Caddy. Keep the root-only backups until rollback is no longer needed.


## Diagnostic navigation follow-up

Deployed a focused seven-file update on 2026-10-24: editor HTML/gzip, editor
source/gzip, and the compiled editor bundle/gzip/map. No receiver, Caddy, service,
player, or other site changes were needed. The bundle cache version advanced to
`?v=edit-game-20261024-jumps`. Its rollback archive and manifest are under the
original backup directory in `diagnostic-jumps-20261024T014602Z/`.

Affected tests pass for ordinary editor initialization, imported-page navigation,
quoted definition/rule targets, and rejection of appended commands, expressions,
negative/fractional/unsafe integers and unrelated handlers. Live Chrome imported
the harmless HTML event-marker source, clicked RUN, then clicked diagnostic links
for lines 2 and 1. The DOM-backed CodeMirror active line changed to 2 and 1
respectively; no event-marker data attribute appeared. Main RUN, rebuild, save,
export and selection controls already use listeners from external scripts; their
handling and ordinary GET behavior were unchanged.

The encoded POST body limit remains exactly 33,554,432 bytes (32 MiB). UTF-8 and
URL encoding affect the practical source limit; percent-encoded bytes take three
ASCII bytes each, so the limit is not a guaranteed 32 MiB of original source.
The actual 5,243,790-byte source local-file round trip remains verified.
