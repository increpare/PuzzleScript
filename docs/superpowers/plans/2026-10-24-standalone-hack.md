# Standalone Hack form implementation plan

The agreed design uses a normal POST navigation from exported games to
https://www.puzzlescript.net/editor.html, with `source` containing the game text.
`nohack` removes the exported button. No warning, URL payload, or hosted game copy.

- [x] Add receiver tests for UTF-8/form parsing, 5 MiB source, HTML/script escaping,
  request limits, malformed requests, and deployed-editor asset resolution.
- [x] Implement standard-library `server/import_source.py`, callable by the existing
  HTTP handler and usable as CGI. Limit encoded request bodies to 32 MiB.
- [x] Add export tests; add the Hack form, safe source serialization, imported-source
  initialization before saved-game loading, and nohack parser/autocomplete/docs.
- [x] Prepare the exact Caddy route and a narrowly scoped patch for the verified
  `/srv/cgi/cgiserver.py` launcher, preserving its existing CGI routes.
- [x] Run feature tests, existing engine tests, and release build; preserve existing
  changes to the build counter and generated standalone template. Do not commit or
  push; deployment was authorized after initial inspection. Record access/configuration evidence and deployment commands.

Verified server arrangement (read-only SSH, 2026-10-24): Caddy serves
`/home/locus/sites/puzzlescript`; `/import` currently returns 404. Existing CGI is
`cgi-bin.service`, running Python 3.11.2 `/srv/cgi/cgiserver.py` at 127.0.0.1:8081
as locus. Configuration and launcher are root-owned and not writable by locus;
there is no sudo/doas executable. `/usr/lib/cgi-bin` does not exist. Existing
CGI credentials were not read. Direct root SSH access was also verified with
`id`; it is available when deployment is authorized.
