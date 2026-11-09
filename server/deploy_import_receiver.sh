#!/usr/bin/env bash
set -euo pipefail

# Update only the POST /editor.html receiver. The static website is separate.
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
receiver="$repo_dir/server/import_source.py"
remote=root@ded.increpare.com
endpoint=https://www.puzzlescript.net/editor.html
remote_tmp=
headers_file=
body_file=

cleanup() {
    if [[ -n "$remote_tmp" ]]; then
        ssh -o BatchMode=yes "$remote" "rm -f -- '$remote_tmp'" >/dev/null 2>&1 || true
    fi
    if [[ -n "$headers_file" ]]; then rm -f -- "$headers_file"; fi
    if [[ -n "$body_file" ]]; then rm -f -- "$body_file"; fi
}
trap cleanup EXIT

python3 -B -m unittest discover -s "$repo_dir/server" -p test_import_source.py
expected_sha=$(shasum -a 256 "$receiver" | awk '{print $1}')

# Fail if the deployed route or service is not the one this script expects.
ssh -o BatchMode=yes "$remote" \
    'test -f /srv/cgi/import_source.py && systemctl is-active --quiet cgi-bin.service && grep -Fq "@editor_import" /etc/caddy/Caddyfile'

remote_tmp=$(ssh -o BatchMode=yes "$remote" 'mktemp /tmp/puzzlescript-import.XXXXXXXX')
if [[ ! "$remote_tmp" =~ ^/tmp/puzzlescript-import\.[A-Za-z0-9]{8}$ ]]; then
    echo "Unexpected remote temporary path; deployment stopped." >&2
    exit 1
fi
scp -o BatchMode=yes -q "$receiver" "$remote:$remote_tmp"

backup=$(ssh -o BatchMode=yes "$remote" "bash -s -- '$remote_tmp' '$expected_sha'" <<'REMOTE'
set -euo pipefail
staged=$1
expected_sha=$2
target=/srv/cgi/import_source.py

if [[ "$(sha256sum "$staged" | awk '{print $1}')" != "$expected_sha" ]]; then
    echo 'Transferred receiver does not match the local file.' >&2
    exit 1
fi
python3 -m py_compile "$staged"

backup=$(mktemp /srv/cgi/import_source.py.backup.XXXXXXXX)
cp -a -- "$target" "$backup"
restore() {
    cp -a -- "$backup" "$target"
    systemctl restart cgi-bin.service
}

if ! install -o root -g root -m 644 "$staged" "$target" ||
   ! python3 -m py_compile "$target" ||
   [[ "$(sha256sum "$target" | awk '{print $1}')" != "$expected_sha" ]] ||
   ! systemctl restart cgi-bin.service ||
   ! systemctl is-active --quiet cgi-bin.service; then
    echo 'Install or service check failed; restoring the previous receiver.' >&2
    restore
    exit 1
fi
printf '%s\n' "$backup"
REMOTE
)
printf 'Receiver installed; server backup: %s\n' "$backup"

headers_file=$(mktemp /tmp/puzzlescript-import-headers.XXXXXXXX)
body_file=$(mktemp /tmp/puzzlescript-import-body.XXXXXXXX)
status=$(curl --silent --show-error --max-time 20 --dump-header "$headers_file" \
    --output "$body_file" --write-out '%{http_code}' \
    --data-urlencode 'source=title Deploy probe' "$endpoint")
if [[ "$status" != 200 ]] ||
   grep -qi '^content-security-policy:' "$headers_file" ||
   ! grep -Fq '"title Deploy probe"' "$body_file"; then
    echo 'Public POST verification failed. The server backup path is above.' >&2
    exit 1
fi
echo 'Public POST verified: source imports and no import-only CSP header is present.'
