#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/diff_parser_state_against_js.sh <source.txt>" >&2
  exit 1
fi

SOURCE_FILE="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

JS_OUT="$TMP_DIR/js-parser-state.json"
CPP_OUT="$TMP_DIR/cpp-parser-state.json"

node "$ROOT_DIR/src/tests/js_oracle/export_ir_json.js" "$SOURCE_FILE" "$JS_OUT" --snapshot-phase parser
"$ROOT_DIR/build/native/puzzlescript_cpp" compile "$SOURCE_FILE" --emit-parser-state > "$CPP_OUT"

diff -u "$JS_OUT" "$CPP_OUT"
