#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/diff_diagnostics_against_js.sh <source.ps>" >&2
  exit 1
fi

SOURCE_FILE="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

JS_OUT="$TMP_DIR/js-parser-diagnostics.ndjson"
CPP_OUT="$TMP_DIR/cpp-parser-diagnostics.ndjson"

node "$ROOT_DIR/src/tests/js_oracle/export_ir_json.js" "$SOURCE_FILE" "$JS_OUT" --snapshot-phase parser-diagnostics
"$ROOT_DIR/build/native/puzzlescript_cpp" compile "$SOURCE_FILE" --diagnostics > "$CPP_OUT"

node "$ROOT_DIR/scripts/compare_parser_phase_diagnostics.js" "$JS_OUT" "$CPP_OUT"
