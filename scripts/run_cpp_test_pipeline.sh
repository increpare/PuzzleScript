#!/usr/bin/env bash
# Run the native/C++ checks used for JS parity: ctest + full parser-state and
# parser-diagnostics corpus diffs (testdata.js and errormessage_testdata.js).
#
# Prerequisites: repo-root CMake build with `ps_cli` (e.g. `make build-native`
# or `cmake --build build --target ps_cli`; default BUILD_DIR matches the Makefile).
#
# Usage:
#   scripts/run_cpp_test_pipeline.sh
#   BUILD_DIR=build/other bash scripts/run_cpp_test_pipeline.sh
#   PS_CLI=/path/to/ps_cli bash scripts/run_cpp_test_pipeline.sh
#
# Optional skips (set to 1):
#   SKIP_CTEST, SKIP_PARSER_STATE_CORPUS, SKIP_DIAGNOSTICS_CORPUS

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="${BUILD_DIR:-build}"
export BUILD_DIR

resolve_ps_cli() {
	if [[ -n "${PS_CLI:-}" && -x "$PS_CLI" ]]; then
		echo "$PS_CLI"
		return
	fi
	local cand
	for cand in \
		"$ROOT/$BUILD_DIR/ps_cli" \
		"$ROOT/$BUILD_DIR/native/ps_cli" \
		"$ROOT/$BUILD_DIR/native/native/ps_cli"; do
		if [[ -x "$cand" ]]; then
			echo "$cand"
			return
		fi
	done
	echo "error: no executable ps_cli found (tried \$PS_CLI, $ROOT/$BUILD_DIR/ps_cli, $ROOT/$BUILD_DIR/native/ps_cli, $ROOT/$BUILD_DIR/native/native/ps_cli)" >&2
	echo "Build first, e.g.: cmake -S . -B $BUILD_DIR && cmake --build $BUILD_DIR --target ps_cli" >&2
	exit 1
}

PS_CLI="$(resolve_ps_cli)"
export PS_CLI

if [[ "${SKIP_CTEST:-0}" != "1" ]]; then
	echo "== ctest --test-dir $BUILD_DIR =="
	ctest --test-dir "$BUILD_DIR" --output-on-failure
else
	echo "== ctest (skipped, SKIP_CTEST=1) =="
fi

if [[ "${SKIP_PARSER_STATE_CORPUS:-0}" != "1" ]]; then
	echo "== parser state corpus (testdata.js vs export_ir_json parser snapshot) =="
	node scripts/diff_parser_state_corpus.js --cli "$PS_CLI"
else
	echo "== parser state corpus (skipped) =="
fi

if [[ "${SKIP_DIAGNOSTICS_CORPUS:-0}" != "1" ]]; then
	echo "== parser diagnostics corpus (testdata.js) =="
	node scripts/diff_diagnostics_corpus.js --cli "$PS_CLI" --corpus testdata
	echo "== parser diagnostics corpus (errormessage_testdata.js) =="
	node scripts/diff_diagnostics_corpus.js --cli "$PS_CLI" --corpus errormessage
else
	echo "== parser diagnostics corpus (skipped) =="
fi

echo "== cpp test pipeline finished OK =="
