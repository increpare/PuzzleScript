#!/usr/bin/env bash
# Run the native/C++ checks used for JS parity: ctest + full parser-state and
# parser-diagnostics corpus diffs. Parser-state: `diff_parser_state_corpus.js`
# defaults to both testdata.js and errormessage_testdata.js (`--corpus all`).
#
# Prerequisites: repo-root CMake build with `puzzlescript_cpp` (e.g. `make build-native`
# or `cmake --build build --target puzzlescript_cpp`; default BUILD_DIR matches the Makefile).
#
# Usage:
#   scripts/run_cpp_test_pipeline.sh
#   BUILD_DIR=build/other bash scripts/run_cpp_test_pipeline.sh
#   PUZZLESCRIPT_CPP=/path/to/puzzlescript_cpp bash scripts/run_cpp_test_pipeline.sh
#
# Optional skips (set to 1):
#   SKIP_CTEST, SKIP_PARSER_STATE_CORPUS, SKIP_DIAGNOSTICS_CORPUS

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="${BUILD_DIR:-build}"
export BUILD_DIR

resolve_puzzlescript_cpp() {
	if [[ -n "${PUZZLESCRIPT_CPP:-}" && -x "$PUZZLESCRIPT_CPP" ]]; then
		echo "$PUZZLESCRIPT_CPP"
		return
	fi
	local cand
	for cand in \
		"$ROOT/$BUILD_DIR/puzzlescript_cpp" \
		"$ROOT/$BUILD_DIR/native/puzzlescript_cpp" \
		"$ROOT/$BUILD_DIR/native/native/puzzlescript_cpp"; do
		if [[ -x "$cand" ]]; then
			echo "$cand"
			return
		fi
	done
	echo "error: no executable puzzlescript_cpp found (tried \$PUZZLESCRIPT_CPP, $ROOT/$BUILD_DIR/puzzlescript_cpp, $ROOT/$BUILD_DIR/native/puzzlescript_cpp, $ROOT/$BUILD_DIR/native/native/puzzlescript_cpp)" >&2
	echo "Build first, e.g.: cmake -S . -B $BUILD_DIR && cmake --build $BUILD_DIR --target puzzlescript_cpp" >&2
	exit 1
}

PUZZLESCRIPT_CPP="$(resolve_puzzlescript_cpp)"
export PUZZLESCRIPT_CPP

if [[ "${SKIP_CTEST:-0}" != "1" ]]; then
	echo "== ctest --test-dir $BUILD_DIR =="
	ctest --test-dir "$BUILD_DIR" --output-on-failure
else
	echo "== ctest (skipped, SKIP_CTEST=1) =="
fi

if [[ "${SKIP_PARSER_STATE_CORPUS:-0}" != "1" ]]; then
	echo "== parser state corpus (testdata + errormessage vs export_ir_json parser snapshot) =="
	node scripts/diff_parser_state_corpus.js --cli "$PUZZLESCRIPT_CPP"
else
	echo "== parser state corpus (skipped) =="
fi

if [[ "${SKIP_DIAGNOSTICS_CORPUS:-0}" != "1" ]]; then
	echo "== parser diagnostics corpus (testdata.js, puzzlescript_cpp diagnostics-parity) =="
	mkdir -p "$ROOT/$BUILD_DIR"
	node "$ROOT/scripts/build_parser_corpus_bundle.js" testdata >"$ROOT/$BUILD_DIR/parser_corpus_testdata.bundle.ndjson"
	"$PUZZLESCRIPT_CPP" diagnostics-parity "$ROOT/$BUILD_DIR/parser_corpus_testdata.bundle.ndjson"
	echo "== parser diagnostics corpus (errormessage_testdata.js, puzzlescript_cpp diagnostics-parity) =="
	node "$ROOT/scripts/build_parser_corpus_bundle.js" errormessage >"$ROOT/$BUILD_DIR/parser_corpus_errormessage.bundle.ndjson"
	"$PUZZLESCRIPT_CPP" diagnostics-parity "$ROOT/$BUILD_DIR/parser_corpus_errormessage.bundle.ndjson"
else
	echo "== parser diagnostics corpus (skipped) =="
fi

echo "== cpp test pipeline finished OK =="
