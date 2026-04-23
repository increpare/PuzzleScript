# Test entry points:
#   make tests_js  — JS harness: node src/tests/run_tests_node.js (testdata + errormessage fixtures).
#   make simulation_tests_js    — JS simulation only (testdata.js via run_tests_node.js --sim-only).
#   make compilation_tests_js   — JS compile/error fixtures only (errormessage_testdata.js via --compilation-only).
#   make tests_cpp — Native CTest suite (builds ps_cli first if needed).
#   make simulation_tests_cpp   — Native trace replay (same as make tests): export fixtures, then
#                                  run_native_trace_suite.js + ps_cli check-trace-sweep.
#   make compilation_tests_cpp  — Parser-diagnostics parity: rebuild a corpus bundle with Node once
#                                  (scripts/build_parser_corpus_bundle.js), then ps_cli diagnostics-parity
#                                  (pure C++ per fixture, no subprocess farm). Legacy per-row harness:
#                                  node scripts/diff_diagnostics_corpus.js --cli … --limit 5
#   make tests     — Alias for simulation_tests_cpp.
#   make cpp-test-pipeline — make tests_cpp, then parser-state + diagnostics corpus diffs vs JS.
.PHONY: tests tests_js tests_cpp simulation_tests_js simulation_tests_cpp \
	compilation_tests_js compilation_tests_cpp parser_corpus_errormessage_bundle \
	parser_corpus_testdata_bundle run cpp-test-pipeline \
	clean clean-native clean-fixtures configure-native build-native coverage-fixtures

NODE ?= node
CMAKE ?= cmake
# CMake binary dir (cmake -S . -B $(BUILD_DIR)); ps_cli is emitted under $(BUILD_DIR)/native/ps_cli.
# Do not default to build/native as the CMake root — that nests native twice and points PS_CLI at a stale path.
BUILD_DIR ?= build
PS_CLI := $(BUILD_DIR)/native/ps_cli
COVERAGE_FIXTURES_DIR := $(BUILD_DIR)/coverage-fixtures
COVERAGE_FIXTURES_MANIFEST := $(COVERAGE_FIXTURES_DIR)/fixtures.json
ERRORMESSAGE_PARSER_BUNDLE := $(BUILD_DIR)/parser_corpus_errormessage.bundle.ndjson
TESTDATA_PARSER_BUNDLE := $(BUILD_DIR)/parser_corpus_testdata.bundle.ndjson

PARSER_CORPUS_BUNDLE_INPUTS := \
	scripts/build_parser_corpus_bundle.js \
	src/tests/resources/errormessage_testdata.js \
	src/tests/resources/testdata.js \
	src/tests/lib/puzzlescript_parser_snapshot.js \
	src/tests/lib/puzzlescript_node_env.js

CMAKE_CACHE := $(BUILD_DIR)/CMakeCache.txt

tests_js:
	$(NODE) src/tests/run_tests_node.js

simulation_tests_js:
	$(NODE) src/tests/run_tests_node.js --sim-only

compilation_tests_js:
	$(NODE) src/tests/run_tests_node.js --compilation-only

# Note: make can't automatically infer Node's require() dependency graph.
# These are intentionally coarse so C++-only iterations don't keep regenerating fixtures.
JS_FIXTURE_INPUTS := \
	src/tests/export_native_fixtures.js \
	src/tests/run_native_trace_suite.js \
	$(wildcard src/tests/lib/*.js) \
	$(wildcard src/tests/lib/*/*.js) \
	$(wildcard src/tests/lib/*/*/*.js) \
	$(wildcard src/js/*.js) \
	$(wildcard src/js/*/*.js) \
	$(wildcard src/js/*/*/*.js)

$(CMAKE_CACHE):
	$(CMAKE) -S . -B $(BUILD_DIR)

$(PS_CLI): $(CMAKE_CACHE)
	$(CMAKE) --build $(BUILD_DIR)

# Native unit/smoke tests (everything registered via CMake add_test).
tests_cpp: $(PS_CLI)
	ctest --test-dir $(BUILD_DIR) --output-on-failure

# Native smoke + fixture checks (ctest), then full parser-state / diagnostics corpus vs JS.
# Expect several minutes for the corpus steps. Override: PS_CLI=... BUILD_DIR=...
cpp-test-pipeline: build-native
	bash scripts/run_cpp_test_pipeline.sh

$(COVERAGE_FIXTURES_MANIFEST): $(JS_FIXTURE_INPUTS)
	$(NODE) src/tests/export_native_fixtures.js $(COVERAGE_FIXTURES_DIR)

simulation_tests_cpp: build-native $(COVERAGE_FIXTURES_MANIFEST)
	$(NODE) src/tests/run_native_trace_suite.js $(COVERAGE_FIXTURES_MANIFEST) --cli $(PS_CLI) --progress-every 1 --timeout-ms 45000

$(ERRORMESSAGE_PARSER_BUNDLE): $(PARSER_CORPUS_BUNDLE_INPUTS) $(JS_FIXTURE_INPUTS)
	mkdir -p "$(BUILD_DIR)"
	$(NODE) scripts/build_parser_corpus_bundle.js errormessage > "$(ERRORMESSAGE_PARSER_BUNDLE)"

$(TESTDATA_PARSER_BUNDLE): $(PARSER_CORPUS_BUNDLE_INPUTS) $(JS_FIXTURE_INPUTS)
	mkdir -p "$(BUILD_DIR)"
	$(NODE) scripts/build_parser_corpus_bundle.js testdata > "$(TESTDATA_PARSER_BUNDLE)"

parser_corpus_errormessage_bundle: $(ERRORMESSAGE_PARSER_BUNDLE)

parser_corpus_testdata_bundle: $(TESTDATA_PARSER_BUNDLE)

compilation_tests_cpp: build-native $(ERRORMESSAGE_PARSER_BUNDLE)
	$(PS_CLI) diagnostics-parity "$(ERRORMESSAGE_PARSER_BUNDLE)"

tests: simulation_tests_cpp

# Backwards-compatible aliases.
configure-native: $(CMAKE_CACHE)
build-native: $(CMAKE_CACHE)
	$(CMAKE) --build $(BUILD_DIR)
coverage-fixtures: $(COVERAGE_FIXTURES_MANIFEST)

ifneq ($(filter run,$(MAKECMDGOALS)),)
RUN_SOURCE_FILE := $(word 2,$(MAKECMDGOALS))
ifneq ($(strip $(RUN_SOURCE_FILE)),)
$(eval .PHONY: $(RUN_SOURCE_FILE))
$(eval $(RUN_SOURCE_FILE):;@:)
endif
endif

run: build-native
ifndef RUN_SOURCE_FILE
	@echo "Usage: make run path/to/game.txt"
	@exit 1
endif
	$(PS_CLI) play-source $(RUN_SOURCE_FILE)

clean: clean-native clean-fixtures

clean-native:
	rm -rf "$(BUILD_DIR)"

clean-fixtures:
	rm -rf "$(COVERAGE_FIXTURES_DIR)"
