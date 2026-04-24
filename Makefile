# Compact C++ workflow:
#   make build             Build build/native/puzzlescript_cpp.
#   make run game.txt      Build and play a PuzzleScript source file.
#   make ctest             Run fast C++ smoke/unit tests registered with CMake.
#   make js_parity_tests   Run C++ against the original JS test corpus.
#   make simulation_tests  Run JS simulation tests and direct C++ simulation tests.
#   make compilation_tests Run JS compiler tests and direct C++ compiler tests.
#   make profile_simulation_tests
#                           Profile the C++ simulation replay workload.
#   make tests             Run the full native correctness suite.

.DEFAULT_GOAL := help

.PHONY: help build run ctest tests js_parity_tests tests_js simulation_tests_js compilation_tests_js \
	simulation_tests_cpp compilation_tests_cpp simulation_tests compilation_tests \
	simulation_tests_cpp_js_parity compilation_tests_cpp_direct \
	profile_simulation_tests basic_test_suite_cpp basic_test_suite_js \
	parser_corpus_errormessage_bundle parser_corpus_testdata_bundle clean clean-native \
	clean-js-parity-data configure-native build-native js-parity-data

NODE ?= node
CMAKE ?= cmake
BUILD_DIR ?= build
PUZZLESCRIPT_CPP := $(BUILD_DIR)/native/puzzlescript_cpp
JS_PARITY_DATA_DIR := $(BUILD_DIR)/js-parity-data
JS_PARITY_MANIFEST := $(JS_PARITY_DATA_DIR)/fixtures.json
ERRORMESSAGE_PARSER_BUNDLE := $(BUILD_DIR)/parser_corpus_errormessage.bundle.ndjson
TESTDATA_PARSER_BUNDLE := $(BUILD_DIR)/parser_corpus_testdata.bundle.ndjson

PARSER_CORPUS_BUNDLE_INPUTS := \
	scripts/build_parser_corpus_bundle.js \
	src/tests/resources/errormessage_testdata.js \
	src/tests/resources/testdata.js \
	src/tests/js_oracle/lib/puzzlescript_parser_snapshot.js \
	src/tests/js_oracle/lib/puzzlescript_node_env.js

JS_PARITY_INPUTS := \
	src/tests/js_oracle/export_native_fixtures.js \
	src/tests/run_native_trace_suite.js \
	$(wildcard src/tests/js_oracle/lib/*.js) \
	$(wildcard src/tests/js_oracle/lib/*/*.js) \
	$(wildcard src/tests/js_oracle/lib/*/*/*.js) \
	$(wildcard src/js/*.js) \
	$(wildcard src/js/*/*.js) \
	$(wildcard src/js/*/*/*.js)

CMAKE_CACHE := $(BUILD_DIR)/CMakeCache.txt

help:
	@echo "PuzzleScript C++ workflow"
	@echo ""
	@echo "Common commands:"
	@echo "  make build                         Build build/native/puzzlescript_cpp"
	@echo "  make run path/to/game.txt          Build and play a PuzzleScript game"
	@echo "  make ctest                         Run fast C++ smoke/unit tests"
	@echo "  make js_parity_tests               Run C++ against the original JS test corpus"
	@echo "  make simulation_tests              Run JS sim tests, then mirrored C++ sim parity"
	@echo "  make compilation_tests             Run JS compiler tests, then mirrored C++ diagnostics"
	@echo "  make profile_simulation_tests      Profile C++ simulation replay hot functions"
	@echo "  make tests                         Run the full native correctness suite"
	@echo "  make clean                         Remove native build outputs and JS parity data"
	@echo ""
	@echo "Single-side test commands for timing:"
	@echo "  make simulation_tests_js           Run JS simulation tests only"
	@echo "  make simulation_tests_cpp          Run C++ simulation corpus directly"
	@echo "  make compilation_tests_js          Run JS compiler tests only"
	@echo "  make compilation_tests_cpp         Run C++ diagnostics corpus directly"
	@echo "  make tests_js                      Run the original JavaScript test suite"
	@echo ""
	@echo "Direct executable after build:"
	@echo "  build/native/puzzlescript_cpp --help"

$(CMAKE_CACHE): CMakeLists.txt native/CMakeLists.txt
	$(CMAKE) -S . -B $(BUILD_DIR)

$(PUZZLESCRIPT_CPP): $(CMAKE_CACHE) native/CMakeLists.txt
	$(CMAKE) -S . -B $(BUILD_DIR)
	$(CMAKE) --build $(BUILD_DIR) --target puzzlescript_cpp

build: $(CMAKE_CACHE)
	$(CMAKE) -S . -B $(BUILD_DIR)
	$(CMAKE) --build $(BUILD_DIR) --target puzzlescript_cpp

configure-native: $(CMAKE_CACHE)

build-native: build

ctest: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

tests_js:
	$(NODE) src/tests/run_tests_node.js

simulation_tests_js:
	$(NODE) src/tests/run_tests_node.js --sim-only

compilation_tests_js:
	$(NODE) src/tests/run_tests_node.js --compilation-only

$(JS_PARITY_MANIFEST): $(JS_PARITY_INPUTS)
	$(NODE) src/tests/js_oracle/export_native_fixtures.js $(JS_PARITY_DATA_DIR)

js-parity-data: $(JS_PARITY_MANIFEST)

simulation_tests_cpp: build
	$(PUZZLESCRIPT_CPP) test simulation-corpus src/tests/resources/testdata.js --jobs auto --progress-every 0

simulation_tests_cpp_js_parity: build $(JS_PARITY_MANIFEST)
	$(NODE) src/tests/run_native_trace_suite.js $(JS_PARITY_MANIFEST) --cli $(PUZZLESCRIPT_CPP) --progress-every 1 --timeout-ms 45000

$(ERRORMESSAGE_PARSER_BUNDLE): $(PARSER_CORPUS_BUNDLE_INPUTS) $(JS_PARITY_INPUTS)
	mkdir -p "$(BUILD_DIR)"
	$(NODE) scripts/build_parser_corpus_bundle.js errormessage > "$(ERRORMESSAGE_PARSER_BUNDLE)"

$(TESTDATA_PARSER_BUNDLE): $(PARSER_CORPUS_BUNDLE_INPUTS) $(JS_PARITY_INPUTS)
	mkdir -p "$(BUILD_DIR)"
	$(NODE) scripts/build_parser_corpus_bundle.js testdata > "$(TESTDATA_PARSER_BUNDLE)"

parser_corpus_errormessage_bundle: $(ERRORMESSAGE_PARSER_BUNDLE)

parser_corpus_testdata_bundle: $(TESTDATA_PARSER_BUNDLE)

compilation_tests_cpp: build
	$(PUZZLESCRIPT_CPP) test diagnostics-corpus src/tests/resources/errormessage_testdata.js --progress-every 50

compilation_tests_cpp_direct: build
	$(PUZZLESCRIPT_CPP) test diagnostics-corpus src/tests/resources/errormessage_testdata.js --progress-every 50

js_parity_tests: simulation_tests_cpp_js_parity compilation_tests_cpp

simulation_tests: simulation_tests_js simulation_tests_cpp

compilation_tests: compilation_tests_js compilation_tests_cpp

profile_simulation_tests: build
	src/tests/profile_native_trace_suite.sh

tests: ctest js_parity_tests

basic_test_suite_cpp: js_parity_tests

basic_test_suite_js: tests_js

ifneq ($(filter run,$(MAKECMDGOALS)),)
RUN_SOURCE_FILE := $(word 2,$(MAKECMDGOALS))
ifneq ($(strip $(RUN_SOURCE_FILE)),)
$(eval .PHONY: $(RUN_SOURCE_FILE))
$(eval $(RUN_SOURCE_FILE):;@:)
endif
endif

run: build
ifndef RUN_SOURCE_FILE
	@echo "Usage: make run path/to/game.txt"
	@exit 1
endif
	$(PUZZLESCRIPT_CPP) play $(RUN_SOURCE_FILE)

clean: clean-native clean-js-parity-data

clean-native:
	rm -rf "$(BUILD_DIR)"

clean-js-parity-data:
	rm -rf "$(JS_PARITY_DATA_DIR)"
