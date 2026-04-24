# Compact C++ workflow:
#   make build             Build build/native/puzzlescript_cpp.
#   make run game.txt      Build and play a PuzzleScript source file.
#   make ctest             Run fast C++ smoke/unit tests registered with CMake.
#   make js_parity_tests   Run C++ against the original JS test corpus.
#   make rule_plan_parity_tests
#                           Compare JS/native game.rule_plan_v1 for simulation-corpus games.
#   make simulation_tests  Run JS simulation tests and direct C++ simulation tests.
#   make compilation_tests Run JS compiler tests and direct C++ compiler tests.
#   make profile_simulation_tests
#                           Profile the C++ simulation replay workload.
#   make simulation_tests_cpp_32
#                           Run direct C++ simulation tests with JS-style 32-bit mask words.
#   make tests             Run the full native correctness suite.

.DEFAULT_GOAL := help

.PHONY: help build build_32 build_solver run ctest tests js_parity_tests tests_js simulation_tests_js simulation_tests_js_profile simulation_tests_js_profile_breakdown compilation_tests_js \
	simulation_tests_cpp compilation_tests_cpp simulation_tests compilation_tests \
	simulation_tests_cpp_32 compilation_tests_cpp_32 \
	solver_tests_cpp solver_tests_js solver_tests \
	simulation_tests_cpp_js_parity compilation_tests_cpp_direct \
	rule_plan_parity_tests \
	profile_simulation_tests profile_simulation_tests_32 basic_test_suite_cpp basic_test_suite_js \
	parser_corpus_errormessage_bundle parser_corpus_testdata_bundle clean clean-native \
	clean-native-32 clean-js-parity-data configure-native build-native js-parity-data

NODE ?= node
CMAKE ?= cmake
BUILD_DIR ?= build
BUILD_DIR_32 ?= build-32
PUZZLESCRIPT_CPP := $(BUILD_DIR)/native/puzzlescript_cpp
PUZZLESCRIPT_CPP_32 := $(BUILD_DIR_32)/native/puzzlescript_cpp
PUZZLESCRIPT_SOLVER := $(BUILD_DIR)/native/puzzlescript_solver
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
	@echo "  make build                         Build build/native/puzzlescript_cpp (64-bit masks)"
	@echo "  make build_solver                  Build build/native/puzzlescript_solver"
	@echo "  make build_32                      Build JS-style 32-bit-mask executable into build-32"
	@echo "  make run path/to/game.txt          Build and play a PuzzleScript game"
	@echo "  make ctest                         Run fast C++ smoke/unit tests"
	@echo "  make js_parity_tests               Run 32-bit C++ against the original JS test corpus"
	@echo "  make rule_plan_parity_tests        Compare JS/native game.rule_plan_v1 for simulation games"
	@echo "  make simulation_tests              Run JS sim tests, then mirrored C++ sim parity"
	@echo "  make compilation_tests             Run JS compiler tests, then mirrored C++ diagnostics"
	@echo "  make profile_simulation_tests      Profile C++ simulation replay hot functions"
	@echo "  make profile_simulation_tests_32   Profile the 32-bit-mask C++ simulation path"
	@echo "  make tests                         Run the full native correctness suite"
	@echo "  make solver_tests                  Run native solver and JS comparison solver"
	@echo "  make clean                         Remove native build outputs and JS parity data"
	@echo ""
	@echo "Single-side test commands for timing:"
	@echo "  make simulation_tests_js           Run JS simulation tests only"
	@echo "  make simulation_tests_js_profile   Run JS simulation tests 5 times and report avg/median"
	@echo "  make simulation_tests_js_profile_breakdown"
	@echo "                                     Run JS profile with compile/input timing averages"
	@echo "  make simulation_tests_cpp          Run C++ simulation corpus directly (64-bit masks)"
	@echo "  make simulation_tests_cpp_32       Run C++ simulation corpus with JS-style 32-bit masks"
	@echo "  make compilation_tests_js          Run JS compiler tests only"
	@echo "  make compilation_tests_cpp         Run C++ diagnostics corpus directly (64-bit masks)"
	@echo "  make compilation_tests_cpp_32      Run C++ diagnostics corpus with JS-style 32-bit masks"
	@echo "  make tests_js                      Run the original JavaScript test suite"
	@echo "  make solver_tests_cpp              Run standalone native solver corpus"
	@echo "  make solver_tests_js               Run JavaScript comparison solver corpus"
	@echo ""
	@echo "Direct executable after build:"
	@echo "  build/native/puzzlescript_cpp --help"
	@echo "  build/native/puzzlescript_solver src/tests/solver_tests --timeout-ms 5000"

$(CMAKE_CACHE): CMakeLists.txt native/CMakeLists.txt
	$(CMAKE) -S . -B $(BUILD_DIR) -DPS_MASK_WORD_BITS=64

$(PUZZLESCRIPT_CPP): $(CMAKE_CACHE) native/CMakeLists.txt
	$(CMAKE) -S . -B $(BUILD_DIR) -DPS_MASK_WORD_BITS=64
	$(CMAKE) --build $(BUILD_DIR) --target puzzlescript_cpp

$(PUZZLESCRIPT_SOLVER): $(CMAKE_CACHE) native/CMakeLists.txt native/src/solver/main.cpp
	$(CMAKE) -S . -B $(BUILD_DIR) -DPS_MASK_WORD_BITS=64
	$(CMAKE) --build $(BUILD_DIR) --target puzzlescript_solver

build: $(CMAKE_CACHE)
	$(CMAKE) -S . -B $(BUILD_DIR) -DPS_MASK_WORD_BITS=64
	$(CMAKE) --build $(BUILD_DIR) --target puzzlescript_cpp

build_solver: $(PUZZLESCRIPT_SOLVER)

build_32:
	$(CMAKE) -S . -B $(BUILD_DIR_32) -DPS_MASK_WORD_BITS=32
	$(CMAKE) --build $(BUILD_DIR_32) --target puzzlescript_cpp

configure-native: $(CMAKE_CACHE)

build-native: build

ctest: build build_solver
	ctest --test-dir $(BUILD_DIR) --output-on-failure

tests_js:
	$(NODE) src/tests/run_tests_node.js

simulation_tests_js:
	$(NODE) src/tests/run_tests_node.js --sim-only

simulation_tests_js_profile:
	$(NODE) src/tests/run_tests_node.js --profile --profile-runs 5 --sim-only

simulation_tests_js_profile_breakdown:
	$(NODE) src/tests/run_tests_node.js --profile --profile-runs 5 --sim-only --breakdown

compilation_tests_js:
	$(NODE) src/tests/run_tests_node.js --compilation-only

solver_tests_cpp: $(PUZZLESCRIPT_SOLVER)
	$(PUZZLESCRIPT_SOLVER) src/tests/solver_tests --timeout-ms 5000

solver_tests_js:
	$(NODE) src/tests/run_solver_tests_js.js src/tests/solver_tests --timeout-ms 5000

solver_tests: solver_tests_cpp solver_tests_js

$(JS_PARITY_MANIFEST): $(JS_PARITY_INPUTS)
	$(NODE) src/tests/js_oracle/export_native_fixtures.js $(JS_PARITY_DATA_DIR)

js-parity-data: $(JS_PARITY_MANIFEST)

simulation_tests_cpp: build
	$(PUZZLESCRIPT_CPP) test simulation-corpus src/tests/resources/testdata.js --jobs auto --progress-every 0

simulation_tests_cpp_32: build_32
	$(PUZZLESCRIPT_CPP_32) test simulation-corpus src/tests/resources/testdata.js --jobs auto --progress-every 0

simulation_tests_cpp_js_parity: build_32 $(JS_PARITY_MANIFEST)
	$(NODE) src/tests/run_native_trace_suite.js $(JS_PARITY_MANIFEST) --cli $(PUZZLESCRIPT_CPP_32) --progress-every 1 --timeout-ms 45000

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

compilation_tests_cpp_32: build_32
	$(PUZZLESCRIPT_CPP_32) test diagnostics-corpus src/tests/resources/errormessage_testdata.js --progress-every 50

compilation_tests_cpp_direct: build
	$(PUZZLESCRIPT_CPP) test diagnostics-corpus src/tests/resources/errormessage_testdata.js --progress-every 50

js_parity_tests: simulation_tests_cpp_js_parity compilation_tests_cpp_32

rule_plan_parity_tests: build
	$(NODE) src/tests/run_rule_plan_parity.js src/tests/resources/testdata.js --cli $(PUZZLESCRIPT_CPP) --artifacts-dir $(BUILD_DIR)/native/rule_plan_parity_testdata

simulation_tests: simulation_tests_js simulation_tests_cpp

compilation_tests: compilation_tests_js compilation_tests_cpp

profile_simulation_tests: build
	src/tests/profile_native_trace_suite.sh

profile_simulation_tests_32: build_32
	PUZZLESCRIPT_CPP="$(abspath $(PUZZLESCRIPT_CPP_32))" \
	PROFILE_STATS_OUT="$(abspath $(BUILD_DIR_32))/profile_stats.txt" \
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
	rm -rf "$(BUILD_DIR_32)"

clean-native-32:
	rm -rf "$(BUILD_DIR_32)"

clean-js-parity-data:
	rm -rf "$(JS_PARITY_DATA_DIR)"
