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

.PHONY: help build build_32 build_solver build_generator generator solver run ctest tests js_parity_tests tests_js simulation_tests_js simulation_tests_js_profile simulation_tests_js_profile_breakdown compilation_tests_js \
	simulation_tests_cpp compilation_tests_cpp simulation_tests compilation_tests \
	simulation_tests_cpp_32 compilation_tests_cpp_32 \
	solver_tests_cpp solver_tests_js solver_tests solver_smoke_tests solver_determinism_tests solver_parity_smoke solver_benchmark solver_mine_pippable solver_benchmark_targets generator_smoke_tests generator_benchmark \
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
PUZZLESCRIPT_GENERATOR := $(BUILD_DIR)/native/puzzlescript_generator
GENERATOR_MAKE_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
GENERATOR_GAME := $(word 1,$(GENERATOR_MAKE_ARGS))
GENERATOR_SPEC := $(word 2,$(GENERATOR_MAKE_ARGS))
GENERATOR_ARGS ?=
SOLVER_MAKE_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
SOLVER_GAME := $(word 1,$(SOLVER_MAKE_ARGS))
SOLVER_ARGS ?=
SOLVER_TIMEOUT_MS ?= 250
SOLVER_TESTS_CORPUS ?= src/tests/solver_tests
SOLVER_JOBS ?= 1
SOLVER_STRATEGY ?= portfolio
SOLVER_PROGRESS_EVERY ?= game
SOLVER_OUTPUT_ARGS ?= --summary-only
SOLVER_SOLUTIONS_DIR ?= $(BUILD_DIR)/solver-solutions
SOLVER_BENCH_RUNS ?= 5
SOLVER_BENCH_TIMEOUT_MS ?= 250
SOLVER_BENCH_CORPUS ?= src/tests/solver_tests
SOLVER_BENCH_OUT ?= $(BUILD_DIR)/native/solver_benchmark.json
SOLVER_PERF_BASELINE ?= solver_perf_baseline.json
SOLVER_BENCH_JOBS ?= 1
SOLVER_BENCH_STRATEGY ?= portfolio
SOLVER_MINE_CORPUS ?= src/tests/solver_tests
SOLVER_MINE_TIMEOUTS_MS ?= 50,100,250,500
SOLVER_MINE_STRATEGY ?= portfolio
SOLVER_MINE_NEAR_RATIO ?= 0.5
SOLVER_MINE_MAX_TARGETS ?=
SOLVER_PIPPABLE_MANIFEST ?= $(BUILD_DIR)/native/solver_pippable_targets.json
SOLVER_TARGET_BENCH_RUNS ?= 5
SOLVER_TARGET_BENCH_CORPUS ?= $(SOLVER_MINE_CORPUS)
SOLVER_TARGET_BENCH_MANIFEST ?= $(SOLVER_PIPPABLE_MANIFEST)
SOLVER_TARGET_BENCH_OUT ?= $(BUILD_DIR)/native/solver_target_benchmark.json
SOLVER_TARGET_BENCH_TIMEOUT_MS ?=
SOLVER_TARGET_BENCH_STRATEGY ?= $(SOLVER_MINE_STRATEGY)
GENERATOR_BENCH_GAME ?= src/demo/sokoban_basic.txt
GENERATOR_BENCH_PRESETS_DIR ?= src/tests/generator_presets
GENERATOR_BENCH_SAMPLES ?= 200
GENERATOR_BENCH_RUNS ?= 3
GENERATOR_BENCH_JOBS ?= 1
GENERATOR_BENCH_SEED ?= 11
GENERATOR_BENCH_SOLVER_TIMEOUT_MS ?= 50
GENERATOR_BENCH_SOLVER_STRATEGY ?= portfolio
GENERATOR_BENCH_TOP_K ?= 10
GENERATOR_BENCH_OUT ?= $(BUILD_DIR)/native/generator_benchmark.json
SPECIALIZE ?= false
empty :=
space := $(empty) $(empty)
COMPILED_RULES_CMAKE_GENERATOR ?= $(if $(shell command -v ninja 2>/dev/null),Ninja,)
COMPILED_RULES_BUILD_GENERATOR_SUFFIX = $(if $(COMPILED_RULES_CMAKE_GENERATOR),-$(subst $(space),_,$(COMPILED_RULES_CMAKE_GENERATOR)),)
COMPILED_RULES_BUILD_ROOT ?= $(BUILD_DIR)/compiled-rules-builds$(COMPILED_RULES_BUILD_GENERATOR_SUFFIX)
COMPILED_RULES_ARTIFACT_ROOT ?= $(BUILD_DIR)/compiled-rules
COMPILED_RULES_MAX_ROWS ?= 1
COMPILED_RULES_LTO ?= false
COMPILED_RULES_LINK_DEDUP ?= false
COMPILED_RULES_EXPORT_SYMBOLS ?= false
COMPILED_RULES_OPT_LEVEL ?= 1
COMPILED_RULES_BUILD_JOBS ?= auto
COMPILED_RULES_SHARED_SINGLE_BUILD ?= true
COMPILED_RULES_REUSE_SINGLE_CPP ?= true
COMPILED_RULES_REUSE_SHARDED_CPP ?= true
COMPILED_RULES_COMPILER_LAUNCHER ?=
COMPILED_RULES_CMAKE_GENERATOR_ARG = $(if $(COMPILED_RULES_CMAKE_GENERATOR),-G "$(COMPILED_RULES_CMAKE_GENERATOR)",)
COMPILED_RULES_COMPILER_LAUNCHER_ARGS = $(if $(COMPILED_RULES_COMPILER_LAUNCHER),-DCMAKE_C_COMPILER_LAUNCHER=$(COMPILED_RULES_COMPILER_LAUNCHER) -DCMAKE_CXX_COMPILER_LAUNCHER=$(COMPILED_RULES_COMPILER_LAUNCHER),)
COMPILED_RULES_CMAKE_ARGS = $(COMPILED_RULES_CMAKE_GENERATOR_ARG) -DPS_MASK_WORD_BITS=64 -DPS_ENABLE_LTO=$(COMPILED_RULES_LTO) -DPS_ENABLE_LINK_DEDUP=$(COMPILED_RULES_LINK_DEDUP) -DPS_ENABLE_EXPORTED_SYMBOLS=$(COMPILED_RULES_EXPORT_SYMBOLS) -DPS_COMPILED_RULES_OPT_LEVEL=$(COMPILED_RULES_OPT_LEVEL) $(COMPILED_RULES_COMPILER_LAUNCHER_ARGS)
ifeq ($(COMPILED_RULES_BUILD_JOBS),auto)
COMPILED_RULES_BUILD_PARALLEL_ARG = --parallel
else
COMPILED_RULES_BUILD_PARALLEL_ARG = --parallel $(COMPILED_RULES_BUILD_JOBS)
endif
define COMPILED_RULES_CONFIGURE
configure_stamp="$(1)/.compiled-rules-cmake-args"; \
new_configure_stamp="$(COMPILED_RULES_CMAKE_ARGS) $(2)"; \
if [ ! -f "$(1)/CMakeCache.txt" ] || [ ! -f "$$configure_stamp" ] || [ "$$(cat "$$configure_stamp")" != "$$new_configure_stamp" ]; then \
	$(CMAKE) -S . -B "$(1)" $(COMPILED_RULES_CMAKE_ARGS) $(2); \
	printf '%s\n' "$$new_configure_stamp" > "$$configure_stamp"; \
fi
endef
define COMPILED_RULES_BOOTSTRAP_CPP
native_configure_stamp="$(BUILD_DIR)/.native-configure.stamp"; \
needs_bootstrap_build=0; \
if [ ! -f "$$native_configure_stamp" ] || [ CMakeLists.txt -nt "$$native_configure_stamp" ] || [ native/CMakeLists.txt -nt "$$native_configure_stamp" ]; then \
	$(CMAKE) -S . -B $(BUILD_DIR) -DPS_MASK_WORD_BITS=64; \
	$(CMAKE) -E touch "$$native_configure_stamp"; \
	needs_bootstrap_build=1; \
fi; \
if [ ! -x "$(PUZZLESCRIPT_CPP)" ]; then \
	needs_bootstrap_build=1; \
elif find native/src/cli native/src/compiler native/src/runtime native/src/player native/include/puzzlescript -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.c' \) -newer "$(PUZZLESCRIPT_CPP)" -print -quit | grep -q .; then \
	needs_bootstrap_build=1; \
fi; \
if [ "$$needs_bootstrap_build" -eq 1 ]; then \
	$(CMAKE) --build $(BUILD_DIR) --target puzzlescript_cpp; \
fi
endef
define COMPILED_RULES_EMIT_SHARDED
out_stamp="$(1)/sources.stamp"; \
out_stamp_text="max_rows=$(COMPILED_RULES_MAX_ROWS)"; \
if [ "$(COMPILED_RULES_REUSE_SHARDED_CPP)" = "true" ] && [ -f "$$sources_file" ] && [ -f "$$out_stamp" ] && [ "$$(cat "$$out_stamp")" = "$$out_stamp_text" ] && [ ! "$(PUZZLESCRIPT_CPP)" -nt "$$out_stamp" ]; then \
	echo "compiled-rules: reuse output=$$out_cpp_dir"; \
else \
	$(PUZZLESCRIPT_CPP) compile-rules "$(2)" --emit-cpp-dir "$$out_cpp_dir" --emit-sources-list "$$sources_file" --symbol $(3) --max-rows $(COMPILED_RULES_MAX_ROWS); \
	printf '%s\n' "$$out_stamp_text" > "$$out_stamp"; \
fi
endef
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
PUZZLESCRIPT_SOLVER_REBUILD_INPUTS := \
	$(wildcard native/src/solver/*.cpp) \
	$(wildcard native/src/generator/*.cpp) \
	$(wildcard native/src/runtime/*.cpp) \
	$(wildcard native/src/runtime/*.hpp) \
	$(wildcard native/src/compiler/*.cpp) \
	$(wildcard native/src/compiler/*.hpp) \
	$(wildcard native/include/puzzlescript/*.h)
PUZZLESCRIPT_CPP_REBUILD_INPUTS := \
	$(wildcard native/src/cli/*.cpp) \
	$(wildcard native/src/cli/*.hpp) \
	$(wildcard native/src/player/*.cpp) \
	$(wildcard native/src/player/*.hpp) \
	$(wildcard native/src/runtime/*.cpp) \
	$(wildcard native/src/runtime/*.hpp) \
	$(wildcard native/src/compiler/*.cpp) \
	$(wildcard native/src/compiler/*.hpp) \
	$(wildcard native/include/puzzlescript/*.h)
SOLVER_MINE_MAX_TARGETS_ARG := $(if $(SOLVER_MINE_MAX_TARGETS),--max-targets $(SOLVER_MINE_MAX_TARGETS),)
SOLVER_TARGET_BENCH_TIMEOUT_ARG := $(if $(SOLVER_TARGET_BENCH_TIMEOUT_MS),--timeout-ms $(SOLVER_TARGET_BENCH_TIMEOUT_MS),)
ifeq ($(SPECIALIZE),true)
SOLVER_TARGET_PREREQ :=
GENERATOR_TARGET_PREREQ :=
else
SOLVER_TARGET_PREREQ := $(PUZZLESCRIPT_SOLVER)
GENERATOR_TARGET_PREREQ := $(PUZZLESCRIPT_GENERATOR)
endif

help:
	@echo "PuzzleScript C++ workflow"
	@echo ""
	@echo "Common commands:"
	@echo "  make build                         Build build/native/puzzlescript_cpp (64-bit masks)"
	@echo "  make build_solver                  Build build/native/puzzlescript_solver"
	@echo "  make build_generator               Build build/native/puzzlescript_generator"
	@echo "  make solver game.txt               Run solver on a PuzzleScript game"
	@echo "  make solver game.txt SPECIALIZE=true"
	@echo "                                     Run solver with linked compiled-rule kernels"
	@echo "  make generator game.txt spec.gen   Run generator on a PuzzleScript game/spec pair"
	@echo "  make generator game.txt spec.gen SPECIALIZE=true"
	@echo "                                     Run generator with linked compiled-rule kernels"
	@echo "                                     Set COMPILED_RULES_MAX_ROWS=N for experimental multi-row kernels"
	@echo "                                     Set COMPILED_RULES_LTO=true to re-enable LTO for specialized builds"
	@echo "                                     Set COMPILED_RULES_LINK_DEDUP=true to re-enable Darwin link dedup"
	@echo "                                     Set COMPILED_RULES_EXPORT_SYMBOLS=true to keep Darwin main exports"
	@echo "                                     Set COMPILED_RULES_OPT_LEVEL=2/3 to spend more compile time on generated rules"
	@echo "                                     Set COMPILED_RULES_BUILD_JOBS=N to tune specialized build parallelism"
	@echo "                                     Set COMPILED_RULES_SHARED_SINGLE_BUILD=false for per-game build dirs"
	@echo "                                     Set COMPILED_RULES_REUSE_SINGLE_CPP=false to force one-game regeneration"
	@echo "                                     Set COMPILED_RULES_REUSE_SHARDED_CPP=false to force corpus regeneration"
	@echo "                                     Uses Ninja automatically when installed; override COMPILED_RULES_CMAKE_GENERATOR="
	@echo "                                     Set COMPILED_RULES_COMPILER_LAUNCHER=ccache after installing ccache"
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
	@echo "  make generator_smoke_tests         Run native generator smoke tests"
	@echo "  make generator_benchmark           Run fixed-seed generator preset benchmark"
	@echo "  make solver_mine_pippable          Mine near-threshold native solver targets"
	@echo "  make solver_benchmark_targets      Benchmark mined solver targets repeatedly"
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
	@echo "  make solver_tests_cpp SPECIALIZE=true"
	@echo "                                     Run standalone native solver corpus with compiled rules"
	@echo "  make solver_tests_js               Run JavaScript comparison solver corpus"
	@echo "  make solver_tests SOLVER_TIMEOUT_MS=5000"
	@echo "                                     Run solver corpus with a deeper timeout"
	@echo "  make solver_tests SOLVER_JOBS=1"
	@echo "                                     Run native solver corpus serially"
	@echo "  make solver_tests SOLVER_JOBS=auto"
	@echo "                                     Run native solver corpus in parallel for faster iteration"
	@echo "  make solver_tests SOLVER_STRATEGY=bfs"
	@echo "                                     Run native solver with one strategy"
	@echo "  make solver_tests SOLVER_PROGRESS_EVERY=1"
	@echo "                                     Show solver progress for every level"
	@echo "  make solver_tests SOLVER_OUTPUT_ARGS="
	@echo "                                     Print per-level solver results after the run"
	@echo "  make solver_tests SOLVER_SOLUTIONS_DIR=/tmp/solver-solutions"
	@echo "                                     Write annotated solved-level sources elsewhere"
	@echo "  make generator_benchmark GENERATOR_BENCH_SAMPLES=200 GENERATOR_BENCH_RUNS=3"
	@echo "                                     Run fixed-seed generator preset benchmark"
	@echo "  make solver_mine_pippable SOLVER_MINE_TIMEOUTS_MS=50,100,250,500"
	@echo "                                     Write $(SOLVER_PIPPABLE_MANIFEST)"
	@echo "  make solver_benchmark_targets SOLVER_TARGET_BENCH_RUNS=10"
	@echo "                                     Write $(SOLVER_TARGET_BENCH_OUT)"
	@echo "  make solver_benchmark SPECIALIZE=true"
	@echo "                                     Benchmark solver with compiled rules for the corpus"
	@echo ""
	@echo "Direct executable after build:"
	@echo "  build/native/puzzlescript_cpp --help"
	@echo "  build/native/puzzlescript_solver src/tests/solver_tests --timeout-ms $(SOLVER_TIMEOUT_MS) --jobs $(SOLVER_JOBS) --strategy $(SOLVER_STRATEGY) --solutions-dir $(SOLVER_SOLUTIONS_DIR)/native $(SOLVER_PROGRESS_ARGS) $(SOLVER_OUTPUT_ARGS)"
	@echo "  make generator src/demo/sokoban_basic.txt src/tests/generator_presets/sokoban_room_scatter.gen"
	@echo "  make generator src/demo/sokoban_basic.txt src/tests/generator_presets/sokoban_room_scatter.gen GENERATOR_ARGS='--time-ms 5000 --jobs auto --json-out build/generated/results.json'"

$(CMAKE_CACHE): CMakeLists.txt native/CMakeLists.txt
	$(CMAKE) -S . -B $(BUILD_DIR) -DPS_MASK_WORD_BITS=64

$(PUZZLESCRIPT_CPP): $(CMAKE_CACHE) $(PUZZLESCRIPT_CPP_REBUILD_INPUTS)
	$(CMAKE) -S . -B $(BUILD_DIR) -DPS_MASK_WORD_BITS=64
	$(CMAKE) --build $(BUILD_DIR) --target puzzlescript_cpp

$(PUZZLESCRIPT_SOLVER): $(CMAKE_CACHE) $(PUZZLESCRIPT_SOLVER_REBUILD_INPUTS)
	$(CMAKE) -S . -B $(BUILD_DIR) -DPS_MASK_WORD_BITS=64
	$(CMAKE) --build $(BUILD_DIR) --target puzzlescript_solver

$(PUZZLESCRIPT_GENERATOR): $(CMAKE_CACHE) $(PUZZLESCRIPT_SOLVER_REBUILD_INPUTS)
	$(CMAKE) -S . -B $(BUILD_DIR) -DPS_MASK_WORD_BITS=64
	$(CMAKE) --build $(BUILD_DIR) --target puzzlescript_generator

build: $(CMAKE_CACHE)
	$(CMAKE) -S . -B $(BUILD_DIR) -DPS_MASK_WORD_BITS=64
	$(CMAKE) --build $(BUILD_DIR) --target puzzlescript_cpp

build_solver: $(PUZZLESCRIPT_SOLVER)

build_generator: $(PUZZLESCRIPT_GENERATOR)

generator:
	@if [ -z "$(GENERATOR_GAME)" ] || [ -z "$(GENERATOR_SPEC)" ]; then \
		echo "Usage: make generator path/to/game.txt path/to/spec.gen"; \
		echo "       make generator path/to/game.txt path/to/spec.gen GENERATOR_ARGS='--time-ms 5000 --jobs auto --json-out build/generated/results.json'"; \
		exit 2; \
	fi
	@if [ "$(SPECIALIZE)" = "true" ]; then \
		set -e; \
		if [ ! -f "$(GENERATOR_GAME)" ]; then echo "Missing generator game: $(GENERATOR_GAME)"; exit 2; fi; \
		if [ ! -f "$(GENERATOR_SPEC)" ]; then echo "Missing generator spec: $(GENERATOR_SPEC)"; exit 2; fi; \
		$(COMPILED_RULES_BOOTSTRAP_CPP); \
		hash=$$(shasum -a 256 "$(GENERATOR_GAME)" | awk '{print $$1}'); \
		out_dir="$(COMPILED_RULES_ARTIFACT_ROOT)/generator-$$hash"; \
		if [ "$(COMPILED_RULES_SHARED_SINGLE_BUILD)" = "true" ]; then \
			build_dir="$(COMPILED_RULES_BUILD_ROOT)/generator-single"; \
		else \
			build_dir="$(COMPILED_RULES_BUILD_ROOT)/generator-$$hash"; \
		fi; \
		out_cpp="$$out_dir/compiled_rules.cpp"; \
		out_stamp="$$out_dir/compiled_rules.stamp"; \
		out_stamp_text="max_rows=$(COMPILED_RULES_MAX_ROWS)"; \
		mkdir -p "$$out_dir"; \
		if [ "$(COMPILED_RULES_REUSE_SINGLE_CPP)" = "true" ] && [ -f "$$out_cpp" ] && [ -f "$$out_stamp" ] && [ "$$(cat "$$out_stamp")" = "$$out_stamp_text" ] && [ ! "$(PUZZLESCRIPT_CPP)" -nt "$$out_stamp" ]; then \
			echo "compiled-rules: reuse output=$$out_cpp"; \
		else \
			$(PUZZLESCRIPT_CPP) compile-rules "$(GENERATOR_GAME)" --emit-cpp "$$out_cpp" --symbol generator_$$hash --max-rows $(COMPILED_RULES_MAX_ROWS); \
			printf '%s\n' "$$out_stamp_text" > "$$out_stamp"; \
		fi; \
		$(call COMPILED_RULES_CONFIGURE,$$build_dir,-DPS_COMPILED_RULES_SOURCE="$$PWD/$$out_cpp" -DPS_COMPILED_RULES_SOURCES_FILE=); \
		$(CMAKE) --build "$$build_dir" $(COMPILED_RULES_BUILD_PARALLEL_ARG) --target puzzlescript_generator; \
		"$$build_dir/native/puzzlescript_generator" $(GENERATOR_GAME) $(GENERATOR_SPEC) $(GENERATOR_ARGS); \
	else \
		$(MAKE) build_generator; \
		$(PUZZLESCRIPT_GENERATOR) $(GENERATOR_GAME) $(GENERATOR_SPEC) $(GENERATOR_ARGS); \
	fi


ifeq ($(firstword $(MAKECMDGOALS)),generator)
ifneq ($(strip $(GENERATOR_MAKE_ARGS)),)
.PHONY: $(GENERATOR_MAKE_ARGS)
$(eval $(GENERATOR_MAKE_ARGS):;@:)
endif
endif

solver:
	@if [ -z "$(SOLVER_GAME)" ]; then \
		echo "Usage: make solver path/to/game.txt"; \
		echo "       make solver path/to/game.txt SOLVER_ARGS='--level 0 --json --quiet'"; \
		exit 2; \
	fi
	@if [ "$(SPECIALIZE)" = "true" ]; then \
		set -e; \
		if [ ! -f "$(SOLVER_GAME)" ]; then echo "Missing solver game: $(SOLVER_GAME)"; exit 2; fi; \
		$(COMPILED_RULES_BOOTSTRAP_CPP); \
		hash=$$(shasum -a 256 "$(SOLVER_GAME)" | awk '{print $$1}'); \
		out_dir="$(COMPILED_RULES_ARTIFACT_ROOT)/solver-$$hash"; \
		if [ "$(COMPILED_RULES_SHARED_SINGLE_BUILD)" = "true" ]; then \
			build_dir="$(COMPILED_RULES_BUILD_ROOT)/solver-single"; \
		else \
			build_dir="$(COMPILED_RULES_BUILD_ROOT)/solver-$$hash"; \
		fi; \
		out_cpp="$$out_dir/compiled_rules.cpp"; \
		out_stamp="$$out_dir/compiled_rules.stamp"; \
		out_stamp_text="max_rows=$(COMPILED_RULES_MAX_ROWS)"; \
		mkdir -p "$$out_dir"; \
		if [ "$(COMPILED_RULES_REUSE_SINGLE_CPP)" = "true" ] && [ -f "$$out_cpp" ] && [ -f "$$out_stamp" ] && [ "$$(cat "$$out_stamp")" = "$$out_stamp_text" ] && [ ! "$(PUZZLESCRIPT_CPP)" -nt "$$out_stamp" ]; then \
			echo "compiled-rules: reuse output=$$out_cpp"; \
		else \
			$(PUZZLESCRIPT_CPP) compile-rules "$(SOLVER_GAME)" --emit-cpp "$$out_cpp" --symbol solver_$$hash --max-rows $(COMPILED_RULES_MAX_ROWS); \
			printf '%s\n' "$$out_stamp_text" > "$$out_stamp"; \
		fi; \
		$(call COMPILED_RULES_CONFIGURE,$$build_dir,-DPS_COMPILED_RULES_SOURCE="$$PWD/$$out_cpp" -DPS_COMPILED_RULES_SOURCES_FILE=); \
		$(CMAKE) --build "$$build_dir" $(COMPILED_RULES_BUILD_PARALLEL_ARG) --target puzzlescript_solver; \
		"$$build_dir/native/puzzlescript_solver" $(SOLVER_GAME) --timeout-ms $(SOLVER_TIMEOUT_MS) --jobs $(SOLVER_JOBS) --strategy $(SOLVER_STRATEGY) --solutions-dir $(SOLVER_SOLUTIONS_DIR)/native $(SOLVER_PROGRESS_ARGS) $(SOLVER_OUTPUT_ARGS) $(SOLVER_ARGS); \
	else \
		$(MAKE) build_solver; \
		$(PUZZLESCRIPT_SOLVER) $(SOLVER_GAME) --timeout-ms $(SOLVER_TIMEOUT_MS) --jobs $(SOLVER_JOBS) --strategy $(SOLVER_STRATEGY) --solutions-dir $(SOLVER_SOLUTIONS_DIR)/native $(SOLVER_PROGRESS_ARGS) $(SOLVER_OUTPUT_ARGS) $(SOLVER_ARGS); \
	fi


ifeq ($(firstword $(MAKECMDGOALS)),solver)
ifneq ($(strip $(SOLVER_MAKE_ARGS)),)
.PHONY: $(SOLVER_MAKE_ARGS)
$(eval $(SOLVER_MAKE_ARGS):;@:)
endif
endif

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

ifeq ($(SOLVER_PROGRESS_EVERY),game)
SOLVER_PROGRESS_ARGS := --progress-per-game
else
SOLVER_PROGRESS_ARGS := --progress-every $(SOLVER_PROGRESS_EVERY)
endif

solver_smoke_tests: $(SOLVER_TARGET_PREREQ)
	@if [ "$(SPECIALIZE)" = "true" ]; then \
		set -e; \
		$(COMPILED_RULES_BOOTSTRAP_CPP); \
		hash=$$(find src/tests/solver_smoke_tests -type f -name '*.txt' -print0 | sort -z | xargs -0 shasum -a 256 | shasum -a 256 | awk '{print $$1}'); \
		out_dir="$(COMPILED_RULES_ARTIFACT_ROOT)/solver-smoke-$$hash"; \
		build_dir="$(COMPILED_RULES_BUILD_ROOT)/solver-smoke-$$hash"; \
		out_cpp_dir="$$out_dir/sources"; \
		sources_file="$$out_dir/sources.txt"; \
		mkdir -p "$$out_dir"; \
		$(call COMPILED_RULES_EMIT_SHARDED,$$out_dir,src/tests/solver_smoke_tests,solver_smoke_$$hash); \
		$(call COMPILED_RULES_CONFIGURE,$$build_dir,-DPS_COMPILED_RULES_SOURCE= -DPS_COMPILED_RULES_SOURCES_FILE="$$PWD/$$sources_file"); \
		$(CMAKE) --build "$$build_dir" $(COMPILED_RULES_BUILD_PARALLEL_ARG) --target puzzlescript_solver; \
		$(NODE) src/tests/run_solver_smoke_assert.js "$$build_dir/native/puzzlescript_solver" src/tests/solver_smoke_tests --timeout-ms 1000; \
	else \
		$(NODE) src/tests/run_solver_smoke_assert.js $(PUZZLESCRIPT_SOLVER) src/tests/solver_smoke_tests --timeout-ms 1000; \
	fi

solver_determinism_tests: $(SOLVER_TARGET_PREREQ)
	@if [ "$(SPECIALIZE)" = "true" ]; then \
		set -e; \
		$(COMPILED_RULES_BOOTSTRAP_CPP); \
		hash=$$(find src/tests/solver_smoke_tests -type f -name '*.txt' -print0 | sort -z | xargs -0 shasum -a 256 | shasum -a 256 | awk '{print $$1}'); \
		out_dir="$(COMPILED_RULES_ARTIFACT_ROOT)/solver-smoke-$$hash"; \
		build_dir="$(COMPILED_RULES_BUILD_ROOT)/solver-smoke-$$hash"; \
		out_cpp_dir="$$out_dir/sources"; \
		sources_file="$$out_dir/sources.txt"; \
		mkdir -p "$$out_dir"; \
		$(call COMPILED_RULES_EMIT_SHARDED,$$out_dir,src/tests/solver_smoke_tests,solver_smoke_$$hash); \
		$(call COMPILED_RULES_CONFIGURE,$$build_dir,-DPS_COMPILED_RULES_SOURCE= -DPS_COMPILED_RULES_SOURCES_FILE="$$PWD/$$sources_file"); \
		$(CMAKE) --build "$$build_dir" $(COMPILED_RULES_BUILD_PARALLEL_ARG) --target puzzlescript_solver; \
		$(NODE) src/tests/run_solver_determinism.js "$$build_dir/native/puzzlescript_solver" src/tests/solver_smoke_tests --runs 5 --timeout-ms 1000; \
	else \
		$(NODE) src/tests/run_solver_determinism.js $(PUZZLESCRIPT_SOLVER) src/tests/solver_smoke_tests --runs 5 --timeout-ms 1000; \
	fi

solver_parity_smoke: $(SOLVER_TARGET_PREREQ)
	@if [ "$(SPECIALIZE)" = "true" ]; then \
		set -e; \
		$(COMPILED_RULES_BOOTSTRAP_CPP); \
		hash=$$(find src/tests/solver_smoke_tests -type f -name '*.txt' -print0 | sort -z | xargs -0 shasum -a 256 | shasum -a 256 | awk '{print $$1}'); \
		out_dir="$(COMPILED_RULES_ARTIFACT_ROOT)/solver-smoke-$$hash"; \
		build_dir="$(COMPILED_RULES_BUILD_ROOT)/solver-smoke-$$hash"; \
		out_cpp_dir="$$out_dir/sources"; \
		sources_file="$$out_dir/sources.txt"; \
		mkdir -p "$$out_dir"; \
		$(call COMPILED_RULES_EMIT_SHARDED,$$out_dir,src/tests/solver_smoke_tests,solver_smoke_$$hash); \
		$(call COMPILED_RULES_CONFIGURE,$$build_dir,-DPS_COMPILED_RULES_SOURCE= -DPS_COMPILED_RULES_SOURCES_FILE="$$PWD/$$sources_file"); \
		$(CMAKE) --build "$$build_dir" $(COMPILED_RULES_BUILD_PARALLEL_ARG) --target puzzlescript_solver; \
		$(NODE) src/tests/run_solver_parity_smoke.js "$$build_dir/native/puzzlescript_solver" src/tests/solver_smoke_tests; \
	else \
		$(NODE) src/tests/run_solver_parity_smoke.js $(PUZZLESCRIPT_SOLVER) src/tests/solver_smoke_tests; \
	fi

generator_smoke_tests: $(GENERATOR_TARGET_PREREQ)
	@if [ "$(SPECIALIZE)" = "true" ]; then \
		$(MAKE) generator src/demo/sokoban_basic.txt src/tests/generator_presets/sokoban_room_scatter.gen SPECIALIZE=true GENERATOR_ARGS="--time-ms 100 --quiet"; \
	else \
		$(NODE) src/tests/run_generator_smoke.js $(PUZZLESCRIPT_GENERATOR) src/demo/sokoban_basic.txt; \
	fi

generator_benchmark: $(PUZZLESCRIPT_GENERATOR)
	$(NODE) src/tests/run_generator_benchmark.js $(PUZZLESCRIPT_GENERATOR) $(GENERATOR_BENCH_GAME) --presets-dir $(GENERATOR_BENCH_PRESETS_DIR) --samples $(GENERATOR_BENCH_SAMPLES) --runs $(GENERATOR_BENCH_RUNS) --jobs $(GENERATOR_BENCH_JOBS) --seed $(GENERATOR_BENCH_SEED) --solver-timeout-ms $(GENERATOR_BENCH_SOLVER_TIMEOUT_MS) --solver-strategy $(GENERATOR_BENCH_SOLVER_STRATEGY) --top-k $(GENERATOR_BENCH_TOP_K) --out $(GENERATOR_BENCH_OUT)

solver_tests_cpp: $(SOLVER_TARGET_PREREQ)
	@if [ "$(SPECIALIZE)" = "true" ]; then \
		set -e; \
		if [ ! -e "$(SOLVER_TESTS_CORPUS)" ]; then echo "Missing solver corpus: $(SOLVER_TESTS_CORPUS)"; exit 2; fi; \
		$(COMPILED_RULES_BOOTSTRAP_CPP); \
		hash=$$(find "$(SOLVER_TESTS_CORPUS)" -type f -name '*.txt' -print0 | sort -z | xargs -0 shasum -a 256 | shasum -a 256 | awk '{print $$1}'); \
		out_dir="$(COMPILED_RULES_ARTIFACT_ROOT)/solver-corpus-$$hash"; \
		build_dir="$(COMPILED_RULES_BUILD_ROOT)/solver-corpus-$$hash"; \
		out_cpp_dir="$$out_dir/sources"; \
		sources_file="$$out_dir/sources.txt"; \
		mkdir -p "$$out_dir"; \
		$(call COMPILED_RULES_EMIT_SHARDED,$$out_dir,$(SOLVER_TESTS_CORPUS),solver_corpus_$$hash); \
		$(call COMPILED_RULES_CONFIGURE,$$build_dir,-DPS_COMPILED_RULES_SOURCE= -DPS_COMPILED_RULES_SOURCES_FILE="$$PWD/$$sources_file"); \
		$(CMAKE) --build "$$build_dir" $(COMPILED_RULES_BUILD_PARALLEL_ARG) --target puzzlescript_solver; \
		"$$build_dir/native/puzzlescript_solver" $(SOLVER_TESTS_CORPUS) --timeout-ms $(SOLVER_TIMEOUT_MS) --jobs $(SOLVER_JOBS) --strategy $(SOLVER_STRATEGY) --solutions-dir $(SOLVER_SOLUTIONS_DIR)/native $(SOLVER_PROGRESS_ARGS) $(SOLVER_OUTPUT_ARGS); \
	else \
		$(PUZZLESCRIPT_SOLVER) $(SOLVER_TESTS_CORPUS) --timeout-ms $(SOLVER_TIMEOUT_MS) --jobs $(SOLVER_JOBS) --strategy $(SOLVER_STRATEGY) --solutions-dir $(SOLVER_SOLUTIONS_DIR)/native $(SOLVER_PROGRESS_ARGS) $(SOLVER_OUTPUT_ARGS); \
	fi

solver_tests_js:
	$(NODE) src/tests/run_solver_tests_js.js src/tests/solver_tests --timeout-ms $(SOLVER_TIMEOUT_MS) --solutions-dir $(SOLVER_SOLUTIONS_DIR)/js $(SOLVER_PROGRESS_ARGS) $(SOLVER_OUTPUT_ARGS)

solver_tests: solver_smoke_tests solver_determinism_tests solver_parity_smoke solver_tests_cpp solver_tests_js

solver_benchmark: $(SOLVER_TARGET_PREREQ)
	@if [ "$(SPECIALIZE)" = "true" ]; then \
		set -e; \
		if [ ! -e "$(SOLVER_BENCH_CORPUS)" ]; then echo "Missing solver benchmark corpus: $(SOLVER_BENCH_CORPUS)"; exit 2; fi; \
		$(COMPILED_RULES_BOOTSTRAP_CPP); \
		hash=$$(find "$(SOLVER_BENCH_CORPUS)" -type f -name '*.txt' -print0 | sort -z | xargs -0 shasum -a 256 | shasum -a 256 | awk '{print $$1}'); \
		out_dir="$(COMPILED_RULES_ARTIFACT_ROOT)/solver-bench-$$hash"; \
		build_dir="$(COMPILED_RULES_BUILD_ROOT)/solver-bench-$$hash"; \
		out_cpp_dir="$$out_dir/sources"; \
		sources_file="$$out_dir/sources.txt"; \
		mkdir -p "$$out_dir"; \
		$(call COMPILED_RULES_EMIT_SHARDED,$$out_dir,$(SOLVER_BENCH_CORPUS),solver_bench_$$hash); \
		$(call COMPILED_RULES_CONFIGURE,$$build_dir,-DPS_COMPILED_RULES_SOURCE= -DPS_COMPILED_RULES_SOURCES_FILE="$$PWD/$$sources_file"); \
		$(CMAKE) --build "$$build_dir" $(COMPILED_RULES_BUILD_PARALLEL_ARG) --target puzzlescript_solver; \
		$(NODE) src/tests/run_solver_benchmark.js "$$build_dir/native/puzzlescript_solver" $(SOLVER_BENCH_CORPUS) --runs $(SOLVER_BENCH_RUNS) --timeout-ms $(SOLVER_BENCH_TIMEOUT_MS) --jobs $(SOLVER_BENCH_JOBS) --strategy $(SOLVER_BENCH_STRATEGY) --out $(SOLVER_BENCH_OUT) --baseline $(SOLVER_PERF_BASELINE); \
	else \
		$(NODE) src/tests/run_solver_benchmark.js $(PUZZLESCRIPT_SOLVER) $(SOLVER_BENCH_CORPUS) --runs $(SOLVER_BENCH_RUNS) --timeout-ms $(SOLVER_BENCH_TIMEOUT_MS) --jobs $(SOLVER_BENCH_JOBS) --strategy $(SOLVER_BENCH_STRATEGY) --out $(SOLVER_BENCH_OUT) --baseline $(SOLVER_PERF_BASELINE); \
	fi

solver_mine_pippable: $(PUZZLESCRIPT_SOLVER)
	$(NODE) src/tests/mine_solver_near_threshold.js $(PUZZLESCRIPT_SOLVER) $(SOLVER_MINE_CORPUS) --timeouts-ms $(SOLVER_MINE_TIMEOUTS_MS) --strategy $(SOLVER_MINE_STRATEGY) --near-ratio $(SOLVER_MINE_NEAR_RATIO) --out $(SOLVER_PIPPABLE_MANIFEST) $(SOLVER_MINE_MAX_TARGETS_ARG)

solver_benchmark_targets: $(PUZZLESCRIPT_SOLVER)
	$(NODE) src/tests/run_solver_level_benchmark.js $(PUZZLESCRIPT_SOLVER) $(SOLVER_TARGET_BENCH_CORPUS) $(SOLVER_TARGET_BENCH_MANIFEST) --runs $(SOLVER_TARGET_BENCH_RUNS) --strategy $(SOLVER_TARGET_BENCH_STRATEGY) --out $(SOLVER_TARGET_BENCH_OUT) $(SOLVER_TARGET_BENCH_TIMEOUT_ARG)

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
