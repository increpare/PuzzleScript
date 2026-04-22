.PHONY: tests run ctest-native clean clean-native clean-fixtures configure-native build-native coverage-fixtures

NODE ?= node
CMAKE ?= cmake
BUILD_DIR ?= build/native
PS_CLI := $(BUILD_DIR)/native/ps_cli
COVERAGE_FIXTURES_DIR := $(BUILD_DIR)/coverage-fixtures
COVERAGE_FIXTURES_MANIFEST := $(COVERAGE_FIXTURES_DIR)/fixtures.json

CMAKE_CACHE := $(BUILD_DIR)/CMakeCache.txt

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

ctest-native: $(PS_CLI)
	ctest --test-dir $(BUILD_DIR) --output-on-failure

$(COVERAGE_FIXTURES_MANIFEST): $(JS_FIXTURE_INPUTS)
	$(NODE) src/tests/export_native_fixtures.js $(COVERAGE_FIXTURES_DIR)

tests: build-native $(COVERAGE_FIXTURES_MANIFEST)
	$(NODE) src/tests/run_native_trace_suite.js $(COVERAGE_FIXTURES_MANIFEST) --cli $(PS_CLI) --progress-every 1 --timeout-ms 45000

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
