.PHONY: configure-native build-native ctest-native coverage-fixtures tests

NODE ?= node
CMAKE ?= cmake
BUILD_DIR ?= build/native
PS_CLI := $(BUILD_DIR)/native/ps_cli
COVERAGE_FIXTURES_DIR := $(BUILD_DIR)/coverage-fixtures
COVERAGE_FIXTURES_MANIFEST := $(COVERAGE_FIXTURES_DIR)/fixtures.json

configure-native:
	$(CMAKE) -S . -B $(BUILD_DIR)

build-native: configure-native
	$(CMAKE) --build $(BUILD_DIR)

ctest-native: build-native
	ctest --test-dir $(BUILD_DIR) --output-on-failure

coverage-fixtures:
	$(NODE) src/tests/export_native_fixtures.js $(COVERAGE_FIXTURES_DIR)

tests: build-native coverage-fixtures
	$(NODE) src/tests/run_native_trace_suite.js $(COVERAGE_FIXTURES_MANIFEST) --cli $(PS_CLI) --progress-every 1 --timeout-ms 30000
