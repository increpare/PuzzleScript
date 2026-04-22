.PHONY: configure-native build-native ctest-native coverage-fixtures tests run

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
