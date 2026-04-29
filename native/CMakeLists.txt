cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Performance defaults ----------------------------------------------------
# Default to Release when the user hasn't set anything. Honor explicit Debug.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Build type (Release, Debug, RelWithDebInfo, MinSizeRel)" FORCE)
  message(STATUS "Defaulting CMAKE_BUILD_TYPE to Release (native perf build).")
endif()

# Release-build flags: -O3, optional LTO, native codegen, NDEBUG. Tuned for the
# performance target in docs/superpowers/specs/2026-04-22-native-perf-crush-design.md.
# Note: -fno-exceptions / -fno-rtti omitted because core.cpp and cli_main.cpp
# use throw / try / catch (e.g. json::ParseError) pervasively for parse-error
# reporting; enabling those flags would require a large rewrite outside the
# scope of this perf phase.
option(PS_ENABLE_LTO "Enable Release-build link-time optimization" ON)
option(PS_ENABLE_LINK_DEDUP "Enable Darwin ld64 duplicate-code folding in Release builds" ON)
option(PS_ENABLE_EXPORTED_SYMBOLS "Enable exported symbols from Release main executables on Darwin" ON)
add_compile_options(
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-DNDEBUG>
    $<$<CONFIG:Release>:-march=native>
)
if(PS_ENABLE_LTO)
  add_compile_options($<$<CONFIG:Release>:-flto>)
  add_link_options($<$<CONFIG:Release>:-flto>)
endif()
if(NOT PS_ENABLE_LINK_DEDUP AND APPLE)
  add_link_options($<$<CONFIG:Release>:-Wl,-no_deduplicate>)
endif()
if(NOT PS_ENABLE_EXPORTED_SYMBOLS AND APPLE)
  add_link_options($<$<CONFIG:Release>:-Wl,-no_exported_symbols>)
endif()

# libc++ safe-mode instrumentation (seen as __annotate_contiguous_container in
# baseline profiles) is a large tax; turn it off on Release only.
add_compile_definitions(
    $<$<CONFIG:Release>:_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_NONE>
)

find_package(Threads REQUIRED)
find_package(PkgConfig QUIET)
find_program(NODE_EXECUTABLE node REQUIRED)

# Repo root (parent of this `native/` tree). `CMAKE_SOURCE_DIR` is the checkout
# only when this file is not the top-level project; when users configure from
# `native/` directly, `CMAKE_SOURCE_DIR` points at `native/` and repo paths must
# not be derived from it.
get_filename_component(PUZZLESCRIPT_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

if(PKG_CONFIG_FOUND)
  pkg_check_modules(SDL2 IMPORTED_TARGET sdl2)
endif()

set(PUZZLESCRIPT_NATIVE_PUBLIC_HEADERS
  include/puzzlescript/puzzlescript.h
)

set(PUZZLESCRIPT_COMPILER_PUBLIC_HEADERS
  include/puzzlescript/compiler.h
)

set(PUZZLESCRIPT_COMPILER_SOURCES
  src/compiler/compact_turn_codegen.cpp
  src/compiler/compiled_rules_codegen.cpp
  src/compiler/compile_diagnostics.cpp
  src/compiler/c_api.cpp
  src/compiler/diagnostic.cpp
  src/compiler/lower_to_runtime.cpp
  src/compiler/parser.cpp
  src/compiler/rule_text.cpp
  third_party/utf8proc/utf8proc.c
)

set(PUZZLESCRIPT_NATIVE_SOURCES
  src/runtime/compiled_rules.cpp
  src/runtime/c_api.cpp
  src/runtime/core.cpp
  src/runtime/hash.cpp
  src/runtime/json.cpp
  src/runtime/simd.cpp
  third_party/simdjson/simdjson.cpp
)

add_library(puzzlescript_native STATIC
  ${PUZZLESCRIPT_NATIVE_PUBLIC_HEADERS}
  ${PUZZLESCRIPT_NATIVE_SOURCES}
)

option(PS_ENABLE_DEV_SERIALIZERS "Enable dev-only compiler serializers and commands" ON)
set(PS_MASK_WORD_BITS "64" CACHE STRING "PuzzleScript runtime mask word width: 64 by default, or 32 for JS-oracle compatibility")
set_property(CACHE PS_MASK_WORD_BITS PROPERTY STRINGS 32 64)
if(NOT PS_MASK_WORD_BITS STREQUAL "32" AND NOT PS_MASK_WORD_BITS STREQUAL "64")
  message(FATAL_ERROR "PS_MASK_WORD_BITS must be 32 or 64")
endif()
option(PS_INTERPRETER_OBJECT_CELL_INDEX "Maintain per-object cell index for anchored interpreter rule scans" ON)

add_library(puzzlescript_compiler STATIC
  ${PUZZLESCRIPT_COMPILER_PUBLIC_HEADERS}
  ${PUZZLESCRIPT_COMPILER_SOURCES}
)

target_include_directories(puzzlescript_compiler
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/utf8proc
)

target_compile_definitions(puzzlescript_compiler
  PRIVATE
    UTF8PROC_STATIC
    PS_MASK_WORD_BITS=${PS_MASK_WORD_BITS}
    $<$<BOOL:${PS_ENABLE_DEV_SERIALIZERS}>:PS_ENABLE_DEV_SERIALIZERS=1>
)

set_source_files_properties(
  third_party/utf8proc/utf8proc.c
  PROPERTIES COMPILE_FLAGS "-Wno-everything")

target_include_directories(puzzlescript_native
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson
)

# simdjson's amalgamation uses patterns (SIMD dispatch, internal attributes)
# that produce many warnings under our project-wide flags. Silence warnings
# on that one translation unit; we don't maintain simdjson.
set_source_files_properties(
  third_party/simdjson/simdjson.cpp
  PROPERTIES COMPILE_FLAGS "-Wno-everything")

target_link_libraries(puzzlescript_native
  PUBLIC
    Threads::Threads
)

target_compile_definitions(puzzlescript_native
  PUBLIC
    PS_MASK_WORD_BITS=${PS_MASK_WORD_BITS}
    PS_INTERPRETER_OBJECT_CELL_INDEX=$<BOOL:${PS_INTERPRETER_OBJECT_CELL_INDEX}>
)

add_executable(puzzlescript_cpp
  src/cli/main.cpp
  src/cli/diagnostics_parity.cpp
)

add_executable(puzzlescript_solver
  src/solver/main.cpp
)

add_executable(puzzlescript_generator
  src/generator/main.cpp
)

set(PS_COMPILED_RULES_SOURCE "" CACHE FILEPATH "Optional generated compiled-rules C++ source to link into solver/generator")
set(PS_COMPILED_RULES_SOURCES_FILE "" CACHE FILEPATH "Optional newline-delimited list of generated compiled-rules C++ sources")
set(PS_COMPILED_RULES_OPT_LEVEL "" CACHE STRING "Optional Release optimization level for generated compiled-rules source: 0, 1, 2, 3, s, or z")
if(PS_COMPILED_RULES_SOURCE AND PS_COMPILED_RULES_SOURCES_FILE)
  message(FATAL_ERROR "Set only one of PS_COMPILED_RULES_SOURCE or PS_COMPILED_RULES_SOURCES_FILE")
endif()

set(PS_COMPILED_RULES_SOURCES)
if(PS_COMPILED_RULES_SOURCE)
  list(APPEND PS_COMPILED_RULES_SOURCES "${PS_COMPILED_RULES_SOURCE}")
endif()
if(PS_COMPILED_RULES_SOURCES_FILE)
  get_filename_component(PS_COMPILED_RULES_SOURCES_FILE_ABS "${PS_COMPILED_RULES_SOURCES_FILE}" ABSOLUTE)
  if(NOT EXISTS "${PS_COMPILED_RULES_SOURCES_FILE_ABS}")
    message(FATAL_ERROR "PS_COMPILED_RULES_SOURCES_FILE does not exist: ${PS_COMPILED_RULES_SOURCES_FILE_ABS}")
  endif()
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${PS_COMPILED_RULES_SOURCES_FILE_ABS}")
  file(STRINGS "${PS_COMPILED_RULES_SOURCES_FILE_ABS}" PS_COMPILED_RULES_LISTED_SOURCES)
  list(APPEND PS_COMPILED_RULES_SOURCES ${PS_COMPILED_RULES_LISTED_SOURCES})
endif()

if(PS_COMPILED_RULES_SOURCES)
  set(PS_COMPILED_RULES_SOURCE_ABS_LIST)
  foreach(PS_COMPILED_RULES_SOURCE_ENTRY IN LISTS PS_COMPILED_RULES_SOURCES)
    get_filename_component(PS_COMPILED_RULES_SOURCE_ABS "${PS_COMPILED_RULES_SOURCE_ENTRY}" ABSOLUTE)
    if(NOT EXISTS "${PS_COMPILED_RULES_SOURCE_ABS}")
      message(FATAL_ERROR "Compiled-rules source does not exist: ${PS_COMPILED_RULES_SOURCE_ABS}")
    endif()
    list(APPEND PS_COMPILED_RULES_SOURCE_ABS_LIST "${PS_COMPILED_RULES_SOURCE_ABS}")
  endforeach()
  list(LENGTH PS_COMPILED_RULES_SOURCE_ABS_LIST PS_COMPILED_RULES_SOURCE_COUNT)
  if(PS_COMPILED_RULES_SOURCE_COUNT EQUAL 1)
    list(GET PS_COMPILED_RULES_SOURCE_ABS_LIST 0 PS_COMPILED_RULES_ONLY_SOURCE)
    message(STATUS "Linking compiled-rules source: ${PS_COMPILED_RULES_ONLY_SOURCE}")
  else()
    message(STATUS "Linking compiled-rules sources: ${PS_COMPILED_RULES_SOURCE_COUNT}")
  endif()
  if(NOT PS_COMPILED_RULES_OPT_LEVEL STREQUAL "")
    if(NOT PS_COMPILED_RULES_OPT_LEVEL MATCHES "^(0|1|2|3|s|z)$")
      message(FATAL_ERROR "PS_COMPILED_RULES_OPT_LEVEL must be one of: 0, 1, 2, 3, s, z")
    endif()
    message(STATUS "Compiled-rules source Release opt level: -O${PS_COMPILED_RULES_OPT_LEVEL}")
    foreach(PS_COMPILED_RULES_SOURCE_ABS IN LISTS PS_COMPILED_RULES_SOURCE_ABS_LIST)
      set_source_files_properties(
        "${PS_COMPILED_RULES_SOURCE_ABS}"
        PROPERTIES COMPILE_OPTIONS "$<$<CONFIG:Release>:-O${PS_COMPILED_RULES_OPT_LEVEL}>")
    endforeach()
  endif()
  target_sources(puzzlescript_cpp PRIVATE ${PS_COMPILED_RULES_SOURCE_ABS_LIST})
  target_sources(puzzlescript_solver PRIVATE ${PS_COMPILED_RULES_SOURCE_ABS_LIST})
  target_sources(puzzlescript_generator PRIVATE ${PS_COMPILED_RULES_SOURCE_ABS_LIST})
endif()

target_link_libraries(puzzlescript_solver
  PRIVATE
    puzzlescript_native
    puzzlescript_compiler
)

target_link_libraries(puzzlescript_generator
  PRIVATE
    puzzlescript_native
    puzzlescript_compiler
)

target_include_directories(puzzlescript_solver
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_include_directories(puzzlescript_generator
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(puzzlescript_cpp
  PRIVATE
    puzzlescript_native
    puzzlescript_compiler
)

target_include_directories(puzzlescript_cpp
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(puzzlescript_cpp
  PRIVATE
    PS_NODE_EXECUTABLE="${NODE_EXECUTABLE}"
    PS_EXPORT_IR_SCRIPT="${PUZZLESCRIPT_REPO_ROOT}/src/tests/js_oracle/export_ir_json.js"
    PS_EXPORT_TRACE_SCRIPT="${PUZZLESCRIPT_REPO_ROOT}/src/tests/js_oracle/export_execution_trace.js"
    PS_FONT_JS_PATH="${PUZZLESCRIPT_REPO_ROOT}/src/js/font.js"
)

if(TARGET PkgConfig::SDL2)
  target_sources(puzzlescript_cpp PRIVATE src/player/sdl_player.cpp src/player/sfxr.cpp)
  target_link_libraries(puzzlescript_cpp PRIVATE PkgConfig::SDL2)
  target_compile_definitions(puzzlescript_cpp PRIVATE PS_HAVE_SDL2=1)
endif()

enable_testing()

set(SMOKE_IR ${CMAKE_CURRENT_BINARY_DIR}/generated/sokoban_basic_ir.json)
set(SMOKE_TRACE ${CMAKE_CURRENT_BINARY_DIR}/generated/sokoban_basic_trace.json)
set(SMOKE_TRACE_INPUTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/smoke_trace_inputs.json)
add_custom_command(
  OUTPUT ${SMOKE_IR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/generated
  COMMAND node ${PUZZLESCRIPT_REPO_ROOT}/src/tests/js_oracle/export_ir_json.js
          ${PUZZLESCRIPT_REPO_ROOT}/src/demo/sokoban_basic.txt
          ${SMOKE_IR}
          --level 0
          --seed native-smoke
          --settle-again
  DEPENDS
    ${PUZZLESCRIPT_REPO_ROOT}/src/demo/sokoban_basic.txt
    ${PUZZLESCRIPT_REPO_ROOT}/src/tests/js_oracle/export_ir_json.js
    ${PUZZLESCRIPT_REPO_ROOT}/src/tests/js_oracle/lib/puzzlescript_ir.js
    ${PUZZLESCRIPT_REPO_ROOT}/src/tests/js_oracle/lib/puzzlescript_node_env.js
)

add_custom_command(
  OUTPUT ${SMOKE_TRACE}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/generated
  COMMAND node ${PUZZLESCRIPT_REPO_ROOT}/src/tests/js_oracle/export_execution_trace.js
          ${PUZZLESCRIPT_REPO_ROOT}/src/demo/sokoban_basic.txt
          ${SMOKE_TRACE}
          --level 0
          --seed native-trace-smoke
          --inputs-file ${SMOKE_TRACE_INPUTS}
  DEPENDS
    ${PUZZLESCRIPT_REPO_ROOT}/src/demo/sokoban_basic.txt
    ${PUZZLESCRIPT_REPO_ROOT}/src/tests/js_oracle/export_execution_trace.js
    ${PUZZLESCRIPT_REPO_ROOT}/src/tests/js_oracle/lib/puzzlescript_trace.js
    ${PUZZLESCRIPT_REPO_ROOT}/src/tests/js_oracle/lib/puzzlescript_node_env.js
    ${SMOKE_TRACE_INPUTS}
)

add_custom_target(generate_native_smoke_ir DEPENDS ${SMOKE_IR} ${SMOKE_TRACE})
add_dependencies(puzzlescript_cpp generate_native_smoke_ir)

add_test(
  NAME puzzlescript_cpp_run_ir_smoke
  COMMAND puzzlescript_cpp run-ir ${SMOKE_IR}
)

add_test(
  NAME puzzlescript_cpp_bench_ir_smoke
  COMMAND puzzlescript_cpp bench-ir ${SMOKE_IR} --iterations 64 --threads 2
)

add_test(
  NAME puzzlescript_cpp_run_smoke
  COMMAND puzzlescript_cpp run ${PUZZLESCRIPT_REPO_ROOT}/src/demo/sokoban_basic.txt --level 0 --seed native-source-smoke --settle-again
)

add_executable(puzzlescript_cpp_player_api_tests
  tests/player_api_tests.cpp
)

target_link_libraries(puzzlescript_cpp_player_api_tests
  PRIVATE
    puzzlescript_native
    puzzlescript_compiler
)

target_include_directories(puzzlescript_cpp_player_api_tests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_dependencies(puzzlescript_cpp puzzlescript_cpp_player_api_tests)

add_test(
  NAME puzzlescript_cpp_player_api_tests
  COMMAND puzzlescript_cpp_player_api_tests
)

if(TARGET PkgConfig::SDL2)
  add_test(
    NAME puzzlescript_cpp_player_sdl_dummy_smoke
    COMMAND puzzlescript_cpp play ${PUZZLESCRIPT_REPO_ROOT}/src/demo/sokoban_basic.txt
  )
  set_tests_properties(puzzlescript_cpp_player_sdl_dummy_smoke
    PROPERTIES
      ENVIRONMENT "SDL_VIDEODRIVER=dummy;SDL_AUDIODRIVER=dummy;PS_PLAYER_FRAME_LIMIT=2"
      TIMEOUT 15
  )
endif()

add_test(
  NAME puzzlescript_cpp_step_smoke
  COMMAND puzzlescript_cpp step ${PUZZLESCRIPT_REPO_ROOT}/src/demo/sokoban_basic.txt right --level 0 --seed native-step-smoke --settle-again
)
set_tests_properties(puzzlescript_cpp_step_smoke PROPERTIES PASS_REGULAR_EXPRESSION "step\\[0\\] token=right changed=1")

add_test(
  NAME puzzlescript_solver_smoke
  COMMAND ${NODE_EXECUTABLE}
          ${PUZZLESCRIPT_REPO_ROOT}/src/tests/run_solver_smoke_assert.js
          $<TARGET_FILE:puzzlescript_solver>
          ${PUZZLESCRIPT_REPO_ROOT}/src/tests/solver_smoke_tests
          --timeout-ms 1000
)
set_tests_properties(puzzlescript_solver_smoke PROPERTIES TIMEOUT 30)

add_test(
  NAME puzzlescript_solver_determinism_smoke
  COMMAND ${NODE_EXECUTABLE}
          ${PUZZLESCRIPT_REPO_ROOT}/src/tests/run_solver_determinism.js
          $<TARGET_FILE:puzzlescript_solver>
          ${PUZZLESCRIPT_REPO_ROOT}/src/tests/solver_smoke_tests
          --runs 5
          --timeout-ms 1000
)
set_tests_properties(puzzlescript_solver_determinism_smoke PROPERTIES TIMEOUT 60)

add_test(
  NAME puzzlescript_solver_js_parity_smoke
  COMMAND ${NODE_EXECUTABLE}
          ${PUZZLESCRIPT_REPO_ROOT}/src/tests/run_solver_parity_smoke.js
          $<TARGET_FILE:puzzlescript_solver>
          ${PUZZLESCRIPT_REPO_ROOT}/src/tests/solver_smoke_tests
)
set_tests_properties(puzzlescript_solver_js_parity_smoke PROPERTIES TIMEOUT 30)

add_test(
  NAME puzzlescript_generator_smoke
  COMMAND ${NODE_EXECUTABLE}
          ${PUZZLESCRIPT_REPO_ROOT}/src/tests/run_generator_smoke.js
          $<TARGET_FILE:puzzlescript_generator>
          ${PUZZLESCRIPT_REPO_ROOT}/src/demo/sokoban_basic.txt
)
set_tests_properties(puzzlescript_generator_smoke PROPERTIES TIMEOUT 30)

add_test(
  NAME puzzlescript_rule_plan_parity
  COMMAND ${NODE_EXECUTABLE}
          ${PUZZLESCRIPT_REPO_ROOT}/src/tests/run_rule_plan_parity.js
          ${PUZZLESCRIPT_REPO_ROOT}/src/tests/resources/testdata.js
          --cli $<TARGET_FILE:puzzlescript_cpp>
          --artifacts-dir ${CMAKE_CURRENT_BINARY_DIR}/rule_plan_parity_testdata
          --progress-every 0
)
set_tests_properties(puzzlescript_rule_plan_parity PROPERTIES TIMEOUT 180)

add_test(
  NAME puzzlescript_cpp_parser_state_smoke
  COMMAND puzzlescript_cpp compile ${PUZZLESCRIPT_REPO_ROOT}/src/demo/sokoban_basic.txt --emit-parser-state
)
set_tests_properties(puzzlescript_cpp_parser_state_smoke PROPERTIES PASS_REGULAR_EXPRESSION "\"parser_state\"")

add_test(
  NAME puzzlescript_cpp_parser_state_diff_smoke
  COMMAND bash ${PUZZLESCRIPT_REPO_ROOT}/scripts/diff_parser_state_against_js.sh ${PUZZLESCRIPT_REPO_ROOT}/src/demo/sokoban_basic.txt
)

add_test(
  NAME puzzlescript_cpp_parser_diagnostics_diff_smoke
  COMMAND bash ${PUZZLESCRIPT_REPO_ROOT}/scripts/diff_diagnostics_against_js.sh ${PUZZLESCRIPT_REPO_ROOT}/src/demo/sokoban_basic.txt
)

add_test(
  NAME puzzlescript_cpp_diff_trace_smoke
  COMMAND puzzlescript_cpp diff-trace ${SMOKE_IR} ${SMOKE_TRACE}
)

add_test(
  NAME puzzlescript_cpp_diff_trace_smoke_source
  COMMAND puzzlescript_cpp diff-trace ${PUZZLESCRIPT_REPO_ROOT}/src/demo/sokoban_basic.txt --level 0 --seed native-trace-smoke --inputs-file ${SMOKE_TRACE_INPUTS}
)

add_test(
  NAME puzzlescript_cpp_diff_trace_right_smoke
  COMMAND puzzlescript_cpp diff-trace ${PUZZLESCRIPT_REPO_ROOT}/src/demo/sokoban_basic.txt --level 0 --seed native-trace-right --inputs-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/right_trace_inputs.json
)

add_test(
  NAME puzzlescript_cpp_diff_trace_push_smoke
  COMMAND puzzlescript_cpp diff-trace ${PUZZLESCRIPT_REPO_ROOT}/src/demo/sokoban_basic.txt --level 0 --seed native-trace-push --inputs-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/push_trace_inputs.json
)

if(PS_MASK_WORD_BITS STREQUAL "32")
  add_test(
    NAME puzzlescript_cpp_js_parity_data_manifest
    COMMAND ${NODE_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_fixture_check.js
            ${PUZZLESCRIPT_REPO_ROOT}/src/tests/js_oracle/export_native_fixtures.js
            $<TARGET_FILE:puzzlescript_cpp>
            ${CMAKE_CURRENT_BINARY_DIR}/generated-fixtures
  )

  set_tests_properties(puzzlescript_cpp_js_parity_data_manifest PROPERTIES TIMEOUT 180)

  add_test(
    NAME puzzlescript_cpp_js_parity_data_trace_sample
    COMMAND ${NODE_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_fixture_check.js
            ${PUZZLESCRIPT_REPO_ROOT}/src/tests/js_oracle/export_native_fixtures.js
            $<TARGET_FILE:puzzlescript_cpp>
            ${CMAKE_CURRENT_BINARY_DIR}/generated-fixtures-trace-sample
            --trace-limit 10
            --trace-allow-failures
  )

  set_tests_properties(puzzlescript_cpp_js_parity_data_trace_sample PROPERTIES TIMEOUT 180)
endif()
