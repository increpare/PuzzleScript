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

# Release-build flags: -O3, LTO, native codegen, NDEBUG. Tuned for the
# performance target in docs/superpowers/specs/2026-04-22-native-perf-crush-design.md.
# Note: -fno-exceptions / -fno-rtti omitted because core.cpp and cli_main.cpp
# use throw / try / catch (e.g. json::ParseError) pervasively for parse-error
# reporting; enabling those flags would require a large rewrite outside the
# scope of this perf phase.
add_compile_options(
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-DNDEBUG>
    $<$<CONFIG:Release>:-march=native>
    $<$<CONFIG:Release>:-flto>
)

# libc++ safe-mode instrumentation (seen as __annotate_contiguous_container in
# baseline profiles) is a large tax; turn it off on Release only.
add_compile_definitions(
    $<$<CONFIG:Release>:_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_NONE>
)

add_link_options($<$<CONFIG:Release>:-flto>)

find_package(Threads REQUIRED)
find_package(PkgConfig QUIET)
find_program(NODE_EXECUTABLE node REQUIRED)

if(PKG_CONFIG_FOUND)
  pkg_check_modules(SDL2 IMPORTED_TARGET sdl2)
endif()

set(PUZZLESCRIPT_NATIVE_PUBLIC_HEADERS
  include/puzzlescript/puzzlescript.h
)

set(PUZZLESCRIPT_FRONTEND_PUBLIC_HEADERS
  include/puzzlescript/frontend.h
)

set(PUZZLESCRIPT_FRONTEND_SOURCES
  src/c_api_frontend.cpp
  src/frontend/diagnostic.cpp
  src/frontend/parser.cpp
  third_party/utf8proc/utf8proc.c
)

set(PUZZLESCRIPT_NATIVE_SOURCES
  src/c_api.cpp
  src/core.cpp
  src/hash.cpp
  src/json.cpp
  src/simd.cpp
  third_party/simdjson/simdjson.cpp
)

add_library(puzzlescript_native STATIC
  ${PUZZLESCRIPT_NATIVE_PUBLIC_HEADERS}
  ${PUZZLESCRIPT_NATIVE_SOURCES}
)

option(PS_ENABLE_DEV_SERIALIZERS "Enable dev-only frontend serializers and commands" ON)

add_library(puzzlescript_frontend STATIC
  ${PUZZLESCRIPT_FRONTEND_PUBLIC_HEADERS}
  ${PUZZLESCRIPT_FRONTEND_SOURCES}
)

target_include_directories(puzzlescript_frontend
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/utf8proc
)

target_compile_definitions(puzzlescript_frontend
  PRIVATE
    UTF8PROC_STATIC
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

add_executable(ps_cli
  src/cli_main.cpp
  src/diagnostics_parity.cpp
)

target_link_libraries(ps_cli
  PRIVATE
    puzzlescript_native
    puzzlescript_frontend
)

target_compile_definitions(ps_cli
  PRIVATE
    PS_NODE_EXECUTABLE="${NODE_EXECUTABLE}"
    PS_EXPORT_IR_SCRIPT="${CMAKE_SOURCE_DIR}/src/tests/export_ir_json.js"
    PS_EXPORT_TRACE_SCRIPT="${CMAKE_SOURCE_DIR}/src/tests/export_execution_trace.js"
)

if(TARGET PkgConfig::SDL2)
  target_sources(ps_cli PRIVATE src/player_sdl.cpp)
  target_link_libraries(ps_cli PRIVATE PkgConfig::SDL2)
  target_compile_definitions(ps_cli PRIVATE PS_HAVE_SDL2=1)
endif()

enable_testing()

set(SMOKE_IR ${CMAKE_CURRENT_BINARY_DIR}/generated/sokoban_basic_ir.json)
set(SMOKE_TRACE ${CMAKE_CURRENT_BINARY_DIR}/generated/sokoban_basic_trace.json)
set(SMOKE_TRACE_INPUTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/smoke_trace_inputs.json)
add_custom_command(
  OUTPUT ${SMOKE_IR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/generated
  COMMAND node ${CMAKE_SOURCE_DIR}/src/tests/export_ir_json.js
          ${CMAKE_SOURCE_DIR}/src/demo/sokoban_basic.txt
          ${SMOKE_IR}
          --level 0
          --seed native-smoke
          --settle-again
  DEPENDS
    ${CMAKE_SOURCE_DIR}/src/demo/sokoban_basic.txt
    ${CMAKE_SOURCE_DIR}/src/tests/export_ir_json.js
    ${CMAKE_SOURCE_DIR}/src/tests/lib/puzzlescript_ir.js
    ${CMAKE_SOURCE_DIR}/src/tests/lib/puzzlescript_node_env.js
)

add_custom_command(
  OUTPUT ${SMOKE_TRACE}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/generated
  COMMAND node ${CMAKE_SOURCE_DIR}/src/tests/export_execution_trace.js
          ${CMAKE_SOURCE_DIR}/src/demo/sokoban_basic.txt
          ${SMOKE_TRACE}
          --level 0
          --seed native-trace-smoke
          --inputs-file ${SMOKE_TRACE_INPUTS}
  DEPENDS
    ${CMAKE_SOURCE_DIR}/src/demo/sokoban_basic.txt
    ${CMAKE_SOURCE_DIR}/src/tests/export_execution_trace.js
    ${CMAKE_SOURCE_DIR}/src/tests/lib/puzzlescript_trace.js
    ${CMAKE_SOURCE_DIR}/src/tests/lib/puzzlescript_node_env.js
    ${SMOKE_TRACE_INPUTS}
)

add_custom_target(generate_native_smoke_ir DEPENDS ${SMOKE_IR} ${SMOKE_TRACE})
add_dependencies(ps_cli generate_native_smoke_ir)

add_test(
  NAME ps_cli_run_smoke
  COMMAND ps_cli run ${SMOKE_IR}
)

add_test(
  NAME ps_cli_bench_smoke
  COMMAND ps_cli bench ${SMOKE_IR} --iterations 64 --threads 2
)

add_test(
  NAME ps_cli_run_source_smoke
  COMMAND ps_cli run-source ${CMAKE_SOURCE_DIR}/src/demo/sokoban_basic.txt --level 0 --seed native-source-smoke --settle-again
)

add_test(
  NAME ps_cli_step_source_smoke
  COMMAND ps_cli step-source ${CMAKE_SOURCE_DIR}/src/demo/sokoban_basic.txt right --level 0 --seed native-step-smoke --settle-again
)
set_tests_properties(ps_cli_step_source_smoke PROPERTIES PASS_REGULAR_EXPRESSION "step\\[0\\] token=right changed=1")

add_test(
  NAME ps_cli_parser_state_smoke
  COMMAND ps_cli compile-source ${CMAKE_SOURCE_DIR}/src/demo/sokoban_basic.txt --emit-parser-state
)
set_tests_properties(ps_cli_parser_state_smoke PROPERTIES PASS_REGULAR_EXPRESSION "\"parser_state\"")

add_test(
  NAME ps_cli_parser_state_diff_smoke
  COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/diff_parser_state_against_js.sh ${CMAKE_SOURCE_DIR}/src/demo/sokoban_basic.txt
)

add_test(
  NAME ps_cli_parser_diagnostics_diff_smoke
  COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/diff_diagnostics_against_js.sh ${CMAKE_SOURCE_DIR}/src/demo/sokoban_basic.txt
)

add_test(
  NAME ps_cli_diff_trace_smoke
  COMMAND ps_cli diff-trace ${SMOKE_IR} ${SMOKE_TRACE}
)

add_test(
  NAME ps_cli_diff_trace_source_smoke
  COMMAND ps_cli diff-trace-source ${CMAKE_SOURCE_DIR}/src/demo/sokoban_basic.txt --level 0 --seed native-trace-smoke --inputs-file ${SMOKE_TRACE_INPUTS}
)

add_test(
  NAME ps_cli_diff_trace_source_right_smoke
  COMMAND ps_cli diff-trace-source ${CMAKE_SOURCE_DIR}/src/demo/sokoban_basic.txt --level 0 --seed native-trace-right --inputs-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/right_trace_inputs.json
)

add_test(
  NAME ps_cli_diff_trace_source_push_smoke
  COMMAND ps_cli diff-trace-source ${CMAKE_SOURCE_DIR}/src/demo/sokoban_basic.txt --level 0 --seed native-trace-push --inputs-file ${CMAKE_CURRENT_SOURCE_DIR}/tests/push_trace_inputs.json
)

add_test(
  NAME ps_cli_fixture_manifest
  COMMAND ${NODE_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_fixture_check.js
          ${CMAKE_SOURCE_DIR}/src/tests/export_native_fixtures.js
          $<TARGET_FILE:ps_cli>
          ${CMAKE_CURRENT_BINARY_DIR}/generated-fixtures
)

set_tests_properties(ps_cli_fixture_manifest PROPERTIES TIMEOUT 180)

add_test(
  NAME ps_cli_fixture_trace_sample
  COMMAND ${NODE_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_fixture_check.js
          ${CMAKE_SOURCE_DIR}/src/tests/export_native_fixtures.js
          $<TARGET_FILE:ps_cli>
          ${CMAKE_CURRENT_BINARY_DIR}/generated-fixtures-trace-sample
          --trace-limit 10
          --trace-allow-failures
)

set_tests_properties(ps_cli_fixture_trace_sample PROPERTIES TIMEOUT 180)
