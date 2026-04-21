cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)
find_package(PkgConfig QUIET)
find_program(NODE_EXECUTABLE node REQUIRED)

if(PKG_CONFIG_FOUND)
  pkg_check_modules(SDL2 IMPORTED_TARGET sdl2)
endif()

set(PUZZLESCRIPT_NATIVE_PUBLIC_HEADERS
  include/puzzlescript/puzzlescript.h
)

set(PUZZLESCRIPT_NATIVE_SOURCES
  src/c_api.cpp
  src/core.cpp
  src/hash.cpp
  src/json.cpp
  src/simd.cpp
)

add_library(puzzlescript_native STATIC
  ${PUZZLESCRIPT_NATIVE_PUBLIC_HEADERS}
  ${PUZZLESCRIPT_NATIVE_SOURCES}
)

target_include_directories(puzzlescript_native
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(puzzlescript_native
  PUBLIC
    Threads::Threads
)

add_executable(ps_cli
  src/cli_main.cpp
)

target_link_libraries(ps_cli
  PRIVATE
    puzzlescript_native
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
