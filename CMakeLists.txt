cmake_minimum_required(VERSION 3.20)

project(PuzzleScriptNative LANGUAGES CXX C)

option(PUZZLESCRIPT_BUILD_NATIVE "Build the native PuzzleScript scaffold." ON)

include(CTest)

if(PUZZLESCRIPT_BUILD_NATIVE)
  add_subdirectory(native)
endif()
