cmake_minimum_required(VERSION 3.20)

project(OpenComputeFlow VERSION 0.1.0 LANGUAGES CXX)

set(OCF_SCHEMA_VERSION 1 CACHE STRING "OpenComputeFlow contract schema version")
set(OCF_LLVM_VERSION 22.1.8 CACHE STRING "Required LLVM version")

find_package(LLVM ${OCF_LLVM_VERSION} EXACT REQUIRED CONFIG)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

message(STATUS "Using LLVM ${LLVM_PACKAGE_VERSION} from ${LLVM_DIR}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(tools/ocf-contract-check)

include(CTest)
if(BUILD_TESTING)
  add_test(
    NAME ocf-contract-reader
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_cpp_contract_test.py
      --checker $<TARGET_FILE:ocf-contract-check>
      --golden ${CMAKE_CURRENT_SOURCE_DIR}/tests/golden/phase0-fingerprints-v1.json
  )
  set_tests_properties(
    ocf-contract-reader
    PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/tests"
  )
endif()
