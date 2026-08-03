cmake_minimum_required(VERSION 3.20)

project(OpenComputeFlow VERSION 0.1.0 LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "In-source builds are not supported. Use: cmake -S . -B build")
endif()

set(OCF_SCHEMA_VERSION 1 CACHE STRING "OpenComputeFlow contract schema version")
set(OCF_LLVM_VERSION 22.1.8 CACHE STRING "Required LLVM version")
set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for LLVM lit")

find_package(LLVM ${OCF_LLVM_VERSION} EXACT REQUIRED CONFIG)
find_package(MLIR ${OCF_LLVM_VERSION} EXACT REQUIRED CONFIG)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

message(STATUS "Using LLVM ${LLVM_PACKAGE_VERSION} from ${LLVM_DIR}")
message(STATUS "Using MLIR ${MLIR_VERSION} from ${MLIR_DIR}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LLVM_RUNTIME_OUTPUT_INTDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LLVM_LIBRARY_OUTPUT_INTDIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${LLVM_LIBRARY_OUTPUT_INTDIR})

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}" "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include
)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools/ocf-contract-check)
add_subdirectory(tools/ocf-opt)

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
  add_subdirectory(test)
endif()
