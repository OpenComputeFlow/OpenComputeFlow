add_executable(ocf-contract-check ocf_contract_check.cpp)

target_include_directories(ocf-contract-check SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(ocf-contract-check PRIVATE OCF_SCHEMA_VERSION=${OCF_SCHEMA_VERSION})
separate_arguments(OCF_LLVM_DEFINITIONS NATIVE_COMMAND "${LLVM_DEFINITIONS}")
list(REMOVE_ITEM OCF_LLVM_DEFINITIONS -D_GNU_SOURCE)
target_compile_options(ocf-contract-check PRIVATE ${OCF_LLVM_DEFINITIONS})
target_link_libraries(ocf-contract-check PRIVATE LLVMSupport)

if(MSVC)
  target_compile_options(ocf-contract-check PRIVATE /W4)
else()
  target_compile_options(ocf-contract-check PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()
