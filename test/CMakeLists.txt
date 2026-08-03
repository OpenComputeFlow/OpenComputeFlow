set(OCF_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(OCF_BINARY_DIR ${PROJECT_BINARY_DIR})
set(OCF_TOOLS_DIR ${LLVM_RUNTIME_OUTPUT_INTDIR})

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(OCF_TEST_DEPENDS ocf-opt FileCheck not)
add_lit_testsuite(
  check-opencomputeflow
  "Running OpenComputeFlow regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${OCF_TEST_DEPENDS}
)
set_target_properties(check-opencomputeflow PROPERTIES FOLDER "Tests")

get_llvm_lit_path(OCF_LIT_BASE_DIR OCF_LIT_FILE_NAME ALLOW_EXTERNAL)
add_test(
  NAME ocf-lit-smoke
  COMMAND
    ${Python3_EXECUTABLE}
    ${OCF_LIT_BASE_DIR}/${OCF_LIT_FILE_NAME}
    -sv
    ${CMAKE_CURRENT_BINARY_DIR}
)
