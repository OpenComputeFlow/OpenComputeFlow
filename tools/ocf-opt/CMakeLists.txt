add_executable(ocf-opt ocf-opt.cpp)

target_include_directories(
  ocf-opt
  SYSTEM PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
target_compile_definitions(
  ocf-opt
  PRIVATE
    OCF_PROJECT_VERSION="${PROJECT_VERSION}"
    OCF_SCHEMA_VERSION=${OCF_SCHEMA_VERSION}
)
target_link_libraries(
  ocf-opt
  PRIVATE
    MLIRAffineDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRMemRefDialect
    MLIROptLib
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRTransformDialect
    MLIRTransforms
    MLIRVectorDialect
)

if(MSVC)
  target_compile_options(ocf-opt PRIVATE /W4)
else()
  target_compile_options(ocf-opt PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

mlir_check_all_link_libraries(ocf-opt)
