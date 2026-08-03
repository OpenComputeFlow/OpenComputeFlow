add_mlir_dialect_library(MLIROCFConvDialect
  ConvDialect.cpp
  ConvOps.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/opencomputeflow/Dialect/Conv

  DEPENDS
    MLIRConvOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSideEffectInterfaces
)
