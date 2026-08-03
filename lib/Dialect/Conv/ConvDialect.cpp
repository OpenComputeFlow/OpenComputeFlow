#include "opencomputeflow/Dialect/Conv/IR/ConvDialect.h"
#include "opencomputeflow/Dialect/Conv/IR/ConvOps.h"

using namespace mlir;
using namespace mlir::ocf;

#include "opencomputeflow/Dialect/Conv/IR/ConvOpsDialect.cpp.inc"

void OCFConvDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "opencomputeflow/Dialect/Conv/IR/ConvOps.cpp.inc"
      >();
}
