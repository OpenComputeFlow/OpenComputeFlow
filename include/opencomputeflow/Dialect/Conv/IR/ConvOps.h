#ifndef OCF_CONV_OPS_H
#define OCF_CONV_OPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "opencomputeflow/Dialect/Conv/IR/ConvOps.h.inc"

#endif // OCF_CONV_OPS_H
