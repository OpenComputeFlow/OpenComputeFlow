#include "opencomputeflow/Dialect/Conv/IR/ConvOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::ocf;

#define GET_OP_CLASSES
#include "opencomputeflow/Dialect/Conv/IR/ConvOps.cpp.inc"

namespace {

LogicalResult verifyVector(ArrayRef<int64_t> values, unsigned expectedSize,
                           StringRef name, Operation *op) {
  if (values.size() != expectedSize)
    return op->emitOpError(name) << " must have " << expectedSize
                                 << " elements";
  return success();
}

} // namespace

LogicalResult Conv2DOp::verify() {
  auto inputType = getInput().getType();
  auto filterType = getFilter().getType();
  auto outputType = getOutput().getType();

  if (!inputType.hasStaticShape() || !filterType.hasStaticShape() ||
      !outputType.hasStaticShape())
    return emitOpError("requires static tensor shapes");

  if (inputType.getRank() != 4 || filterType.getRank() != 4 ||
      outputType.getRank() != 4)
    return emitOpError("requires rank-4 NCHW/OIHW tensors");

  if (!inputType.getElementType().isF32() ||
      !filterType.getElementType().isF32() ||
      !outputType.getElementType().isF32())
    return emitOpError("requires f32 input, filter, and output tensors");

  if (getGroups() != 1)
    return emitOpError("MVP only supports groups = 1");

  if (failed(verifyVector(getStrides(), 2, "strides", *this)) ||
      failed(verifyVector(getPadding(), 4, "padding", *this)) ||
      failed(verifyVector(getDilations(), 2, "dilations", *this)))
    return failure();

  auto strides = getStrides();
  auto padding = getPadding();
  auto dilations = getDilations();
  if (strides[0] <= 0 || strides[1] <= 0)
    return emitOpError("strides must be positive");
  if (dilations[0] != 1 || dilations[1] != 1)
    return emitOpError("MVP only supports dilation = 1");
  if (llvm::any_of(padding, [](int64_t value) { return value < 0; }))
    return emitOpError("padding must be non-negative");

  if (inputType.getShape()[1] != filterType.getShape()[1])
    return emitOpError("input channels must equal filter input channels");

  const int64_t inputHeight = inputType.getShape()[2];
  const int64_t inputWidth = inputType.getShape()[3];
  const int64_t kernelHeight = filterType.getShape()[2];
  const int64_t kernelWidth = filterType.getShape()[3];
  const int64_t outputHeight =
      (inputHeight + padding[0] + padding[1] - kernelHeight) / strides[0] + 1;
  const int64_t outputWidth =
      (inputWidth + padding[2] + padding[3] - kernelWidth) / strides[1] + 1;
  if (outputHeight <= 0 || outputWidth <= 0)
    return emitOpError("padding, kernel, and stride produce an empty output");

  SmallVector<int64_t> expectedShape = {inputType.getShape()[0],
                                        filterType.getShape()[0], outputHeight,
                                        outputWidth};
  if (!llvm::equal(outputType.getShape(), expectedShape))
    return emitOpError("output shape must be [N, O, OH, OW] = [")
           << expectedShape[0] << ", " << expectedShape[1] << ", "
           << expectedShape[2] << ", " << expectedShape[3] << "]";
  return success();
}
