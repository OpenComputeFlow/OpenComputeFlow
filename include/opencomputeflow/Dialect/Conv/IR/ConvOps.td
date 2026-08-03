//===- ConvOps.td - OpenComputeFlow Conv operations --------*- tablegen -*-===//

#ifndef OCF_CONV_OPS
#define OCF_CONV_OPS

include "opencomputeflow/Dialect/Conv/IR/ConvDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def OCF_Conv2DOp : OCF_Op<"conv2d", [Pure]> {
  let summary = "Restricted direct f32 Conv2D semantic operation";
  let description = [{
    Describes cross-correlation over static NCHW input and OIHW filter
    tensors. The verifier defines the MVP semantic boundary: f32 tensors,
    groups=1, dilation=1, compile-time stride and non-negative four-sided
    padding.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$filter,
    DenseI64ArrayAttr:$strides,
    DenseI64ArrayAttr:$padding,
    DenseI64ArrayAttr:$dilations,
    I64Attr:$groups
  );
  let results = (outs AnyRankedTensor:$output);
  let hasVerifier = 1;
}

#endif // OCF_CONV_OPS
