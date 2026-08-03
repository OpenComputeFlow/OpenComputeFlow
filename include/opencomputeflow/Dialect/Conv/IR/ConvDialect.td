//===- ConvDialect.td - OpenComputeFlow Conv dialect --------*- tablegen -*-===//

#ifndef OCF_CONV_DIALECT
#define OCF_CONV_DIALECT

include "mlir/IR/OpBase.td"

def OCF_Conv_Dialect : Dialect {
  let name = "ocf";
  let summary = "OpenComputeFlow operator semantic dialect";
  let description = [{
    The OCF dialect carries hardware-independent operator semantics before
    mapping and lowering. The MVP currently contains a restricted Conv2D op.
  }];
  let cppNamespace = "::mlir::ocf";
}

class OCF_Op<string mnemonic, list<Trait> traits = []> :
    Op<OCF_Conv_Dialect, mnemonic, traits>;

#endif // OCF_CONV_DIALECT
